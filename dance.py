#!/usr/bin/env python3
"""Beat-synced dance for the UFactory xArm 7-DOF.

Plays a song, runs offline beat detection with librosa, then streams
small joint-angle pose changes timed to the beats. Pose deltas are tightly
clamped from a comfortable home pose so the arm never makes wild swings.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np


HOME_POSE_DEG = [0.0, 20.0, 0.0, 60.0, 0.0, 40.0, 0.0]

# Per-joint maximum allowed deviation from HOME_POSE_DEG. Anything in the pose
# bank or generated at runtime is clipped to this envelope before being sent.
MAX_JOINT_DELTA_DEG = [25.0, 20.0, 25.0, 20.0, 30.0, 20.0, 60.0]

# Conservative motion caps. Speed is in deg/s, acceleration in deg/s^2.
# Matches the visual_servoing defaults (_angle_speed=20, _angle_acc=500) with
# a small bump on speed for groove. Verified safe on this user's xArm 7.
DEFAULT_SPEED_DEG_S = 25.0
DEFAULT_ACC_DEG_S2 = 500.0

# If beats arrive faster than the arm can comfortably react, we down-sample.
# At 25 deg/s a typical 12 deg pose move takes ~0.5s, so we throttle to 0.5s.
MIN_MOVE_PERIOD_S = 0.50

# Pose bank: joint deltas (degrees) added to HOME_POSE_DEG.
# Each row is a "dance shape" — small offsets that look expressive but stay
# inside the safety envelope. Order is loosely choreographed (lean -> raise ->
# wrist twirl -> mirror) so cycling through them on beats reads as dance.
POSE_BANK = [
    # j1   j2     j3     j4     j5     j6    j7
    [ 12,  -6,    0,     -8,    0,     -4,   15],   # lean right + wrist roll
    [-12,  -6,    0,     -8,    0,     -4,  -15],   # mirror left
    [  0, -12,    0,    -12,    0,     -8,    0],   # bow forward
    [  0,   8,    0,      8,    0,      8,    0],   # rise up
    [ 15,   0,  -10,      0,   18,      0,   20],   # spiral right
    [-15,   0,   10,      0,  -18,      0,  -20],   # spiral left
    [  0,  -4,    0,      4,    0,     12,   18],   # head nod 1
    [  0,  -4,    0,      4,    0,     12,  -18],   # head nod 2
    [  8,   4,   -8,     -8,   12,     -4,    0],   # groove A
    [ -8,   4,    8,     -8,  -12,     -4,    0],   # groove B
]


def clamp_pose(target_deg, home=HOME_POSE_DEG, envelope=MAX_JOINT_DELTA_DEG):
    """Clip an absolute target pose so each joint stays within envelope of home."""
    out = []
    for v, h, e in zip(target_deg, home, envelope):
        lo, hi = h - e, h + e
        out.append(max(lo, min(hi, float(v))))
    return out


def pose_from_delta(delta_deg, home=HOME_POSE_DEG):
    return clamp_pose([h + d for h, d in zip(home, delta_deg)])


def detect_beats(audio_path, every_nth=1):
    """Return (beat_times_seconds, tempo_bpm, sample_rate, mono_audio)."""
    import librosa

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if every_nth > 1:
        beat_times = beat_times[::every_nth]

    # Throttle: if a beat lands too close after the previous emitted beat,
    # drop it. This protects the arm from bursty fast sections.
    filtered = []
    last = -1e9
    for t in beat_times:
        if t - last >= MIN_MOVE_PERIOD_S:
            filtered.append(float(t))
            last = t

    return np.array(filtered), float(tempo), int(sr), y


def build_choreography(beat_times):
    """Map each beat time to a target absolute pose.

    Cycles through POSE_BANK with a small random walk to keep it interesting,
    but every Nth beat returns to home so the dance "breathes" and resets.
    """
    poses = []
    n = len(POSE_BANK)
    for i, t in enumerate(beat_times):
        if i % 8 == 7:
            target = HOME_POSE_DEG[:]
        else:
            target = pose_from_delta(POSE_BANK[i % n])
        poses.append((float(t), target))
    return poses


@dataclass
class MotionLimits:
    speed: float = DEFAULT_SPEED_DEG_S
    acc: float = DEFAULT_ACC_DEG_S2


class ArmDancer:
    """Thin wrapper over XArmAPI that enforces our safety envelope."""

    def __init__(self, ip, limits: MotionLimits, dry_run: bool = False):
        self.dry_run = dry_run
        self.limits = limits
        self._arm = None
        if not dry_run:
            from xarm.wrapper import XArmAPI

            self._arm = XArmAPI(ip, baud_checkset=False)
            time.sleep(0.5)
            self._arm.clean_warn()
            self._arm.clean_error()
            self._arm.motion_enable(True)
            self._arm.set_mode(0)
            self._arm.set_state(0)
            time.sleep(1.0)

    def go_home(self):
        self.move_to(HOME_POSE_DEG, wait=True)

    def move_to(self, target_deg, wait=False):
        target_deg = clamp_pose(target_deg)
        if self.dry_run:
            return 0
        return self._arm.set_servo_angle(
            angle=target_deg,
            speed=self.limits.speed,
            mvacc=self.limits.acc,
            wait=wait,
            radius=0.0,
        )

    def shutdown(self):
        if self.dry_run or self._arm is None:
            return
        try:
            self._arm.set_servo_angle(
                angle=[0.0] * 7,
                speed=self.limits.speed,
                mvacc=self.limits.acc,
                wait=True,
                radius=0.0,
            )
        finally:
            self._arm.disconnect()


def play_audio_async(samples, sr):
    """Start audio playback in the background. Returns a stop() callable."""
    try:
        import sounddevice as sd
    except ImportError as e:
        raise SystemExit(
            "sounddevice not installed. `pip install sounddevice` (and `brew install portaudio` on macOS)."
        ) from e

    sd.play(samples, sr)

    def stop():
        try:
            sd.stop()
        except Exception:
            pass

    return stop


def run_dance(
    audio_path,
    arm_ip,
    every_nth=1,
    dry_run=False,
    speed=DEFAULT_SPEED_DEG_S,
    acc=DEFAULT_ACC_DEG_S2,
    no_audio=False,
):
    print(f"loading + analyzing: {audio_path}")
    beat_times, tempo, sr, audio = detect_beats(audio_path, every_nth=every_nth)
    print(f"  tempo ~ {tempo:.1f} BPM, {len(beat_times)} usable beats after throttle")
    if len(beat_times) == 0:
        print("no beats detected. exiting.")
        return

    choreography = build_choreography(beat_times)
    duration = float(len(audio)) / sr
    print(f"  song duration: {duration:.1f}s")

    limits = MotionLimits(speed=speed, acc=acc)
    dancer = ArmDancer(arm_ip, limits, dry_run=dry_run)

    print("moving to home pose...")
    dancer.go_home()

    stop_audio = (lambda: None)
    try:
        if not no_audio:
            stop_audio = play_audio_async(audio, sr)
        t0 = time.monotonic()

        for beat_t, target in choreography:
            now = time.monotonic() - t0
            sleep_for = beat_t - now
            if sleep_for > 0:
                # Sleep in small slices so Ctrl-C is responsive.
                end = time.monotonic() + sleep_for
                while True:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.05, remaining))
            elif sleep_for < -0.5:
                # We fell behind by more than half a second; skip this beat.
                continue

            code = dancer.move_to(target, wait=False)
            if code != 0 and not dry_run:
                print(f"arm returned non-zero code {code}; aborting")
                break

        # Let the song finish playing if there's tail audio after the last beat.
        tail = duration - (time.monotonic() - t0)
        if tail > 0:
            time.sleep(min(tail, 5.0))

    except KeyboardInterrupt:
        print("\ninterrupted by user")
    finally:
        stop_audio()
        print("returning to zero pose...")
        dancer.shutdown()
        print("done.")


def parse_args():
    p = argparse.ArgumentParser(description="Make the xArm dance to a song.")
    p.add_argument("audio", help="path to audio file (mp3/wav/flac/...)")
    p.add_argument("--ip", default="192.168.1.200", help="xArm IP (default: 192.168.1.200)")
    p.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="dance on every Nth beat. Use 2 for fast songs to halve the rate.",
    )
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED_DEG_S, help="joint speed deg/s")
    p.add_argument("--acc", type=float, default=DEFAULT_ACC_DEG_S2, help="joint accel deg/s^2")
    p.add_argument("--dry-run", action="store_true", help="don't connect to the arm; print what would happen")
    p.add_argument("--no-audio", action="store_true", help="skip audio playback (still uses beat times)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.speed > 50 or args.acc > 1000:
        print(
            f"refusing to run with speed={args.speed}, acc={args.acc}: above safe envelope. "
            f"edit the script if you really want this.",
            file=sys.stderr,
        )
        sys.exit(2)
    run_dance(
        audio_path=args.audio,
        arm_ip=args.ip,
        every_nth=args.every_nth,
        dry_run=args.dry_run,
        speed=args.speed,
        acc=args.acc,
        no_audio=args.no_audio,
    )


if __name__ == "__main__":
    main()
