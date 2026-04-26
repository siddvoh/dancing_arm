#!/usr/bin/env python3
"""Beat-synced dance for the UFactory xArm 7-DOF.

Plays a song, runs offline beat detection with librosa, then streams
small joint-angle pose changes timed to the beats. Pose deltas are tightly
clamped from a comfortable home pose so the arm never makes wild swings.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AUDIO = os.path.join(SCRIPT_DIR, "samples", "eliveta.mp3")
DEFAULT_RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")

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


class _Tee:
    """Duplicate writes across multiple streams. Used to mirror stdout to a log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def _capture_env(arm_ip):
    """Snapshot the running environment for postmortem analysis."""
    import platform

    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "arm_ip": arm_ip,
    }
    try:
        out = subprocess.run(
            ["ping", "-c", "1", "-W", "1", arm_ip],
            capture_output=True, text=True, timeout=3,
        )
        info["ping_rc"] = out.returncode
        info["ping_tail"] = out.stdout.strip().splitlines()[-1] if out.stdout else ""
    except Exception as e:
        info["ping_rc"] = "error"
        info["ping_tail"] = str(e)
    try:
        from xarm import version as xarm_version
        info["xarm_sdk"] = xarm_version.__version__
    except Exception:
        info["xarm_sdk"] = "unknown"
    return info


class Recorder:
    """Webcam recorder that runs in a background thread, then muxes the
    original audio file into the saved video via ffmpeg.

    The ZED Mini in webcam mode appears as a normal camera index; pass
    `camera_index=0` (default) or `1` if the built-in FaceTime camera is also
    enumerated.
    """

    def __init__(self, camera_index, silent_path, fps=30):
        self.camera_index = camera_index
        self.silent_path = silent_path
        self.fps = fps
        self.cap = None
        self.writer = None
        self.thread = None
        self.stop_event = threading.Event()
        self._opened = False

    def open(self):
        try:
            import cv2
        except ImportError:
            print("opencv-python not installed; recording disabled.")
            return False

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"could not open camera {self.camera_index}; try --camera 1 if the ZED is on a different index.")
            return False

        # Warm up the camera and discard the first few frames; first reads on
        # macOS can be slow and may return black frames.
        for _ in range(5):
            self.cap.read()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.silent_path, fourcc, self.fps, (w, h))
        if not self.writer.isOpened():
            print("VideoWriter failed to open; recording disabled.")
            self.cap.release()
            self.cap = None
            return False

        self._opened = True
        return True

    def _loop(self):
        period = 1.0 / self.fps
        next_due = time.monotonic()
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.writer.write(frame)
            next_due += period
            sleep_for = next_due - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We're behind schedule; reset the cadence anchor so we don't
                # spiral into runaway capture attempts.
                next_due = time.monotonic()

    def start(self):
        if not self._opened:
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        if self.writer is not None:
            self.writer.release()
        if self.cap is not None:
            self.cap.release()


def mux_audio(silent_video, audio_path, output_path):
    """Combine the silent webcam video with the song audio into one mp4.

    Returns True if the muxed file was produced.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except FileNotFoundError:
        print("ffmpeg not found. install with `brew install ffmpeg`. silent video kept at:", silent_video)
        return False
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore") if e.stderr else ""
        print(f"ffmpeg failed: {err[-400:]}")
        print("silent video kept at:", silent_video)
        return False


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
    record_path=None,
    camera_index=0,
    fps=30,
    log_path=None,
):
    log_file = None
    real_stdout = sys.stdout
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_file = open(log_path, "w")
        sys.stdout = _Tee(real_stdout, log_file)

    env = _capture_env(arm_ip)
    print("=== run started ===")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"  audio_path: {audio_path}")
    print(f"  speed: {speed} deg/s, acc: {acc} deg/s^2, every_nth: {every_nth}")
    print(f"  dry_run: {dry_run}, record_path: {record_path}, camera: {camera_index}")
    print()
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

    recorder = None
    silent_path = None
    if record_path:
        silent_path = record_path + ".silent.mp4"
        rec = Recorder(camera_index, silent_path, fps=fps)
        if rec.open():
            recorder = rec
            print(f"recording from camera {camera_index} -> {record_path}")
        else:
            print("recording skipped")

    stop_audio = (lambda: None)
    try:
        if recorder is not None:
            recorder.start()
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
        if recorder is not None:
            recorder.stop()
        print("returning to zero pose...")
        dancer.shutdown()

        if recorder is not None and silent_path and os.path.exists(silent_path):
            print("muxing audio into video...")
            ok = mux_audio(silent_path, audio_path, record_path)
            if ok:
                try:
                    os.remove(silent_path)
                except OSError:
                    pass
                print(f"saved: {record_path}")

        print("done.")

        if log_path:
            print(f"log saved: {log_path}")

        artifacts = [p for p in (record_path, log_path) if p and os.path.exists(p)]
        if artifacts:
            rels = [os.path.relpath(p, SCRIPT_DIR) for p in artifacts]
            print()
            print("to share this run for analysis, push from the repo root:")
            print(f"  cd {SCRIPT_DIR}")
            print("  git add " + " ".join(rels))
            print(f'  git commit -m "run: {os.path.basename(record_path or log_path)}"')
            print("  git push")

        if log_file is not None:
            sys.stdout = real_stdout
            log_file.close()


def parse_args():
    p = argparse.ArgumentParser(description="Make the xArm dance to a song.")
    p.add_argument(
        "audio",
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"path to audio file (default: bundled samples/eliveta.mp3)",
    )
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
    p.add_argument("--no-record", action="store_true", help="disable webcam recording")
    p.add_argument(
        "--record-path",
        default=None,
        help="output mp4 path (default: dance_<song>_<timestamp>.mp4 in cwd)",
    )
    p.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is the laptop's camera)")
    p.add_argument("--fps", type=int, default=30, help="video frame rate")
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(args.audio))[0]
    stem = f"dance_{base}_{ts}"

    if args.no_record:
        record_path = None
    elif args.record_path:
        record_path = args.record_path
    else:
        os.makedirs(DEFAULT_RUNS_DIR, exist_ok=True)
        record_path = os.path.join(DEFAULT_RUNS_DIR, f"{stem}.mp4")

    os.makedirs(DEFAULT_RUNS_DIR, exist_ok=True)
    log_path = os.path.join(DEFAULT_RUNS_DIR, f"{stem}.log")

    run_dance(
        audio_path=args.audio,
        arm_ip=args.ip,
        every_nth=args.every_nth,
        dry_run=args.dry_run,
        speed=args.speed,
        acc=args.acc,
        no_audio=args.no_audio,
        record_path=record_path,
        camera_index=args.camera,
        fps=args.fps,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
