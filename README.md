# dancing_arm

Beat-synced dance for the UFactory xArm 7-DOF. Plays a song, detects beats with librosa, and moves the arm through a small bank of safe joint poses on each beat.

## Hardware

- UFactory xArm (7 joints) on the LAN, default IP `192.168.1.200`.
- The script uses joint-angle control in degrees via `xarm-python-sdk`.

## Install

```bash
# macOS audio backend
brew install portaudio

# python deps
pip install -r requirements.txt
```

## Run

Always start with a dry run to inspect the choreography without moving the arm:

```bash
python dance.py /path/to/song.mp3 --dry-run
```

When you're happy, connect to the arm:

```bash
python dance.py /path/to/song.mp3 --ip 192.168.1.200
```

For high-BPM songs, dance on every other beat:

```bash
python dance.py /path/to/song.mp3 --every-nth 2
```

## Safety design

This script is intentionally conservative. The arm will not be sent into wild swings.

- All target poses are clipped to a per-joint envelope around a comfortable home pose (`HOME_POSE_DEG` and `MAX_JOINT_DELTA_DEG` in `dance.py`). Envelope verified inside published xArm 7 joint limits with wide margin.
- Default joint speed is `25 deg/s`, default acceleration `500 deg/s^2` — the same range used in the user's `visual_servoing_for_suction_grippers` repo. Hard refuses values above `50 deg/s` / `1000 deg/s^2`.
- Beat throttle (`MIN_MOVE_PERIOD_S = 0.50s`) drops beats that arrive too close together, so fast songs don't whip the arm.
- On startup the arm moves to home before any beat command is sent. On shutdown it returns to zero pose.
- `--dry-run` runs the full pipeline (audio analysis, scheduling, audio playback) without ever connecting to the arm.

## Tweaking the dance

- Add or replace poses in `POSE_BANK` in `dance.py`. Each row is a delta in degrees added to `HOME_POSE_DEG`. Stay within `MAX_JOINT_DELTA_DEG`.
- Change `HOME_POSE_DEG` to face the dance toward your audience.
- Increase `--every-nth` for slower, calmer dancing on busy songs.
