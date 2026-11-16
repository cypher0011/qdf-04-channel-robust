# Real Replay Recording Protocol

Replay (play a voice note through a loudspeaker and re-record it) is the weakest condition in Model 4: **16.9 % EER**, compared with ≤ 2.5 % for every codec. The replay in our training is **simulated**. This protocol describes how a small team can collect **real** replay data in a few hours, so that the simulation can be checked and the model fine-tuned.

## 1. Source material
* Use only **public dataset utterances** (ArA-DF-2026 dev: equal numbers of bona fide and spoof, Gulf + Egyptian). **Do not record or replay the voices of private people.**
* 200 utterances (100 real, 100 fake) are enough for a first measurement. Use the list in `aradf_dev_channel_subset.json` so results compare directly with the simulated channels.
* Concatenate them into one playlist with 1 s of silence between files and a 1 kHz, 0.5 s **sync beep** at the start (used to cut the recording back into files).

## 2. Devices (record which ones in `devices.csv`)
| Role | Suggested variety |
|---|---|
| Playback | phone speaker (iPhone + Android), laptop speaker, a Bluetooth speaker |
| Recording | a second phone (built-in voice memo app, **not** WhatsApp), a laptop microphone |

At least **3 playback × 2 recording** devices give 6 combinations.

## 3. Rooms and distances
| Variable | Values |
|---|---|
| Room | small office, living room (majlis), car (engine off), corridor (high reverb) |
| Distance | 0.3 m and 1 m |
| Volume | normal conversational loudness (about 65 dB SPL at 1 m) |
| Background | quiet; optionally an AC unit running (note it) |

## 4. Procedure
1. Set the recorder to 16 kHz or 44.1/48 kHz WAV (or the app's highest quality; note the format).
2. Start recording, wait 2 s, start playback.
3. Stop recording 2 s after the playlist ends.
4. File name: `replay_<playback>_<recorder>_<room>_<distance>.wav`, e.g. `replay_iphone_pixel_majlis_1m.wav`.
5. Repeat for each combination. 200 utterances ≈ 15 min per session.

## 5. Processing
```bash
python scripts/split_replay.py replay_*.wav     # (to write) beep alignment → cut by the known offsets → 16 kHz
python scripts/eval_replay.py                   # (to write) EER of Model 3 vs Model 4 per device/room
```
Labels come from the source playlist, so no manual annotation is needed.

## 6. What we will measure
* EER of Model 3 and Model 4 on **real** replay, against the simulated 37.3 % / 16.9 %.
* Which variable hurts most (device, room, distance).
* Whether fine-tuning the head on half of the real replay data (split by **room**, not by file) improves the other half.

## 7. Ethics
Only public research corpora are played back. Recordings are stored locally and used only to train and evaluate the detector. No person is recorded without consent.
