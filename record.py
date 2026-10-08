"""Record the takes the fine-tune needs, one prompt at a time: your voice (normal, soft, whisper, breath)
and the noises of your room (keyboard, mouse, desk, room tone). Takes go to data/voice and data/noise.

  python record.py                 every take
  python record.py whisper mouse   only these (re-record one)
  python record.py --list-devices
"""
from __future__ import annotations

import argparse
from pathlib import Path
import threading

import numpy as np
import soundfile as sf

from common import RATE

DATA = Path(__file__).resolve().parent / "data"
TAKES = [
    ("speech", "voice", 60, "Talk normally for a minute, as you would on a call. Reading a text aloud is fine."),
    ("soft", "voice", 40, "Talk quietly, as if someone is asleep next door."),
    ("whisper", "voice", 60, "Whisper with breath only, no voice. Mix very quiet and slightly louder whispers."),
    ("breath", "voice", 30, "Breathe in and out, sigh, laugh briefly, clear your throat."),
    ("keyboard", "noise", 40, "Don't talk. Type: fast typing, single keys, space/enter bursts, WASD like in a game."),
    ("mouse", "noise", 30, "Don't talk. Click, double-click and scroll the mouse."),
    ("desk", "noise", 20, "Don't talk. Creak the chair, shift around, put a cup down, rustle your clothes."),
    ("room", "noise", 30, "Stay still. Leave fans, air conditioning and the PC running as usual."),
]


def record(name, kind, seconds, guide, device):
    import sounddevice as sd
    print(f"\n[{kind}] {name}: about {seconds} s\n  {guide}")
    input("  Press Enter to start, then Enter again to stop... ")
    chunks, stop = [], threading.Event()

    def callback(data, frames, time, status):
        chunks.append(data[:, 0].copy())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=device, callback=callback):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        while not stop.wait(0.5):
            print(f"\r  recording {sum(map(len, chunks)) / RATE:5.1f} s", end="", flush=True)
    x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    path = DATA / kind / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, RATE, subtype="PCM_16")
    print(f"\r  saved {path.relative_to(DATA.parent)}: {len(x) / RATE:.1f} s, peak {20 * np.log10(peak + 1e-9):.1f} dBFS")
    if peak > 0.99:
        print("  clipped: lower the mic gain and record this one again")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("takes", nargs="*", help="names to record (default: all)")
    parser.add_argument("--device", help="input device name or index (default: the system default)")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    chosen = [t for t in TAKES if not args.takes or t[0] in args.takes]
    unknown = set(args.takes) - {t[0] for t in TAKES}
    if unknown:
        raise SystemExit(f"unknown take(s): {', '.join(sorted(unknown))}")
    print("Use the microphone, gain and room you will use the denoiser with.")
    for take in chosen:
        record(*take, device)


if __name__ == "__main__":
    main()
