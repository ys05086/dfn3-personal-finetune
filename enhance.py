"""Denoise a recording with the public and/or the fine-tuned model, to listen to the difference.

  python enhance.py input.wav                       writes input_public.wav and input_tuned.wav
  python enhance.py input.wav --model out/DeepFilterNet3_personal --suffix tuned
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from common import RATE, ensure_base, fresh_state, load


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, action="append",
                        help="checkpoint folder (default: the public model and out/DeepFilterNet3_personal)")
    parser.add_argument("--suffix", action="append", help="output suffix per --model")
    args = parser.parse_args()
    from df.enhance import enhance, init_df
    models = args.model or [ensure_base(), here / "out" / "DeepFilterNet3_personal"]
    suffixes = args.suffix or (["public", "tuned"] if not args.model else [m.name for m in models])
    for folder, suffix in zip(models, suffixes):
        if not (folder / "config.ini").is_file():
            print(f"skipping {folder}: not found (run finetune.py first)")
            continue
        model, _, _ = init_df(str(folder), log_level="ERROR", log_file=None)
        model.eval()
        for path in args.inputs:
            x = torch.from_numpy(load(path)).view(1, -1)
            with torch.no_grad():
                y = enhance(model, fresh_state(), x)[0].cpu().numpy()
            target = path.with_name(f"{path.stem}_{suffix}.wav")
            sf.write(target, y, RATE, subtype="PCM_16")
            print(f"wrote {target}")


if __name__ == "__main__":
    main()
