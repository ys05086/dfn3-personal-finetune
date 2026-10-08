"""Fine-tune the public DeepFilterNet3 on your own voice and your own room noise, export it as the ONNX
archive libDF and the `deep-filter` tools load, and compare it with the public model on held-out audio.

Why: the public model is trained on the DNS-challenge data, so whispers, breaths and quiet speech often
look like noise to it, and some everyday noises (a fan, a particular keyboard) are not removed well.
Starting from its checkpoint keeps its general noise removal while it learns your voice and your noise.

Recipe: clean voice takes mixed with your noise takes at -5..20 dB SNR, voice levels spread over 30 dB,
15% noise-only and 10% voice-only batches; DeepFilterNet's compressed spectral loss (gamma 0.3) with
under-estimates (removed voice) weighted 2x; AdamW lr 1e-4 with cosine decay; BatchNorm statistics frozen.
The last quarter of every take is held out for the comparison.

  python finetune.py [--voice data/voice] [--noise data/noise] [--out out] [--steps 2000]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import torch

from common import RATE, ensure_base, fresh_state, load_folder, split

SEGMENT = 4 * RATE


def mixer(clean, noises, seed):
    weights = np.array([len(x) for x in clean], float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)

    def crop(x):
        if len(x) <= SEGMENT:
            return np.pad(x, (0, SEGMENT - len(x)))
        start = int(rng.integers(0, len(x) - SEGMENT))
        return x[start:start + SEGMENT]

    def sample():
        c = crop(clean[rng.choice(len(clean), p=weights)]) * 10 ** (rng.uniform(-25, 5) / 20)
        if rng.random() < 0.15:
            c = np.zeros(SEGMENT, np.float32)  # noise alone must come out silent
        n = crop(noises[rng.integers(len(noises))])
        if rng.random() < 0.1:
            n = np.zeros(SEGMENT, np.float32)  # clean voice must come out untouched
        else:
            active = np.abs(c) > 1e-4
            ref = np.mean(c[active] ** 2) if active.mean() > 0.05 else 10 ** (rng.uniform(-60, -30) / 10)
            n = n * np.sqrt(ref / ((np.mean(n ** 2) + 1e-12) * 10 ** (rng.uniform(-5, 20) / 10)))
        return c.astype(np.float32), n.astype(np.float32)

    return sample


def train(base, clean, noises, steps, batch, lr, device, seed):
    from df.enhance import df_features, init_df
    from df.loss import SpectralLoss
    model, _, _ = init_df(str(base), log_level="ERROR", log_file=None)
    model = model.to(device)
    loss_fn = SpectralLoss(gamma=0.3, factor_magnitude=1000, factor_complex=1000, factor_under=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()  # a few minutes of one room would skew the running statistics
    torch.manual_seed(seed)
    sample = mixer(clean, noises, seed)
    for step in range(steps):
        pairs = [sample() for _ in range(batch)]
        c = torch.tensor(np.stack([p[0] for p in pairs]))
        mix = c + torch.tensor(np.stack([p[1] for p in pairs]))
        spec, erb_feat, spec_feat = df_features(mix, fresh_state(), 96, device=device)
        target, _, _ = df_features(c, fresh_state(), 96, device=device)
        loss = loss_fn(model(spec, erb_feat, spec_feat)[0], target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step + 1:5d}/{steps}  loss {loss.item():.4f}", flush=True)
    return model.eval()


def save_checkpoint(model, base, out):
    """Same layout as the released model, so df.enhance.init_df and enhance.py load it."""
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy(base / "config.ini", out / "config.ini")
    epoch = max(int(p.name.split("_")[1].split(".")[0]) for p in (base / "checkpoints").glob("model_*.ckpt.best"))
    torch.save(model.state_dict(), out / "checkpoints" / f"model_{epoch + 1}.ckpt.best")


def export_onnx(model, base, archive):
    """The encoder and the two decoders as ONNX graphs in the archive layout libDF reads (the same as the
    official DeepFilterNet3_onnx.tar.gz), traced with the TorchScript exporter."""
    import onnx
    from df.enhance import df_features
    work = archive.parent / "tmp" / "export"
    work.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).cpu().eval()
    spec, feat_erb, feat_spec = df_features(torch.randn(1, RATE), fresh_state(), 96, device="cpu")
    feat_spec = feat_spec.transpose(1, 4).squeeze(4)

    def export(module, inputs, names, outputs, axes, path):
        torch.onnx.export(module, inputs, str(path), input_names=names, output_names=outputs, dynamic_axes=axes,
                          opset_version=14, keep_initializers_as_inputs=False, dynamo=False)

    with torch.no_grad():
        e0, e1, e2, e3, emb, c0, lsnr = model.enc(feat_erb, feat_spec)
        export(model.enc, (feat_erb, feat_spec), ["feat_erb", "feat_spec"], ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
               {"feat_erb": {2: "S"}, "feat_spec": {2: "S"}, "e0": {2: "S"}, "e1": {2: "S"}, "e2": {2: "S"},
                "e3": {2: "S"}, "emb": {1: "S"}, "c0": {2: "S"}, "lsnr": {1: "S"}}, work / "enc.onnx")
        export(model.erb_dec, (emb.clone(), e3, e2, e1, e0), ["emb", "e3", "e2", "e1", "e0"], ["m"],
               {"emb": {1: "S"}, "e3": {2: "S"}, "e2": {2: "S"}, "e1": {2: "S"}, "e0": {2: "S"}, "m": {2: "S"}},
               work / "erb_dec.onnx")
        export(model.df_dec, (emb.clone(), c0), ["emb", "c0"], ["coefs"],
               {"emb": {1: "S"}, "c0": {2: "S"}, "coefs": {1: "S"}}, work / "df_dec.onnx")
    for name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx"):
        onnx.checker.check_model(str(work / name))
    shutil.copy(base / "config.ini", work / "config.ini")
    with tarfile.open(archive, mode="w:gz") as tar:
        for name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"):
            tar.add(work / name, arcname=f"tmp/export/{name}")
    shutil.rmtree(archive.parent / "tmp")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def frame_db(x, hop=480):
    n = len(x) // hop
    return 10 * np.log10(np.mean(np.asarray(x[:n * hop], np.float64).reshape(n, hop) ** 2, axis=1) + 1e-20)


def si_sdr(estimate, reference):
    estimate, reference = estimate - estimate.mean(), reference - reference.mean()
    target = (estimate @ reference) / (reference @ reference + 1e-12) * reference
    return float(10 * np.log10((target @ target + 1e-12) / ((estimate - target) @ (estimate - target) + 1e-12)))


def compare(models, voices, noises, snr=10.0):
    """Per voice take x noise take at `snr` dB: voice frames cut by more than 10 dB (lower keeps more of
    the voice) and SI-SDR; per noise take alone: the level change (more negative removes more)."""
    rows = {}
    for vname, v in voices.items():
        fv = frame_db(v)
        audible = fv > np.percentile(fv[fv > -110], 10) + 10 if (fv > -110).any() else np.ones_like(fv, bool)
        for nname, n in noises.items():
            n = np.resize(n, len(v))
            active = np.abs(v) > 1e-4
            n = (n * np.sqrt(np.mean(v[active] ** 2) / ((np.mean(n ** 2) + 1e-12) * 10 ** (snr / 10)))).astype(np.float32)
            mix = (v + n).astype(np.float32)
            row = {"si_sdr_input": si_sdr(mix.astype(np.float64), v.astype(np.float64))}
            for label, run in models.items():
                y = run(mix)[:len(v)]
                fo = frame_db(y)[:len(fv)]
                row[f"voice_cut_{label}"] = float(np.mean((fo - fv[:len(fo)])[audible[:len(fo)]] < -10))
                row[f"si_sdr_{label}"] = si_sdr(y.astype(np.float64), v[:len(y)].astype(np.float64))
            rows[f"{vname} + {nname}"] = row
    noise_rows = {}
    for nname, n in noises.items():
        noise_rows[nname] = {label: float(10 * np.log10((np.sum(run(n.astype(np.float32)).astype(np.float64) ** 2) + 1e-20)
                                                        / (np.sum(n.astype(np.float64) ** 2) + 1e-20)))
                             for label, run in models.items()}
    return {"snr_db": snr, "voice_plus_noise": rows, "noise_alone_db": noise_rows}


REFERENCE_DBFS = -26.0  # a typical speech level; quiet mic signals are also compared at this level


def active_dbfs(x):
    active = x[np.abs(x) > 1e-4]
    return 10 * np.log10(np.mean(active.astype(np.float64) ** 2) + 1e-20) if active.size else -120.0


def print_report(report, title):
    print(f"\n{title}: voice + noise at {report['snr_db']:.0f} dB SNR (held-out audio)")
    print(f"{'':28s} {'voice cut':>19s}   {'SI-SDR dB':>23s}")
    print(f"{'':28s} {'public':>9s} {'tuned':>9s}   {'input':>7s} {'public':>7s} {'tuned':>7s}")
    for name, r in report["voice_plus_noise"].items():
        print(f"{name:28s} {r['voice_cut_public']:9.1%} {r['voice_cut_tuned']:9.1%}   "
              f"{r['si_sdr_input']:7.1f} {r['si_sdr_public']:7.1f} {r['si_sdr_tuned']:7.1f}")
    print("\nnoise alone: level change (dB), more negative removes more")
    for name, r in report["noise_alone_db"].items():
        print(f"{name:28s} {r['public']:9.1f} {r['tuned']:9.1f}")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", type=Path, default=here / "data" / "voice", help="folder of clean voice takes")
    parser.add_argument("--noise", type=Path, default=here / "data" / "noise", help="folder of noise-only takes")
    parser.add_argument("--out", type=Path, default=here / "out")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-only", action="store_true", help="compare the model already in --out, no training")
    args = parser.parse_args()
    from df.enhance import enhance, init_df

    voices, noises = load_folder(args.voice), load_folder(args.noise)
    for name, x in list(noises.items()):
        if len(x) < RATE or np.sqrt(np.mean(x ** 2)) < 1e-4:  # under -80 dBFS: a mic gate wrote silence
            print(f"skipping noise take {name!r}: shorter than 1 s or silent")
            del noises[name]
    if not noises:
        raise SystemExit("no usable noise takes")
    clean = [split(x)[0] for x in voices.values()]
    noise_train = [split(x)[0] for x in noises.values()]
    held_voice = {k: split(x)[1] for k, x in voices.items() if len(split(x)[1]) >= RATE}
    held_noise = {k: split(x)[1] for k, x in noises.items() if len(split(x)[1]) >= RATE // 2}
    print(f"voice: {sum(map(len, voices.values())) / RATE:.0f} s in {len(voices)} takes, "
          f"noise: {sum(map(len, noises.values())) / RATE:.0f} s in {len(noises)} takes")
    base = ensure_base()
    if args.eval_only:
        tuned, _, _ = init_df(str(args.out / "DeepFilterNet3_personal"), log_level="ERROR", log_file=None)
        tuned = tuned.to(args.device).eval()
    else:
        tuned = train(base, clean, noise_train, args.steps, args.batch, args.lr, args.device, args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(tuned, base, args.out / "DeepFilterNet3_personal")
        digest = export_onnx(tuned, base, args.out / "DeepFilterNet3_personal_onnx.tar.gz")
        print(f"\nsaved {args.out / 'DeepFilterNet3_personal'} (PyTorch) and "
              f"{args.out / 'DeepFilterNet3_personal_onnx.tar.gz'} (ONNX, sha256 {digest[:12]}...)")

    public, _, _ = init_df(str(base), log_level="ERROR", log_file=None)
    public = public.to(args.device).eval()

    def runner(model):
        def run(x):
            with torch.no_grad():
                return enhance(model, fresh_state(), torch.from_numpy(x).view(1, -1))[0].cpu().numpy()
        return run

    if held_voice and held_noise:
        models = {"public": runner(public), "tuned": runner(tuned)}
        reports = {"as_recorded": compare(models, held_voice, held_noise)}
        print_report(reports["as_recorded"], "as recorded")
        # DeepFilterNet is level-sensitive: on a quiet signal the public model removes much more voice. So the
        # comparison is repeated as if the mic gain were raised until the loudest voice take sits at -26 dBFS.
        gain_db = REFERENCE_DBFS - max(active_dbfs(x) for x in voices.values())
        if gain_db > 6:
            g = 10 ** (gain_db / 20)
            key = f"gain_{gain_db:+.0f}_db"
            reports[key] = compare(models, {k: x * g for k, x in held_voice.items()},
                                   {k: x * g for k, x in held_noise.items()})
            print_report(reports[key], f"mic gain {gain_db:+.0f} dB (loudest take at {REFERENCE_DBFS:.0f} dBFS)")
        (args.out / "report.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    else:
        print("takes too short for a held-out comparison (needs 4 s of voice and 2 s of noise per take)")


if __name__ == "__main__":
    main()
