# DeepFilterNet3 personal fine-tune

**English** | [한국어](README.ko.md)

Fine-tune the public [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) noise suppression model
on a few minutes of **your own voice** and **your own room noise**, then export it in the same ONNX
archive format as the official release.

The public model is trained on the DNS-challenge data. On a particular microphone and room it can treat
whispers, breaths and quiet speech as noise, and miss everyday noises such as a fan or one specific
keyboard. Starting from its checkpoint keeps its general noise removal while it learns which sounds are
your voice and which are your noise. Training takes a few minutes on a recent NVIDIA GPU.

## Install

Python 3.10 or 3.11. Install PyTorch for your GPU first (see [pytorch.org](https://pytorch.org)), e.g.:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps deepfilternet==0.5.6
```

`deepfilternet` is installed without its dependencies because it pins old versions; the ones it needs
are in `requirements.txt`. `common.py` adds a small shim for newer torchaudio versions.

## Use

```bash
python record.py      # guided takes: data/voice (speech, soft, whisper, breath), data/noise (keyboard, mouse, desk, room)
python finetune.py    # about 2000 steps; prints a before/after comparison
python enhance.py some_recording.wav   # writes *_public.wav and *_tuned.wav to listen to
```

Record with the microphone, gain and room you will use the denoiser in. You can also bring your own
takes: put clean voice recordings in one folder and noise-only recordings in another, and pass them with
`--voice` and `--noise`. Any sample rate works (everything is resampled to 48 kHz).

`finetune.py` writes to `out/`:

| File | What |
| --- | --- |
| `DeepFilterNet3_personal/` | PyTorch checkpoint in the release layout (`config.ini` + `checkpoints/`), loadable with `df.enhance.init_df` |
| `DeepFilterNet3_personal_onnx.tar.gz` | encoder and both decoders as ONNX graphs, in the layout of the official `DeepFilterNet3_onnx.tar.gz`, for tools built on libDF that take a model archive |
| `DeepFilterNet3_personal_onnx.tar.gz.sha256` | its SHA-256 |
| `report.json` | the comparison below |

## How it works

- **Data**: each clean voice take is mixed with your noise takes at −5 to 20 dB SNR, with the voice level
  spread over 30 dB so quiet speech is covered. 15% of the examples are noise only (the output should
  be silent) and 10% are clean voice only (the output should be unchanged).
- **Loss**: DeepFilterNet's compressed spectral loss (γ = 0.3) with under-estimates, i.e. removed
  voice, weighted 2×, so the model prefers keeping a breath to deleting it.
- **Optimiser**: AdamW, learning rate 1e-4 with cosine decay, gradient clipping at 1.0, batch 8 of
  4-second segments. BatchNorm statistics are frozen: a few minutes of one room would skew them.
- **Held-out check**: the last quarter of every take is kept out of training. On it, `finetune.py`
  reports, for each voice take mixed with each noise take at 10 dB SNR:
  - **voice cut**: the share of audible voice frames lowered by more than 10 dB (lower keeps more voice)
  - **SI-SDR** of the input, the public model and the tuned model (higher is better)

  and for each noise take alone, the level change (more negative removes more).

  DeepFilterNet is level-sensitive, so when your takes are quiet the comparison is repeated as if the
  mic gain were raised until the loudest voice take sits at −26 dBFS. `--eval-only` re-runs the
  comparison on the model already in `--out`.

## Example results

One speaker: 4.2 minutes of voice (normal, soft, whisper, breath) and 3.8 minutes of noise (desk,
keyboard, mouse), recorded from a quiet microphone signal (normal speech around −44 dBFS). 2000 steps,
about 3.5 minutes on an RTX 5070 Ti. Held-out quarters, voice + noise at 10 dB SNR, averaged
over the three noises.

| Voice | Level | Voice cut, public → tuned | SI-SDR dB, input / public / tuned |
| --- | --- | ---: | ---: |
| normal | as recorded | 97.0% → 8.7% | 4.4 / −8.3 / **20.4** |
| normal | +18 dB gain | 16.9% → 11.3% | 5.1 / 17.7 / **20.7** |
| soft | as recorded | 100% → 0.0% | 3.5 / −13.8 / **19.8** |
| soft | +18 dB gain | 10.7% → 1.8% | 3.8 / 12.7 / **19.7** |
| breath | as recorded | 94.1% → 2.4% | 4.5 / −8.8 / **13.4** |
| breath | +18 dB gain | 100% → 4.4% | 8.4 / −30.1 / **12.5** |
| whisper | as recorded | 95.3% → 29.7% | 5.6 / −7.8 / **−3.2** |
| whisper | +18 dB gain | 76.3% → 58.2% | 7.4 / −7.5 / **−6.3** |

| Noise alone, level change | as recorded: public / tuned | +18 dB gain: public / tuned |
| --- | ---: | ---: |
| desk | −61.8 / −31.1 dB | −46.7 / −40.8 dB |
| keyboard | −68.1 / −58.0 dB | −37.3 / −64.9 dB |
| mouse | −41.6 / −40.1 dB | −54.6 / −44.8 dB |

What this shows:

- **Much of the public model's loss here is a level effect.** On the quiet signal it removed almost all
  of the voice; with the gain raised it keeps most normal and soft speech. Set your mic gain so normal
  speech is around −26 dBFS before recording.
- **Fine-tuning still helps at both levels**: breaths are kept instead of deleted (100% → 4% cut at the
  raised gain), soft speech is kept almost entirely, and SI-SDR on normal and soft speech is 3–7 dB higher.
- **Whispers stay hard.** The tuned model keeps more whisper frames, but both models reshape whispers
  (SI-SDR below the input), and the gain is smaller at a level it was not trained on.
- **It is a trade-off**: some noises are removed less (desk), others more (keyboard at the raised gain).

Every voice × noise pair at both levels: [results/](results/README.md).

These numbers are for one person, one microphone and one room. Run `finetune.py` on your own takes and
listen with `enhance.py` before switching.

## Privacy

Your takes stay on your machine (`data/` and `out/` are git-ignored). The fine-tuned weights are
learned from your voice; think before sharing them. Only train on voices you have the right to use.

## License

MIT, see [LICENSE](LICENSE). DeepFilterNet and its released weights are by Hendrik Schröter and
contributors under MIT or Apache-2.0; fine-tuned weights are derived from them. See [NOTICE.md](NOTICE.md).
