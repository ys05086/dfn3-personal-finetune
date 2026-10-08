# Example run: full results

Per voice take × noise take, from `finetune.py` on the run described in the README (one speaker, 2000 steps, RTX 5070 Ti). Held-out last quarter of every take, voice + noise at 10 dB SNR. Raw numbers: [`example_report.json`](example_report.json).

- **voice cut**: share of audible voice frames lowered by more than 10 dB (lower keeps more voice)
- **SI-SDR**: scale-invariant signal-to-distortion ratio against the clean voice (higher is better)
- **noise alone**: level change when only noise goes in (more negative removes more)

## As recorded (normal speech around −44 dBFS)

| Voice + noise | Voice cut, public | Voice cut, tuned | SI-SDR input | SI-SDR public | SI-SDR tuned |
| --- | ---: | ---: | ---: | ---: | ---: |
| breath + desk | 85.5% | 3.4% | 4.5 | -9.5 | 14.9 |
| breath + keyboard | 99.7% | 2.8% | 4.4 | -13.1 | 11.3 |
| breath + mouse | 97.2% | 1.1% | 4.5 | -3.8 | 14.0 |
| soft + desk | 100.0% | 0.0% | 3.5 | -21.8 | 22.0 |
| soft + keyboard | 100.0% | 0.0% | 3.5 | -8.3 | 19.5 |
| soft + mouse | 100.0% | 0.0% | 3.5 | -11.3 | 17.8 |
| speech + desk | 97.0% | 8.2% | 4.4 | -10.2 | 20.2 |
| speech + keyboard | 100.0% | 11.2% | 4.4 | -6.4 | 21.8 |
| speech + mouse | 94.0% | 6.7% | 4.4 | -8.3 | 19.3 |
| whisper + desk | 95.3% | 14.5% | 5.6 | -7.8 | -1.0 |
| whisper + keyboard | 95.3% | 54.4% | 5.6 | -7.9 | -2.6 |
| whisper + mouse | 95.4% | 20.1% | 5.6 | -7.8 | -6.1 |

| Noise alone | Public | Tuned |
| --- | ---: | ---: |
| desk | -61.8 dB | -31.1 dB |
| keyboard | -68.1 dB | -58.0 dB |
| mouse | -41.6 dB | -40.1 dB |

## Mic gain +18 dB (loudest voice take at −26 dBFS)

| Voice + noise | Voice cut, public | Voice cut, tuned | SI-SDR input | SI-SDR public | SI-SDR tuned |
| --- | ---: | ---: | ---: | ---: | ---: |
| breath + desk | 100.0% | 5.4% | 8.4 | -26.9 | 13.6 |
| breath + keyboard | 100.0% | 3.7% | 8.4 | -34.7 | 10.6 |
| breath + mouse | 100.0% | 4.2% | 8.4 | -28.6 | 13.2 |
| soft + desk | 10.7% | 0.7% | 3.8 | 11.4 | 22.6 |
| soft + keyboard | 3.4% | 1.3% | 3.9 | 16.5 | 19.3 |
| soft + mouse | 18.1% | 3.4% | 3.8 | 10.1 | 17.3 |
| speech + desk | 14.9% | 8.5% | 5.1 | 16.8 | 20.4 |
| speech + keyboard | 18.5% | 15.3% | 5.1 | 19.0 | 21.9 |
| speech + mouse | 17.4% | 10.0% | 5.1 | 17.4 | 19.7 |
| whisper + desk | 80.9% | 56.0% | 7.4 | -7.7 | -6.0 |
| whisper + keyboard | 65.9% | 62.1% | 7.4 | -7.5 | -6.1 |
| whisper + mouse | 82.0% | 56.5% | 7.4 | -7.4 | -6.9 |

| Noise alone | Public | Tuned |
| --- | ---: | ---: |
| desk | -46.7 dB | -40.8 dB |
| keyboard | -37.3 dB | -64.9 dB |
| mouse | -54.6 dB | -44.8 dB |

Recordings and the fine-tuned weights are not published: they are one person's voice.
