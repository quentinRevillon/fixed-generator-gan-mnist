# fixed-generator-gan-mnist

How far a small, imposed MNIST GAN generator can be pushed without changing its architecture, by acting only on the data, the training and the sampling. DataLab team project (Assignment 2), Master IASD, PSL University, November 2026.

<img height="250" src="figures/teaser.png" />

*Same latent vectors: (a) baseline, (b) RpGAN + R1, (c) same run with EMA (submitted), (d) a collapsed baseline run.*

## Results

Mean ± s.d. over 3 seeds, 10,000 images against the MNIST test set. FID<sub>Inc</sub>: Inception FID of the course platform.

| Configuration | FID<sub>Inc</sub> | Precision | Recall | Images kept |
|---|---:|---:|---:|---:|
| Baseline (provided loss) | 38.7 ± 1.2 | 0.47 | 0.51 | 100% |
| Baseline + DRS | 34.4 ± 0.7 | 0.51 | 0.49 | 30% |
| Baseline + EMA | 30.0 ± 0.2 | 0.56 | 0.54 | 100% |
| RpGAN + R1 + EMA | 22.1 ± 1.3 | 0.59 | 0.59 | 100% |
| RpGAN + R1 + EMA + OBRS, recalibrated D | 19.7 ± 0.8 | 0.62 | 0.59 | 33% |
| Submitted (seed 1) | 19.4 | 0.61 | 0.60 | 33% |

Baseline + EMA was trained on an A100, where the same baseline scores 36.6. The data levers did not beat their reference. Details in the [report](docs/report.pdf) and [slides](docs/slides.pdf).

## Quick start

```bash
pip install -r requirements.txt
python generate.py --batch_size 256
python evaluate.py --samples samples --metric prof --reference test
```

The submitted model is in `checkpoints/`. To retrain its generator:

```bash
python train.py --config configs/compare_local/rpgan_r1_ema.yaml --seed 1
```

## Method

- The fully connected generator stays unchanged.
- Training: relativistic loss with an R1 penalty (RpGAN + R1, γ = 1) and an EMA of the generator (decay 0.999).
- Sampling: discriminator rejection sampling (DRS) and optimal budgeted rejection sampling (OBRS, K = 3). D's last layer is refit before sampling.
- Data: instance selection of the most typical images, or annealed instance noise on the inputs of D.
- Protocol: one frozen 50k/10k/10k split, 200 epochs, last checkpoint.

## Structure

```
configs/      one YAML per experiment
checkpoints/  submitted G, recalibrated D, sampling config
docs/         report.pdf, slides.pdf, per-lever write-ups (French)
report/       report LaTeX sources
slides/       slides LaTeX sources
scripts/      Slurm launchers, D recalibration
```

## References

1. Y. Huang et al. [The GAN is dead; long live the GAN! A modern GAN baseline](https://arxiv.org/abs/2501.05441). NeurIPS, 2024.
2. Y. Yazıcı et al. [The unusual effectiveness of averaging in GAN training](https://arxiv.org/abs/1806.04498). ICLR, 2019.
3. A. Verine et al. [Optimal budgeted rejection sampling for generative models](https://arxiv.org/abs/2311.00460). AISTATS, 2024.

## Team

Group 3, Master IASD (PSL University): Nicodème Gorge (data lever), Noé Mosca (training lever), Quentin Revillon (baseline protocol, inference lever). Computations on the MesoNet Juliet cluster.
