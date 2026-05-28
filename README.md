# CineGen

> CineGen is a research project on **text-conditioned camera trajectory generation for film cinematography**.
> Given a motion caption and a scene logline, CineGen generates a 4×4 c2w camera trajectory consistent with both.

This repo contains the **inference**, **evaluation**, and **visualization** code for the published CineGen model.
Training code and full training data will follow after paper acceptance — see [TODO](#todo).

---

## Pipeline at a glance

```bash
# 1. Clone + install
git clone https://github.com/Jia1018/CineGEN.git
cd CineGEN
pip install -r requirements.txt

# 2. Download checkpoints (~1.5GB) + eval data (~80MB) from HuggingFace Hub
./scripts/download.sh

# 3. Generate trajectories for the val split
python scripts/infer.py \
    --ckpt   checkpoints/cinegen/best.pt \
    --data_root data/cinegen-eval \
    --out_dir results/cinegen-generated

# 4. Reproduce the paper table
python evaluate/eval_attribute_fidelity_paper.py \
    --data_root data/cinegen-eval \
    --clf_dir   checkpoints/clf_paper \
    --gen_dirs  results/cinegen-generated \
    --out_json  results/attribute_fidelity.json

python evaluate/eval_paper_table.py \
    --data_root      data/cinegen-eval \
    --alignment_dsp  checkpoints/align_dirspd_motion/best.pt \
    --clatr_dsp      checkpoints/clatr_dirspd_motion/best.pt \
    --attr_json      results/attribute_fidelity.json \
    --gen_dirs       results/cinegen-generated \
    --out_csv        results/paper_table.csv \
    --out_latex      results/paper_table.tex

# 5. Render a single trajectory (requires Blender 3.6.5)
export BLENDER=/path/to/blender-3.6.5/blender
bash visualize/render_clip.sh <CLIP_ID> results/cinegen-generated results/renders
```

The rest of this README explains each step.

---

## 1. Installation

- Python 3.10+
- PyTorch 2.x with CUDA (any version compatible with your driver)
- Optional: Blender 3.6.5 for trajectory visualization

```bash
git clone https://github.com/Jia1018/CineGEN.git
cd CineGEN
pip install -r requirements.txt
```

---

## 2. Download checkpoints + eval data

Both artifacts live on HuggingFace Hub.

| What | HF Hub repo | Size |
|---|---|---|
| Model + alignment + classifier checkpoints | [`Ziqi1018/CineGen-ckpts`](https://huggingface.co/Ziqi1018/CineGen-ckpts) | ~1.5GB |
| Eval data pack (val split: matrices, captions, loglines, labels) | [`Ziqi1018/CineGen-eval`](https://huggingface.co/datasets/Ziqi1018/CineGen-eval) | ~80MB |

```bash
./scripts/download.sh           # both
./scripts/download.sh --ckpts   # only checkpoints
./scripts/download.sh --data    # only eval data
```

The download script uses `huggingface_hub.snapshot_download` and does not require an HF token for these public repos.

After running, your layout should be:

```
CineGEN/
├── checkpoints/
│   ├── cinegen/best.pt                 # 477MB — main model
│   ├── align_dirspd_motion/best.pt     # 515MB — alignment encoder (F1/FCD/Cov/AlnScore)
│   ├── clatr_dirspd_motion/best.pt     # 195MB — independent CLaTr alignment
│   └── clf_paper/<setting>/...         # 322MB — attribute classifiers
└── data/cinegen-eval/
    ├── index.jsonl                     # per-clip captions + loglines
    ├── matrices/<clip_id>.npz          # real 4×4 c2w GT
    ├── depth/<clip_id>.npy             # depth features (for attribute classifiers)
    ├── clip_movie_mapping.json         # raw labels for attribute eval
    └── held_out_splits.json            # classifier-side splits for clean attribute eval
```

---

## 3. Inference — generate trajectories

```bash
python scripts/infer.py \
    --ckpt        checkpoints/cinegen/best.pt \
    --data_root   data/cinegen-eval \
    --out_dir     results/cinegen-generated \
    --batch_size  32
```

For each val clip, this writes `<out_dir>/<clip_id>.npz` (containing the generated 4×4 c2w matrices) plus a `metadata.jsonl`. The script auto-detects the model config (no-AE / sep-encoded logline / first-pose / dirspd) from the checkpoint.

### Random-unmask ablation

The default sampler picks the next masked positions by lowest hidden-state variance (the variance-guided heuristic). The random-unmask ablation picks them uniformly:

```bash
python scripts/infer_random_mask.py \
    --ckpt      checkpoints/cinegen/best.pt \
    --data_root data/cinegen-eval \
    --out_dir   results/cinegen-randmask
```

---

## 4. Evaluation — reproduce the paper table

The full paper-table eval has **10 metric columns**:

| Group | Columns |
|---|---|
| Trajectory quality | F1, FCD, Cov |
| Text alignment | AlnScore, CLaTr, R@1, MedR |
| Attribute fidelity | Era, Genre, Dir |

Two scripts produce the table:

```bash
# Attribute fidelity (Era / Genre / Dir) — runs the 9 paper-table classifiers
python evaluate/eval_attribute_fidelity_paper.py \
    --data_root data/cinegen-eval \
    --clf_dir   checkpoints/clf_paper \
    --gen_dirs  results/cinegen-generated \
    --out_json  results/attribute_fidelity.json

# Paper table (combines trajectory quality + text alignment + attribute columns)
python evaluate/eval_paper_table.py \
    --data_root      data/cinegen-eval \
    --alignment_dsp  checkpoints/align_dirspd_motion/best.pt \
    --clatr_dsp      checkpoints/clatr_dirspd_motion/best.pt \
    --attr_json      results/attribute_fidelity.json \
    --gen_dirs       results/cinegen-generated \
    --out_csv        results/paper_table.csv \
    --out_latex      results/paper_table.tex
```

Add more `--gen_dirs` paths to evaluate multiple model variants side-by-side (e.g., the random-unmask ablation).

---

## 5. Visualization

Render a single trajectory to a transparent PNG using Blender 3.6.5.

```bash
# 1. Install Blender 3.6.5 (older versions may work but are untested)
# 2. Point BLENDER at the executable
export BLENDER=/path/to/blender-3.6.5/blender

# 3. Render one clip
bash visualize/render_clip.sh <CLIP_ID> results/cinegen-generated results/renders
```

The shell wrapper does two things:
1. `visualize/postprocess.py` reads the generated `<clip_id>.npz`, applies Gaussian smoothing on translation and log-map smoothing on rotation (controlled by `SIGMA` env var, default 3.0), and writes a JSON.
2. `visualize/blender_render.py` (invoked under Blender) reads the JSON and produces a transparent PNG of the camera trajectory.

Tune the smoothing with `SIGMA=<value> bash visualize/render_clip.sh ...` — large values remove per-frame jitter from autoregressive sampling.

---

## Model architecture

CineGen is a **MAR-style diffusion model** with:

- A **Sequencer**: 1-layer Transformer with AdaLN modulation (`d=512, nhead=8, ff=4096`) that contextualizes masked trajectory tokens with text + first-pose conditioning.
- A **Diffuser**: 3-block MLP-AdaLN that denoises individual masked tokens in raw 8-D direction+speed feature space.
- Conditioning is injected **entirely through AdaLN** — text (CLIP ViT-B/32), first-pose (8-D direction+speed), and a separately-encoded scene logline (CLIP) are concatenated in `cond_fusion` and fed into AdaLN modulation in both sequencer and diffuser.
- The published variant uses **IdentityAE** (no autoencoder compression): diffusion operates directly on raw 8-D dirspd features rather than a learned 64-D latent.

See [`cinegen/model.py`](cinegen/model.py) for the implementation.

---

## TODO

- [ ] Release **training code** (`scripts/train.py`, training-loss configs)
- [ ] Release **full training data** (~410GB raw + caption/aspect-extraction pipeline)

Both will be released after paper acceptance.

---

## Citation

If you use CineGen in your work, please cite (preprint forthcoming):

```bibtex
@misc{cinegen,
  title  = {CineGen: text-conditioned camera trajectory generation for film cinematography},
  author = {Zhou, Ziqi and colleagues},
  year   = {2026},
}
```

---

## Acknowledgements

CineGen builds on prior open-source work. We gratefully acknowledge the following projects and datasets:

- [GenDoP](https://github.com/3DTopia/GenDoP) — Blender rendering pipeline (we patched and adapted parts of it for our renderer).
- [The Exceptional Trajectories (ET)](https://github.com/robincourant/the-exceptional-trajectories) — trajectory generation baselines.
- [PulpMotion](https://github.com/robincourant/pulp-motion) — MAR-style architecture inspiration.
- [ShotBench](https://huggingface.co/datasets/Vchitect/ShotBench) — shot-level benchmark data.
- [CineTechBench](https://huggingface.co/datasets/Xinran0906/CineTechBench) — cinematic technique benchmark.
- [VADB](https://huggingface.co/datasets/BestiVictoryLab/VADB) — video-aesthetic dataset.
- [MovieShots](https://movienet.github.io/projects/eccv20shot.html) — shot-type dataset.
- [CondensedMovies](https://github.com/m-bain/CondensedMovies) — condensed-movie clips.

---

## License

This project is released under [CC BY-NC 4.0](LICENSE) — research-only, non-commercial.
The included checkpoints and eval data on HuggingFace Hub are released under the same terms. Source datasets retain their own licenses; see individual links above.
