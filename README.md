# IntentDiT

IntentDiT is a research implementation of prompt-controllable, content-aware
poster layout generation. The model combines a continuous diffusion
transformer with saliency/placement-suitability guidance and optional BERT
token conditioning.

This repository contains the source code, experiment scripts and evaluation
tools behind the IntentDiT manuscript. Datasets, checkpoints, generated
outputs, experiment logs, and user-study responses are intentionally excluded.

## Repository layout

```text
code/           Core model, data pipeline, training, evaluation, and tests
intent_detect/  Placement-suitability (intent-map) predictor
scripts/paper/  Scripts that reproduce every table and figure of the paper
user_study/     Reproducible human-evaluation web application
docs/           Setup, data layout, and reproducibility documentation
data/           Local-only datasets and weights (ignored by Git)
experiments/    Local-only generated results (ignored by Git)
```

## Installation

```bash
git clone https://github.com/aaghakr/intentdit.git
cd intentdit
```

Python 3.9 or 3.10 and a CUDA-capable PyTorch installation are recommended.

```bash
conda create -n intentdit python=3.10 -y
conda activate intentdit
pip install -r requirements.txt
```

Prepare data as described in [docs/DATA.md](docs/DATA.md), then validate the
checkout:

```bash
cd code
python -m unittest discover -s tests -v
```

## Training

```bash
cd code
python scripts/train.py \
  --dataset pku \
  --task uncond \
  --v_encoder vit \
  --spatial_guidance 2 \
  --text_control \
  --seed 1 \
  --experiment_name pku_vit_both_text_seed1
```

On the project server, append `--path-profile server`. Local execution is the
default and resolves paths from the current checkout.

## Evaluation

```bash
cd code
python scripts/test.py \
  --dataset pku \
  --anno anno \
  --task uncond \
  --v_encoder vit \
  --spatial_guidance 2 \
  --text_control \
  --seed 1 \
  --check_path /path/to/Epoch500_cgbdm_weights.pth \
  --experiment_name pku_vit_both_text_seed1
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the controlled
experiment matrix and reporting rules.

The paper-by-paper evaluation audit and remaining journal experiments are in
[docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md).

The complete executable paper pipeline is documented in
[scripts/paper/README.md](scripts/paper/README.md).

## Paper

The current manuscript is [paper/main.tex](paper/main.tex). Compile it on
Overleaf or with an Elsevier `elsarticle` installation:

```bash
cd paper
latexmk -pdf main.tex
```

## Research status

The journal results must be regenerated with the corrected differentiable
count loss, placement padding/masking, coordinate normalization, and
independent training seeds. Historical checkpoints and metrics are not part of
this repository and should not be treated as final journal evidence.

## License

See [LICENSE](LICENSE).
