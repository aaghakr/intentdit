# Setup

## Environment

```bash
conda create -n intentdit python=3.10 -y
conda activate intentdit
pip install -r requirements.txt
```

The intent-map predictor additionally requires `segmentation-models-pytorch`.
The user-study application requires Flask. Both are included in the root
requirements file.

## Path profiles

- `local` (default): resolves project-owned paths from the checkout.
- `server`: resolves them from
  `/home/viplab/Aagha/intent_aware_layout_generation`.

Entry points accept `--path-profile local|server`. Helper scripts can use:

```bash
export INTENTDIT_PATH_PROFILE=server
```

## Validation

```bash
cd code
python -m unittest discover -s tests -v
python validate_configs.py
```

