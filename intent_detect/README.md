# Placement-suitability predictor

This module contains the U-Net/MiT-B1 intent-map predictor adapted from the
density-conditioned layout prior used by DensityLayout and PosterO. Generated
maps and model checkpoints are intentionally excluded from Git.

## Preprocess supervision masks

```bash
cd intent_detect
python preprocess.py --dataset_root ../data/dataset --dataset pku
python preprocess.py --dataset_root ../data/dataset --dataset cgl
```

## Train

`main.py` uses PyTorch distributed execution:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py \
  --dataset_root ../data/dataset \
  --dataset pku \
  --batch_size 128 \
  --learning_rate 1e-6 \
  --model_dm_act none \
  --epoch 101
```

Use `python main.py --help` for the full interface. Archive the checkpoint and
resolved command outside Git.

## Predict one prepared split

The paper pipeline uses the resumable split predictor:

```bash
python intent_detect/predict_split.py \
  --input-dir data/dataset/cgl/split/test_anno/inpaint \
  --output-dir data/dataset/cgl/split/test_anno/intent_map \
  --checkpoint data/model_weights/intent_map/design_intent_cgl_epoch35.pth \
  --device cuda:0
```

Existing outputs are preserved and only absent maps are generated unless
`--overwrite` is passed.
