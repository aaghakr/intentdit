# Data layout

Datasets and model weights are not distributed in this repository. Expected
paths are rooted at `data/`:

```text
data/
  dataset/
    pku/split/
      train/{inpaint,saliency,saliency_sub,intent_map}/
      val/{inpaint,saliency,saliency_sub,intent_map}/
      test_anno/{inpaint,saliency,saliency_sub,intent_map}/
      test_unanno/{inpaint,saliency,saliency_sub,intent_map}/
      csv/
    cgl/split/
      train/{inpaint,saliency,saliency_sub,intent_map}/
      val/{inpaint,saliency,saliency_sub,intent_map}/
      test_anno/{inpaint,saliency,saliency_sub,intent_map}/
      test_unanno/{inpaint,saliency,saliency_sub,intent_map}/
      csv/
  model_weights/
    intent_map/
    saliency_detection/
  checkpoints/{pku,cgl}/
  output/
```

CSV filenames are specified in `code/configs/*.yaml`. Keep official benchmark
splits unchanged and record checksums for any generated prompt CSVs.

Reconstruct missing derived inputs with:

```bash
PATH_PROFILE=server bash scripts/paper/prepare_missing_inputs.sh
```

Oracle prompt files are deterministically derived from annotations. The generated
`free_form_*_template.csv` files contain blank assignments only: their prompts must
be written independently without access to ground-truth boxes, then saved without
the `_template` suffix. Do not substitute oracle prompts for this evaluation.
