# Paper-ready experiment protocol

## Current repository state

The repo contains 13 notebooks across baseline/modified U-Net, U-Net++, ResU-Net, and Attention U-Net. The current data tree has **133 train, 16 validation, and 16 test image-mask pairs** (165 total). This is already close to 8:1:1: train=80.6%, validation=9.7%, test=9.7%.

## Split protocol

Do not split the current train folder blindly into three pieces: validation and test already exist. Preserve the current partition for comparability and create a manifest of exact image/mask pairs. The included `tools/prepare_splits.py` validates pairing and defaults to preserving the existing split. The optional `--existing-val-policy resplit` repartitions only train+validation and leaves test untouched; do so only before final test evaluation and when a predefined protocol requires it.

```bash
python tools/prepare_splits.py --root /raid/home/dgx1725/Warwick_QU_Dataset_Binary_FIXED --write
```

To explicitly rebuild train/validation while preserving test:

```bash
python tools/prepare_splits.py --root /raid/home/dgx1725/Warwick_QU_Dataset_Binary_FIXED --existing-val-policy resplit --write
```

## Training versus final testing

Train only on `train`; choose checkpoints, stopping epochs, and any threshold on `val`; evaluate on `test` only after the model and protocol are frozen. Several existing notebook cells titled `FINAL VALIDATION EVALUATION` iterate over `val_pairs`, so those outputs are validation results, not test results. Some other notebooks already use test data, but the evaluation implementation and naming are inconsistent. Historical notebook outputs should not be treated as evidence that all models were tested under one protocol.

## Final evaluation template

Run `notebooks/Final_Test_Evaluation_Template.ipynb` once for each chosen checkpoint. The template verifies paired test data, computes per-image Dice, IoU, precision, recall, specificity, pixel accuracy, and tolerant boundary F1/precision/recall, then exports per-image and aggregate CSVs, prediction PNGs, a visualization, and evaluation metadata. It does not read optional boundary-ground-truth folders; it derives boundaries from binary masks and predictions.

**Before running:** copy the exact preprocessing and output adapter from the selected training notebook. The template's illustrative default uses RGB images divided by 255 and nearest-neighbour mask resizing. If your training pipeline differs, change the evaluation preprocessing. For U-Net++, point `MODEL_PATH` to the inference model that returns the averaged/single probability mask, not the multi-output training model. Import/register custom layers before loading custom checkpoints.

## Required paper reporting

- Dataset name/version and exact train/validation/test counts; attach `split_manifest.csv`.
- Image dimensions, image/mask interpolation, normalization, mask binarization and augmentation.
- Architecture variant, parameter count, optimizer, loss, learning rate, batch size, maximum epochs and checkpoint-selection metric.
- Evaluation threshold, metric implementation, whether values are macro-averaged or pooled, and boundary tolerance.
- Per-image CSV, aggregate mean ± standard deviation, prediction examples, and software/hardware versions.

Do not invent results or report historical outputs as newly reproduced. Run every selected checkpoint through the same final evaluator.
