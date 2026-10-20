# Comparative Study of U-Net-Based Models for Binary Image Segmentation

This repository contains TensorFlow/Keras implementations and experiments comparing **U-Net, ResU-Net, U-Net++, and Attention U-Net** for binary segmentation on the Warwick-QU / GlaS histology dataset.

> **Research note:** The notebooks contain several architectural variants. Results can only be compared fairly when the dataset split, preprocessing, input resolution, training budget, loss, inference rule, and metric aggregation are identical. The values below are transcribed from saved notebook outputs and are reported as individual experiment results, not as a definitive ranking of every architecture.

## Repository contents

| Folder | Contents |
|---|---|
| [UNET_Variants](UNET_Variants/) | Baseline U-Net and three modified variants |
| [ResUNET_Variants](ResUNET_Variants/) | Baseline ResU-Net and three modified variants |
| [UNET++_Variants](UNET%2B%2B_Variants/) | Baseline U-Net++ and three modified variants |
| [Attention_UNET](Attention_UNET/) | Attention U-Net implementation |
| [Warwick_QU_Dataset_Binary_FIXED](Warwick_QU_Dataset_Binary_FIXED/) | Dataset organized into train, validation, and test image/mask folders |

## Objectives

- Establish baseline implementations of common encoder-decoder segmentation architectures.
- Explore changes to skip-feature fusion and convolutional feature extraction.
- Evaluate pixel-region overlap as well as boundary quality.
- Preserve a held-out test split for final evaluation and save per-image predictions and metric summaries.

## Architectures and modifications

The notebooks are the source of truth for each implementation. The following summarizes the intended design families represented in the repository; consult the corresponding notebook for exact layer definitions and tensor shapes.

### 1. U-Net

**Baseline — `UNET_Variants/Basic_Unet.ipynb`**

A symmetric encoder-decoder. The encoder extracts increasingly abstract features through convolution and downsampling; the decoder upsamples features and concatenates encoder skip features to recover spatial detail.

**Modification 1 — `Modification_1_Unet.ipynb`: multi-skip fusion**

Extends feature reuse by bringing additional encoder/intermediate features into decoder fusion rather than relying only on the conventional single skip at a decoder level. The goal is to expose the decoder to richer spatial and semantic information.

**Modification 2 — `Modification_2_Unet.ipynb`: fixed-kernel feature augmentation**

Augments feature maps with responses from fixed image-processing kernels before/alongside feature fusion. Such responses can provide explicit edge or intensity-transition cues; they do not guarantee improved performance and should be evaluated against the baseline.

**Modification 3 — `Modification_3_Unet.ipynb`: learnable-kernel feature augmentation**

Uses trainable kernel-based feature extraction rather than relying exclusively on fixed filters. This lets the model adapt filter responses to the training objective, at the cost of additional parameters and possible overfitting.

### 2. ResU-Net

**Baseline — `ResUNET_Variants/Basic_ResUNET.ipynb`**

Uses residual blocks in the encoder/decoder feature-processing path. Identity or projection shortcuts help information and gradients flow through deeper convolutional blocks.

The three notebooks named `Modification_1_ResUNET.ipynb`, `Modification_2_ResUNET.ipynb`, and `Modification_3_ResUNET.ipynb` explore the corresponding multi-skip, fixed-kernel, and learnable-kernel modification families in a residual U-Net setting. The residual path is the defining difference from plain U-Net: the proposed feature-fusion changes are evaluated within residual blocks rather than being assumed equivalent to a plain convolutional block.

### 3. U-Net++

**Baseline — `UNET++_Variants/Basic_Unet++.ipynb`**

Uses nested, dense skip pathways. Intermediate decoder nodes combine encoder features with earlier decoder nodes at the same scale, progressively reducing the semantic gap between encoder and decoder representations.

**Modification 1 — `Modification_1_UNET++.ipynb`: multi-skip fusion**

Explores additional feature reuse across the nested skip pathways. This changes the information available to a node while retaining U-Net++'s nested topology.

**Modification 2 — `Modification_2_UNet++.ipynb`: fixed-kernel fusion**

Adds fixed-kernel responses to the nested feature-fusion process. The notebook evaluates the modified network using averaged side outputs for inference.

**Modification 3 — `Modification_3_Unet++.ipynb`: learnable-kernel fusion**

Adds a trainable depthwise kernel bank initialized using Sobel-X, Sobel-Y, and Laplacian filters. The notebook's implementation concatenates the resulting responses with the incoming feature maps, allowing optimization to adapt these filters. The U-Net++ implementation also produces multiple side outputs and averages them for inference.

### 4. Attention U-Net

**Implementation — `Attention_UNET/Attention_UNET.ipynb`**

Attention U-Net uses attention gates on skip connections. The gating signal helps suppress irrelevant encoder activations and emphasize spatial regions useful to the decoder. It is an attention-based reference architecture, distinct from the kernel-augmentation variants described above.

## Dataset and preprocessing

The repository includes a binary-mask version of the Warwick-QU / GlaS dataset, organized as:

```text
Warwick_QU_Dataset_Binary_FIXED/
├── train/
│   ├── img/
│   └── mask/
├── val/
│   ├── img/
│   └── mask/
└── test/
    ├── img/
    └── mask/
```

The notebooks pair images with masks by matching filename stems and removing the `_anno` suffix from annotation filenames. The split counts reported by the notebooks are:

| Split | Image-mask pairs |
|---|---:|
| Train | 133 |
| Validation | 16 |
| Test | 16 |
| **Total** | **165** |

Several updated experiment notebooks use a target input size of **256 × 384** (height × width), batch size **1**, and a maximum of **50 epochs**. Verify the configuration cell of each notebook before running it, because settings can differ across experiments. The saved notebook output reports normalized image values and binary masks in at least the U-Net++ learnable-kernel experiment; confirm the preprocessing code in each notebook before assuming every experiment is identical.

## Evaluation metrics

The notebooks report combinations of:

- **Dice coefficient:** overlap between predicted and ground-truth foreground regions.
- **Intersection over Union (IoU):** intersection divided by union.
- **Recall:** fraction of ground-truth foreground pixels recovered.
- **Precision:** fraction of predicted foreground pixels that are correct.
- **Boundary F1, boundary recall, and boundary precision:** boundary quality with a stated tolerance of 2 pixels in the relevant evaluation notebooks.

Pixel overlap and boundary scores measure different properties. A model can have a high Dice score while still producing imprecise contours. The macro-average test scores below are computed across individual images in the corresponding notebook; do not compare them directly with a global pixel-wise metric or a validation score.

## Reported experiment results

These are the macro-average metrics printed in the saved notebook outputs that were available for inspection. Values are shown as fractions from 0 to 1.

| Experiment | IoU | Dice | Recall | Precision | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| U-Net — Modification 1 (multi-skip) | 0.7510 | 0.8523 | 0.8993 | 0.8253 | 0.5930 |
| U-Net++ — Modification 3 (learnable kernels) | 0.7776 | 0.8697 | 0.9128 | 0.8421 | 0.5148 |

**Additional U-Net++ output:** Modification 2 (fixed-kernel) prints a macro-average of IoU **0.7672**, Dice **0.8625**, recall **0.9155**, precision **0.8281**, and boundary F1 **0.4508** in its saved output. That notebook labels this section “MACRO-AVERAGE VALIDATION METRICS,” while other notebooks label comparable output differently; verify the evaluation split in the code before making a direct comparison.

These values are **not a complete leaderboard**: saved outputs for all baselines and variants were not available in a consistent, verifiable form for this summary. No claim is made here that one architecture is universally best. For a defensible comparison, run all models with the same split and preprocessing, use the same test-set evaluator, and report mean ± standard deviation over the same images.

## Running the notebooks

1. Clone or download this repository.
2. Install a compatible Python environment with TensorFlow, NumPy, OpenCV, pandas, and Matplotlib. The saved experiment output shows TensorFlow 2.15.1 in some notebooks; use a compatible version or adjust APIs as needed.
3. Update each notebook's `ROOT_DIR` / dataset path and GPU selection for your machine. Paths such as `/raid/home/dgx1725/` are specific to the original training environment and must be changed locally.
4. Run the notebook cells in order, starting with the configuration and data-pairing cells.
5. Inspect the saved training curves, predicted masks, per-image CSV, and summary CSV. Check whether each reported metric is validation or held-out test performance before recording it.

The notebooks may save model checkpoints, plots, predicted masks, and CSV metrics to experiment-specific output directories configured in their setup cells. Those paths may point outside the repository.

## Reproducibility checklist

For a fair comparison, keep the following identical across model families:

- Train/validation/test split and image-mask pairing.
- Input dimensions, normalization, and augmentation.
- Loss function, optimizer, learning-rate schedule, batch size, and epoch/early-stopping policy.
- Prediction threshold and side-output aggregation rule.
- Test-set metric implementation, boundary tolerance, and macro- versus micro-averaging.
- Random seed and checkpoint-selection policy.

Do not select models or tune thresholds using the held-out test set.

## Limitations

- The dataset is small, so results can vary with initialization and split composition.
- Additional skip paths and kernel branches can increase memory usage and computation.
- Strong region-overlap scores do not necessarily imply accurate boundaries.
- Notebook output may reflect a particular GPU, filesystem path, software version, and checkpoint; reproducing it requires matching the relevant setup.
- The experiment results currently documented here are partial and should not be presented as a complete controlled benchmark.

## Citation and dataset acknowledgement

If you use the Warwick-QU / GlaS data or build on these experiments in academic work, cite the original dataset and the relevant architecture papers. This repository is an experimental comparison and does not claim ownership of the original datasets or architecture designs.
