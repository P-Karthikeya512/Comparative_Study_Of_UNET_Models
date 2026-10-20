# Comparative Study of U-Net-Based Models for Binary Segmentation

This repository contains TensorFlow/Keras notebooks for comparing **U-Net, ResU-Net, U-Net++, and Attention U-Net** on binary segmentation using the Warwick-QU / GlaS histology dataset.

The notebooks include the baseline architectures, three modification variants for U-Net, ResU-Net, and U-Net++, and one Attention U-Net implementation. They also save training curves, checkpoints, predicted masks, and per-image evaluation metrics in experiment-specific output directories.

> **How to interpret the results:** The scores below are transcribed from the saved notebook outputs. Evaluation headings are not fully consistent: several U-Net++ output cells call the summary “validation metrics” even though the code/output refers to 16 test images and writes files with `test_metrics` names. Treat the table as a record of the notebooks' reported numbers, not yet as a perfectly controlled leaderboard. Run all notebooks with one identical evaluator and verify each split before making scientific claims.

## Repository structure

| Folder | Contents |
|---|---|
| `UNET_Variants/` | Baseline U-Net and Modifications 1–3 |
| `ResUNET_Variants/` | Baseline ResU-Net and Modifications 1–3 |
| `UNET++_Variants/` | Baseline U-Net++ and Modifications 1–3 |
| `Attention_UNET/` | Attention U-Net implementation |
| `Warwick_QU_Dataset_Binary_FIXED/` | Binary segmentation images and masks divided into train, validation, and test subsets |

## Dataset

The repository's dataset folder follows this layout:

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

The notebook loaders match images to annotation masks using the filename stem; Warwick-QU annotation names commonly end in `_anno`, which is removed for pairing.

| Split | Matched image-mask pairs |
|---|---:|
| Train | 133 |
| Validation | 16 |
| Test | 16 |
| **Total** | **165** |

Many of the current notebooks use input resolution **256 × 384** (height × width), batch size **1**, and up to **50 epochs**. The exact model setup, loss, optimizer, and checkpoint selection can differ between notebooks. Check the configuration cell at the top of the specific notebook before running it. The configured paths (for example, `/raid/home/dgx1725/...`) refer to the original training environment and must be changed to match your machine.

## Architectures and modifications

### 1. U-Net

**Baseline — `UNET_Variants/Basic_Unet.ipynb`**

A conventional encoder-decoder. Two-convolution blocks extract features, max pooling reduces spatial resolution, transposed convolutions upsample decoder features, and matching encoder features are concatenated through skip connections. The output is a single-channel sigmoid mask for binary segmentation. The notebook uses a combined binary-cross-entropy and Dice-style loss.

**Modification 1 — `UNET_Variants/Modification_1_Unet.ipynb`: long multi-skip fusion**

The decoder receives additional encoder features rather than only the feature map at the matching scale:

- At the first decoder stage, the skip is `c4`.
- At the next stage, `c3` and `c4` are fused.
- At the next stage, `c2` and `c3` are fused.
- At the final stage, `c1` and `c2` are fused.

The added features are projected with 1×1 convolutions and resized to match the decoder's spatial dimensions, then concatenated. This is intended to expose the decoder to multi-scale features. The implementation uses bilinear upsampling followed by convolution in its modified decoder.

**Modification 2 — `UNET_Variants/Modification_2_Unet.ipynb`: fixed-kernel skip fusion**

A non-trainable depthwise filter bank applies **Sobel-X, Sobel-Y, and Laplacian** kernels to encoder features. Each transformed feature is concatenated with the original encoder feature and compressed with a 1×1 convolution. At each decoder stage, the decoder feature is fused with both the original skip and its fixed-kernel branch.

**Modification 3 — `UNET_Variants/Modification_3_Unet.ipynb`: learnable-kernel skip fusion**

This follows the fixed-kernel design but uses a trainable depthwise kernel layer initialized with Sobel-X, Sobel-Y, and Laplacian filters. The weights can be optimized during training. The decoder fuses the upsampled decoder representation, the original encoder skip, and the learned-filter response.

### 2. ResU-Net

**Baseline — `ResUNET_Variants/Basic_ResUNET.ipynb`**

Uses residual convolutional blocks instead of plain double-convolution blocks. The implementation includes batch normalization and ReLU activations, with a projected shortcut when the spatial stride or channel count changes. The encoder, bridge, and decoder are connected through ordinary U-shaped skip connections; the decoder concatenates the matching encoder feature and processes it with a residual block.

**Modification 1 — `ResUNET_Variants/Modification_1_ResUNET.ipynb`: long multi-skip connections**

The residual encoder is retained, but decoder stages receive additional encoder levels. The documented mapping in the notebook is:

- `d3 ← [e3]`
- `d2 ← [e2, e3]`
- `d1 ← [e1, e2]`
- `d0 ← [e0, e1]`

Each skip is projected and resized to the decoder feature's shape before concatenation, and the merged features are passed through a residual block.

**Modification 2 — `ResUNET_Variants/Modification_2_ResUNET.ipynb`: fixed-kernel skip branches**

Keeps the ResU-Net encoder and bridge, adding non-trainable Sobel-X, Sobel-Y, and Laplacian depthwise responses at the encoder skip levels. Each response is fused with the raw skip; the decoder concatenates its upsampled feature with the raw and filtered skip features, then uses a residual block.

**Modification 3 — `ResUNET_Variants/Modification_3_ResUNET.ipynb`: learnable-kernel skip branches**

Replaces the fixed filter bank with a trainable depthwise bank initialized from Sobel-X, Sobel-Y, and Laplacian filters. The raw and filtered skip features are concatenated and compressed by a 1×1 convolution. Decoder stages fuse the decoder feature, original skip, and learned-filter branch, then use residual processing.

### 3. U-Net++

**Baseline — `UNET++_Variants/Basic_Unet++.ipynb`**

Implements the nested U-Net++ skip topology. A node `X(i,j)` combines the earlier decoder nodes at its scale with an upsampled feature from the next deeper level. The model uses **four deep-supervision side outputs**. Training applies loss to all four outputs, while the inference model averages the four predicted masks.

**Modification 1 — `UNET++_Variants/Modification_1_UNET++.ipynb`: additional direct skip links**

Retains the nested U-Net++ pathways and adds three direct encoder-to-decoder links:

- `X(1,0) → X(0,4)`
- `X(2,0) → X(1,3)`
- `X(3,0) → X(2,2)`

Each direct feature is upsampled and appended to the usual concatenation inputs of its destination node. The model keeps the four deep-supervision outputs and averaged-output inference. This modification is therefore **extra direct skips**, not simply the same multi-skip decoder used by the plain U-Net variant.

**Modification 2 — `UNET++_Variants/Modification_2_UNet++.ipynb`: fixed-kernel enhancement of nested nodes**

For each nested decoder node, the normal U-Net++ input tensors are augmented with responses from the fixed Sobel-X, Sobel-Y, and Laplacian bank. The raw and filtered features are concatenated and compressed with a 1×1 convolution before the node's usual convolution block. Four side outputs and averaged-output inference are retained.

**Modification 3 — `UNET++_Variants/Modification_3_Unet++.ipynb`: learnable-kernel enhancement of nested nodes**

Uses a trainable depthwise kernel bank initialized from Sobel-X, Sobel-Y, and Laplacian filters. In contrast to adding a filter branch only to a single encoder skip, this variant applies the kernel-response augmentation to **each set of incoming tensors used by a nested U-Net++ node**. The merged tensors are compressed by a 1×1 convolution, processed by the usual convolution block, and supervised by four side outputs. Inference averages the four masks.

### 4. Attention U-Net

**Implementation — `Attention_UNET/Attention_UNET.ipynb`**

This folder contains one Attention U-Net implementation, not a separate series of three modifications. The encoder-decoder resembles a standard U-Net, but each skip passes through an additive attention gate before concatenation. The gate projects the encoder skip and decoder gating signal with 1×1 convolutions, combines them with an activation, predicts a sigmoid attention coefficient, and multiplies that coefficient into the encoder feature. The decoder then concatenates the **gated skip** with its upsampled features. The purpose is to suppress less relevant spatial activations while emphasizing features useful for segmentation.

## Reported results

The following metrics are macro-averages over the 16 images in the notebook's evaluation summary. Values are fractions from 0 to 1. Boundary metrics use the notebook's stated tolerance of 2 pixels.

| Architecture | Variant | IoU | Dice | Recall | Precision | Boundary F1 |
|---|---|---:|---:|---:|---:|---:|
| U-Net | Baseline | 0.7400 | 0.8358 | 0.8316 | 0.8753 | 0.5713 |
| U-Net | Mod. 1 — Multi-skip | 0.7510 | 0.8523 | 0.8993 | 0.8253 | 0.5930 |
| U-Net | Mod. 2 — Fixed kernels | 0.7380 | 0.8430 | 0.8922 | 0.8207 | 0.5784 |
| U-Net | Mod. 3 — Learnable kernels | 0.7429 | 0.8466 | 0.8998 | 0.8180 | 0.5904 |
| ResU-Net | Baseline | 0.7319 | 0.8291 | 0.9002 | 0.7889 | 0.4728 |
| ResU-Net | Mod. 1 — Multi-skip | 0.7672 | 0.8616 | 0.8706 | 0.8677 | 0.4795 |
| ResU-Net | Mod. 2 — Fixed kernels | 0.7625 | 0.8569 | 0.8694 | 0.8698 | 0.4401 |
| ResU-Net | Mod. 3 — Learnable kernels | 0.7793 | 0.8703 | 0.8710 | 0.8794 | 0.4750 |
| U-Net++ | Baseline | 0.7742 | 0.8671 | 0.9072 | 0.8447 | 0.5765 |
| U-Net++ | Mod. 1 — Extra direct skips | 0.7723 | 0.8673 | 0.9144 | 0.8375 | 0.5567 |
| U-Net++ | Mod. 2 — Fixed kernels | 0.7672 | 0.8625 | 0.9155 | 0.8281 | 0.4508 |
| U-Net++ | Mod. 3 — Learnable kernels | 0.7776 | 0.8697 | 0.9128 | 0.8421 | 0.5148 |
| Attention U-Net | Attention gates | 0.7381 | 0.8384 | 0.9306 | 0.7816 | 0.5864 |

Additional per-image metrics and summary files are printed/saved by the notebooks. The listed scores are the saved notebook outputs, not newly rerun results.

### What the reported scores suggest

- **Highest reported IoU and Dice:** ResU-Net Modification 3 (0.7793 IoU, 0.8703 Dice) among the values listed.
- **Highest reported Boundary F1:** U-Net Modification 1 (0.5930), narrowly above the Attention U-Net output (0.5864).
- **Highest reported recall:** Attention U-Net (0.9306), paired with a lower precision of 0.7816, indicating a different recall/precision trade-off.
- Fixed/learnable filters do not improve every metric in every architecture. Region overlap and boundary alignment should be considered separately.

These observations are descriptive, not proof of statistical superiority. Some experiments have different model definitions, losses, training setups, or labels on the evaluation output. For a publishable comparison, re-run all models with the same exact split, preprocessing, loss/optimizer policy, prediction threshold, and evaluation script; report macro-average results over the same held-out images.

## Evaluation metrics

- **IoU (Intersection over Union):** foreground intersection divided by foreground union.
- **Dice coefficient:** overlap between the predicted and reference masks.
- **Recall:** proportion of reference foreground pixels recovered by the model.
- **Precision:** proportion of predicted foreground pixels that belong to the reference foreground.
- **Boundary F1 / Recall / Precision:** boundary quality computed from binary masks, with a tolerance of 2 pixels in the evaluation code.

Macro-averaging computes a metric separately for each image and then averages image scores. It is not the same as pooling all pixels across the dataset before computing a score.

## Running the notebooks

1. Clone or download the repository.
2. Create an environment with a TensorFlow/Keras version compatible with the notebook, plus NumPy, OpenCV, pandas, and Matplotlib. Saved output in several notebooks shows TensorFlow 2.15.1.
3. Update the dataset root, output directory, and GPU ID in the configuration cell. The notebooks were configured for a particular DGX/Linux environment; paths and GPU indices must be changed locally.
4. Run cells in order from configuration and image-mask pairing through training and evaluation.
5. Review the metric summary, per-image CSV, training curves, prediction images, and model checkpoint created by the notebook.

The maximum epoch setting is commonly 50, but early stopping can terminate training earlier. U-Net++ variants use multiple side outputs for deep supervision and average the side-output masks for inference. The other models produce a single sigmoid mask or, in the ResU-Net implementation, logits handled by its corresponding loss/inference code.

## Reproducibility checklist

For an apples-to-apples experiment, hold constant:

- The exact train/validation/test file lists and image-mask pairing.
- Image size, image normalization, mask binarization, and data augmentation.
- Loss function, optimizer, learning-rate schedule, batch size, random seed, and stopping/checkpoint rules.
- Prediction threshold, model output/side-output aggregation, boundary extraction, and boundary tolerance.
- Metric code and macro- versus micro-averaging.

Do not use the test set to choose a checkpoint or tune a threshold. Keep the test set for final evaluation only.

## Limitations

- The dataset has 165 matched pairs, with 16 samples per validation and test split, so estimates can be sensitive to individual images.
- Added skip connections and kernel branches increase fusion complexity and may increase memory consumption.
- Strong foreground overlap does not guarantee accurate object boundaries.
- Saved notebook outputs depend on the training run and environment. The reported scores should not be interpreted as reproducible until the notebooks are rerun with a fully matched protocol.
- The repository includes a baseline Attention U-Net; it does not currently provide equivalent Modification 1–3 attention variants for direct comparison.

## Acknowledgement

Please cite the original Warwick-QU / GlaS dataset and the relevant architecture papers when using this dataset or these model designs in academic work. This repository is an experimental implementation and does not claim ownership of the dataset or the original architectures.
