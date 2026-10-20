# Comparative Study of U-Net-Based Models for Binary Segmentation

This repository compares four encoder–decoder segmentation architectures on the Warwick-QU / GlaS histology dataset:

- **U-Net**
- **ResU-Net**
- **U-Net++**
- **Attention U-Net**

Alongside the baseline networks, the project explores three kinds of modifications for U-Net, ResU-Net, and U-Net++: multi-scale/direct skip fusion, fixed edge-filter features, and learnable edge-filter features. The notebooks contain TensorFlow/Keras training and evaluation pipelines, saved outputs, and metric summaries.

> **Important about the results:** The table below records the metrics already saved in the notebooks; the experiments have not been rerun as part of this README update. Some notebook output labels/splits are inconsistent, and experiment configurations may differ. Treat these as reported results rather than a definitive, fully controlled ranking. For a publication-quality comparison, evaluate all checkpoints with the same split and evaluator.

## Contents

| Directory | Description |
|---|---|
| [`UNET_Variants/`](UNET_Variants/) | Baseline U-Net and three modifications |
| [`ResUNET_Variants/`](ResUNET_Variants/) | Baseline ResU-Net and three modifications |
| [`UNET++_Variants/`](UNET%2B%2B_Variants/) | Baseline U-Net++ and three modifications |
| [`Attention_UNET/`](Attention_UNET/) | Attention U-Net |
| [`Warwick_QU_Dataset_Binary_FIXED/`](Warwick_QU_Dataset_Binary_FIXED/) | Images and binary masks in train/validation/test splits |

## Dataset

The included dataset is organized into image/mask pairs:

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

The notebook loaders pair image and annotation by filename stem, removing the trailing `_anno` from mask names when necessary.

| Split | Matched image–mask pairs |
|---|---:|
| Train | 133 |
| Validation | 16 |
| Test | 16 |
| **Total** | **165** |

A common configuration is **256 × 384 (height × width)**, batch size **1**, and up to **50 epochs**. Verify the config cell in each notebook: image size, loss, optimizer, callbacks, output paths, and other settings may vary. The notebooks were originally configured for a DGX/Linux environment with paths such as `/raid/home/dgx1725/...`; update these paths and the GPU selection for your environment.

## Architecture details and modifications

### 1. U-Net

**Baseline — [`Basic_Unet.ipynb`](UNET_Variants/Basic_Unet.ipynb)**

A standard encoder–decoder with convolutional blocks, max-pooling downsampling, upsampling in the decoder, and same-resolution encoder skip connections. The output is a foreground probability mask for binary segmentation.

**Modification 1 — [`Modification_1_Unet.ipynb`](UNET_Variants/Modification_1_Unet.ipynb): multi-scale skip fusion**

Adds features from adjacent encoder scales to decoder stages instead of relying only on the corresponding single-scale skip. The additional features are projected (using 1×1 convolutions where needed), resized to the decoder feature-map resolution, and concatenated. This is intended to make multiple scales available during reconstruction.

**Modification 2 — [`Modification_2_Unet.ipynb`](UNET_Variants/Modification_2_Unet.ipynb): fixed-kernel skip enhancement**

Adds a non-trainable depthwise filter bank initialized with **Sobel-X, Sobel-Y, and Laplacian** filters to encoder skip features. Raw features and filter responses are fused and compressed with a 1×1 convolution, and the decoder receives both the original skip information and the filtered representation.

**Modification 3 — [`Modification_3_Unet.ipynb`](UNET_Variants/Modification_3_Unet.ipynb): learnable-kernel skip enhancement**

Uses a trainable depthwise filter bank initialized from Sobel-X, Sobel-Y, and Laplacian kernels. Unlike Modification 2, the filters can adapt during backpropagation. The decoder fuses the original encoder features with the learned filter responses.

### 2. ResU-Net

**Baseline — [`Basic_ResUNET.ipynb`](ResUNET_Variants/Basic_ResUNET.ipynb)**

Uses residual convolutional blocks in the encoder, bridge, and decoder. Shortcut paths help preserve information across convolutional transformations; projected shortcuts are used when dimensions need to match. The U-shaped skip connections still carry encoder features to the decoder.

**Modification 1 — [`Modification_1_ResUNET.ipynb`](ResUNET_Variants/Modification_1_ResUNET.ipynb): multi-scale skip fusion**

Keeps the residual encoder/decoder design but supplies additional encoder-level features to selected decoder stages. These features are resized/projected to a common spatial resolution, concatenated with the decoder features, and processed through residual blocks.

**Modification 2 — [`Modification_2_ResUNET.ipynb`](ResUNET_Variants/Modification_2_ResUNET.ipynb): fixed-kernel skip enhancement**

Adds fixed, non-trainable Sobel-X, Sobel-Y, and Laplacian depthwise responses to encoder skip features. The decoder fuses the upsampled decoder representation with the raw skip and filtered features, then applies residual processing.

**Modification 3 — [`Modification_3_ResUNET.ipynb`](ResUNET_Variants/Modification_3_ResUNET.ipynb): learnable-kernel skip enhancement**

Replaces fixed filtering with a trainable depthwise kernel bank initialized from the same edge filters. The skip and learned-filter representations are fused before residual decoder processing.

### 3. U-Net++

**Baseline — [`Basic_Unet++.ipynb`](UNET%2B%2B_Variants/Basic_Unet%2B%2B.ipynb)**

Uses nested skip pathways: each node combines earlier features at the same scale with upsampled features from a deeper layer. The notebook uses **four deep-supervision side outputs** during training and averages the four predicted masks for inference.

**Modification 1 — [`Modification_1_UNET++.ipynb`](UNET%2B%2B_Variants/Modification_1_UNET%2B%2B.ipynb): additional direct skips**

Retains the nested U-Net++ pathways and adds direct encoder-to-decoder links, documented in the notebook as:

- `X(1,0) → X(0,4)`
- `X(2,0) → X(1,3)`
- `X(3,0) → X(2,2)`

The additional feature maps are resized and appended to the usual input concatenations of their destination nodes. Deep supervision is retained.

**Modification 2 — [`Modification_2_UNet++.ipynb`](UNET%2B%2B_Variants/Modification_2_UNet%2B%2B.ipynb): fixed-kernel feature enhancement**

Augments the inputs used by nested decoder nodes with responses from fixed Sobel-X, Sobel-Y, and Laplacian filters. The raw and filtered representations are merged and compressed before the node’s convolution block. Four side outputs are retained.

**Modification 3 — [`Modification_3_Unet++.ipynb`](UNET%2B%2B_Variants/Modification_3_Unet%2B%2B.ipynb): learnable-kernel feature enhancement**

Uses a trainable depthwise filter bank initialized from Sobel-X, Sobel-Y, and Laplacian filters to augment inputs to nested nodes. A 1×1 convolution compresses the merged representations before the usual node convolution block. The model retains the four deep-supervision outputs and averages them at inference.

### 4. Attention U-Net

**Implementation — [`Attention_UNET.ipynb`](Attention_UNET/Attention_UNET.ipynb)**

This folder contains a baseline Attention U-Net rather than three modification variants. It inserts additive attention gates on the encoder-to-decoder skip connections. The gate uses the encoder skip feature and decoder gating signal to produce spatial attention coefficients, which weight the encoder features before they are concatenated with the decoder representation. The goal is to focus skip information on regions relevant to the segmentation target.

## Reported evaluation results

Scores below are copied from the saved notebook evaluation summaries, which report macro-averages over 16 images. Values range from 0 to 1. Boundary metrics use the reported tolerance of 2 pixels.

| Architecture | Variant | IoU | Dice | Recall | Precision | Boundary F1 |
|---|---|---:|---:|---:|---:|---:|
| U-Net | Baseline | 0.7400 | 0.8358 | 0.8316 | 0.8753 | 0.5713 |
| U-Net | Modification 1 — Multi-skip | 0.7510 | 0.8523 | 0.8993 | 0.8253 | 0.5930 |
| U-Net | Modification 2 — Fixed kernels | 0.7380 | 0.8430 | 0.8922 | 0.8207 | 0.5784 |
| U-Net | Modification 3 — Learnable kernels | 0.7429 | 0.8466 | 0.8998 | 0.8180 | 0.5904 |
| ResU-Net | Baseline | 0.7319 | 0.8291 | 0.9002 | 0.7889 | 0.4728 |
| ResU-Net | Modification 1 — Multi-skip | 0.7672 | 0.8616 | 0.8706 | 0.8677 | 0.4795 |
| ResU-Net | Modification 2 — Fixed kernels | 0.7625 | 0.8569 | 0.8694 | 0.8698 | 0.4401 |
| ResU-Net | Modification 3 — Learnable kernels | **0.7793** | **0.8703** | 0.8710 | 0.8794 | 0.4750 |
| U-Net++ | Baseline | 0.7742 | 0.8671 | 0.9072 | 0.8447 | 0.5765 |
| U-Net++ | Modification 1 — Direct skips | 0.7723 | 0.8673 | 0.9144 | 0.8375 | 0.5567 |
| U-Net++ | Modification 2 — Fixed kernels | 0.7672 | 0.8625 | 0.9155 | 0.8281 | 0.4508 |
| U-Net++ | Modification 3 — Learnable kernels | 0.7776 | 0.8697 | 0.9128 | 0.8421 | 0.5148 |
| Attention U-Net | Attention gates | 0.7381 | 0.8384 | **0.9306** | 0.7816 | 0.5864 |

### Reading the results

- **Highest reported IoU and Dice:** ResU-Net Modification 3, with IoU **0.7793** and Dice **0.8703**.
- **Highest reported Boundary F1:** U-Net Modification 1, with **0.5930**.
- **Highest reported recall:** Attention U-Net, with **0.9306**; its precision is lower (**0.7816**), showing a different precision–recall trade-off.
- The fixed and learnable filter branches do not improve every metric across all architectures. Region overlap and boundary quality are distinct objectives.

These comparisons are descriptive only. The dataset is small, and saved notebook summaries do not establish that every architecture was evaluated under exactly the same conditions. Before treating small differences as meaningful, use one evaluation script and verify split identities, preprocessing, thresholds, and checkpoint selection.

## Metrics

- **IoU (Intersection over Union):** intersection of predicted and ground-truth foreground divided by their union.
- **Dice:** overlap between predicted and ground-truth foreground masks.
- **Recall:** fraction of ground-truth foreground pixels correctly predicted as foreground.
- **Precision:** fraction of predicted foreground pixels that are truly foreground.
- **Boundary F1:** boundary precision/recall combined into an F1 score using the reported 2-pixel tolerance.

Macro-averaging computes the metric for each image and then averages the image-level scores; it differs from pooling all pixels first.

## Environment and execution

The notebooks use Python with **TensorFlow/Keras**, plus common packages including NumPy, OpenCV, pandas, and Matplotlib. Saved notebook outputs show TensorFlow 2.15.1 in at least some runs.

1. Clone or download this repository.
2. Install TensorFlow/Keras and the dependencies imported by the notebook you plan to run.
3. Update dataset and output-directory paths in the initial configuration cell.
4. Set `CUDA_VISIBLE_DEVICES` to a GPU visible in your environment, or adapt the configuration to run on CPU.
5. Run the notebook from top to bottom and inspect the training curves, saved checkpoint, predictions, and evaluation summaries.

The notebooks may stop before the maximum of 50 epochs when callbacks such as early stopping are triggered. Consult each notebook for its actual training configuration rather than assuming all experiments used identical settings.

## Reproducibility checklist

For a fair comparison, keep the following identical across models wherever the architecture permits:

- Train/validation/test file lists and image–mask pairing.
- Image normalization, mask binarization, resolution, and augmentation.
- Optimizer, learning rate, loss, random seed, batch size, and checkpoint/early-stopping rules.
- Prediction threshold and (for U-Net++) side-output aggregation.
- Evaluation code, boundary tolerance, and macro/micro averaging.

Use the validation split for checkpoint selection and tuning; reserve the test split for final evaluation.

## Limitations

- The dataset contains 165 image–mask pairs, with only 16 images in each validation and test split.
- More skip paths and filter branches can increase model complexity and memory use.
- Region-overlap metrics do not fully capture boundary accuracy.
- Saved numbers reflect the notebook runs and should not be considered independently reproduced.
- Only a single Attention U-Net notebook is included; equivalent attention modifications are not provided for all model families.

## Acknowledgement

Please cite the original Warwick-QU / GlaS dataset and the relevant papers for the architectures when using this work academically. This repository is an experimental comparison and does not claim ownership of the underlying dataset or original network designs.
