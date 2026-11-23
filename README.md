# MangroveHeight-SSL

A sample-balanced self-supervised framework for mangrove canopy height estimation.

## Overview

Sparse spaceborne LiDAR coverage (the study estimates GEDI samples about 1.8% of mangrove area) and an uneven distribution of height observations make tall mangroves difficult to map. The framework uses two height-mapping stages:

- **Stage 1 – Self-supervised baseline:** A frozen DINOv2 encoder extracts features from unlabeled Sentinel-2 imagery. These are fused with geographic embeddings and GLO-30 elevation and decoded by a Dense Prediction Transformer (DPT). The model is supervised by GEDI observations with a composite loss (SigLoss + MSE + cross-entropy) and produces the Global Baseline Canopy Height (GBCH).
- **Stage 2 – Sample-balanced refinement:** Tiles with GBCH or GEDI heights ≥ 20 m are identified as high-canopy regions and refined with dynamic random undersampling. The refined predictions are then mosaicked with the baseline to produce the final 10 m Global Mangrove Canopy Height (GMCH) map.

The framework was validated against GEDI, airborne LiDAR (ALS), and field measurements.

## Environment

See [ENVIRONMENT.md](ENVIRONMENT.md) for dependencies and environment setup.

Run all commands from the repository root.

## Data preparation

Organize the training and validation data as follows:

```text
data/
  train/
    s2/tile001.tif
    dem/tile001.tif
    label/tile001.tif
  val/
    s2/tile101.tif
    dem/tile101.tif
    label/tile101.tif
```

Corresponding files must share the same filenames, spatial grid and coordinate reference system.

- **Sentinel-2:** Ten spectral bands in a consistent order.
- **DEM:** Single-band raster normalized to [0, 1].
- **Reference heights:** Single-band raster in metres, with zero representing missing observations by default.
- **Tile size:** 512 × 512 pixels.

NoData, masked and nonfinite values are excluded. Geographic coordinates are normalized to [0, 1].

## Training

### 1. Self-supervised pretraining

```bash
python train/train.py --dataset data/unlabeled --output-dir runs/ssl --arch large
```

The pretrained encoder uses DINO, iBOT and KoLeo losses.

### 2. Standard canopy height model

```bash
python train/train_dpt.py --train-root data/train --val-root data/val --teacher-checkpoint runs/ssl/teacher_checkpoint.pth --output-dir runs/standard
```

The standard model is trained using all valid reference height observations without sample balancing.

### 3. Sample-balanced canopy height model

```bash
python train/train_dpt_balanced.py --train-root data/balanced_train --val-root data/balanced_val --init-checkpoint runs/standard/standard_TIMESTAMP/best.pth --output-dir runs/balanced
```

During each epoch, the larger height class (<20 m or ≥20 m) is randomly undersampled within each training tile to match the smaller class. Tiles containing only one class are excluded from balanced training.

Both canopy models use the DualDPT architecture and a combination of SigLoss, MSE and height-histogram cross-entropy losses.

The checkpoint with the lowest validation RMSE is saved as `best.pth`.

## Prediction

Run canopy height prediction using a trained checkpoint:

```bash
python train/predict.py --checkpoint runs/balanced/balanced_TIMESTAMP/best.pth --s2 data/predict/s2 --dem data/predict/dem --output predictions/balanced
```

Predictions are saved as GeoTIFF files in metres.

## Pretrained weights

The self-supervised DINOv2 ViT-L/16 encoder weights for Stage 1 are publicly available on Zenodo:

- **DOI:** https://doi.org/10.5281/zenodo.24269172
- **Checkpoint:** `dinov2_vitl16_sentinel2_mangrove_ssl_v1.pth`
- **Architecture:** DINOv2 ViT-L/16

## License

DINOv2-derived components retain their upstream license notices. See [LICENSE](LICENSE).
