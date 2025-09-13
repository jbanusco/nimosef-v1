# NIMOSEF - v1

Official repository for the MICCAI 2025 paper:

**NIMOSEF: Neural Implicit Motion and Segmentation Functions**

Available here: https://papers.miccai.org/miccai-2025/0638-Paper4418.html

---

## About

NIMOSEF is a unified framework that leverages implicit neural representations for joint segmentation, intensity reconstruction, and displacement field estimation in cardiac MRI.

This repository will contain the exact version of the code used in the MICCAI 2025 submission.

---

## Status

🕐 The code is currently being prepared and will be uploaded soon.  
✅ Repository name and structure are stable. Please check back soon.

---

## Quick Start

### 1. Preprocess raw NIfTI into parquet
```bash
python scripts/preprocess_data.py --root /data/ukbb --use-roi
```

### 2. Generate trian/val/test splits
```bash
python scripts/generate_splits.py --root /data/ukbb
```

### 3. Load dataset in PyTorch
```bash
from nimosef.data.dataset import NiftiDataset, nifti_collate_fn
from torch.utils.data import DataLoader

root = "/data/ukbb"
split_file = f"{root}/train_val_test_split.json"

train_set = NiftiDataset(root, split_file, mode="train")
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=nifti_collate_fn)
```

See docs/data_pipeline.md for full details on preprocessing, splitting, and dataset loading.

---

## 1. Preprocess & Create Train/Val/Test Split

Convert raw UK Biobank (or similar BIDS-structured) data into .parquet coordinate tables and generate splits:

Example with an existing databaset

python scripts/generate_splits.py \
  --load_dir /path/to/UKB_Cardiac_BIDS \
  --number_patients 1000 \
  --split_ratios 0.7 0.15 0.15 \
  --split_file derivatives/nimosef_flip_logs/train_val_test_split.json

## 2. rain the Model

Train the MultiHeadNetwork on train/val sets:

python scripts/train.py \
  --data_folder /path/to/UKB_Cardiac_BIDS \
  --save_folder derivatives/nimosef_flip_logs/baseline \
  --split_file derivatives/nimosef_flip_logs/train_val_test_split.json \
  --num_epochs 1000 \
  --latent_size 128 --motion_size 64 --hidden_size 128 --num_res_layers 8 \
  --lambda_rec 2.0 --lambda_seg 1.0 --lambda_dsp 0.5 --lambda_jacobian 1.0

Check logs during training with:

tensorboard --logdir derivatives/nimosef_flip_logs/baseline

## 3. Run Inference (Per-Subject Fine-Tuning)

Freeze the trained decoder and fine-tune only the shape codes for unseen test patients:

python scripts/inference.py \
  --data_folder /path/to/UKB_Cardiac_BIDS \
  --save_folder derivatives/nimosef_flip_logs/baseline \
  --split_file derivatives/nimosef_flip_logs/train_val_test_split.json \
  --mode test \
  --initial_model_path derivatives/nimosef_flip_logs/baseline/experiment_XXX/model.pth \
  --validation_epochs 200 \
  --save_rec_folders derivatives/nimosef_subjects \
  --load_validation_and_rec True

This produces for each subject:
*_rec.nii.gz → reconstructed intensity
*_seg.nii.gz → predicted segmentation
*_seg_gt.nii.gz → ground-truth segmentation

plus displacement and boundary .parquet files.

⚙️ By default, results are saved in:

derivatives/nimosef_flip_logs/baseline/experiment_YYYYMMDD_HHMMSS/
derivatives/nimosef_subjects/

## 4. Results

After inference, results for each subject are saved under:

derivatives/nimosef_subjects/{subject_id}/

# Generated Files
File	Description
{id}_rec.nii.gz	Reconstructed 4D MR intensity volume (model’s prediction).
{id}_seg.nii.gz	Predicted 4D segmentation volume (per time frame).
{id}_seg_gt.nii.gz	Ground-truth segmentation (if available).
{id}_im_gt.nii.gz	Ground-truth intensity image (if available).
{id}_pred_boundaries.parquet	Boundary point coordinates (x, y, z, t) predicted by the model. Useful for shape tracking.
{id}_true_boundaries.parquet	Ground-truth boundary points (if available).
{id}_displacement.parquet	Predicted displacement vectors at boundary points. Encodes motion across time.

# How to Use the Outputs

Visualize reconstructed MR
Load *_rec.nii.gz in FSLeyes or ITK-SNAP to compare against ground truth (*_im_gt.nii.gz).

Evaluate segmentation
Compare *_seg.nii.gz vs *_seg_gt.nii.gz using Dice coefficient, Hausdorff distance, or volume metrics.
Example: see losses/get_average_metrics.py.

Analyze motion fields
Use {id}_displacement.parquet to reconstruct displacement trajectories.

Columns: x, y, z, time = location

Values = 3D displacement vector

Shape tracking
Boundary .parquet files let you track endocardium/epicardium surfaces across time without voxelization.

This means:

Voxel-wise predictions = in NIfTI files (.nii.gz).

Geometry-aware analysis = in .parquet boundary/displacement files.

## Citation

If you use or refer to NIMOSEF, please cite:

@InProceedings{BanJau_NIMOSEF_MICCAI2025,
        author = { Banus, Jaume and Delaloye, Antoine and M. Gordaliza, Pedro and Georgantas, Costa and B. van Heeswijk, Ruud and Richiardi, Jonas},
        title = { { NIMOSEF: Neural implicit motion and segmentation functions } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15970},
        month = {September},

}