
---

### `docs/data_pipeline.md` (full explanation)
```markdown
# NIMOSEF Data Pipeline

This document describes how to preprocess data, generate splits, and load datasets.

---

## 1. Preprocessing (NIfTI → Parquet)

```bash
python scripts/preprocess_data.py \
    --root /data/ukbb \
    --use-roi \
    --patients 100
```

Each subject gets

```bash
sub-XXXX/implicit/
   sub-XXXX_intensity.parquet
   sub-XXXX_segmentation.parquet
   sub-XXXX_coords_scaled_local.parquet
   sub-XXXX_coords_local.parquet
   sub-XXXX_indices.parquet
   sub-XXXX_transform.json
```

Also saves a default train_val_test_split.json.

## 2. Generate Train/Val/Test Splits

```bash
python scripts/generate_splits.py \
    --root /data/ukbb \
    --split_file /data/ukbb/train_val_test_split.json
```

Options:

--bbox x y z : manually specify bounding box

--no-bbox : skip bounding box computation

--seed N : reproducible splits

Example output (train_val_test_split.json):
```bash
{
  "train": ["sub-0001", "sub-0002", ...],
  "val": ["sub-0100", "sub-0101", ...],
  "test": ["sub-0200", "sub-0201", ...],
  "bbox": [128.0, 128.0, 128.0]
}
```

## 3. Loading Dataset in PyTorch

```bash
from nimosef.data.dataset import NiftiDataset, nifti_collate_fn
from torch.utils.data import DataLoader

root = "/data/ukbb"
split_file = f"{root}/train_val_test_split.json"

train_set = NiftiDataset(root, split_file, mode="train")
val_set   = NiftiDataset(root, split_file, mode="val")

train_loader = DataLoader(
    train_set,
    batch_size=2,
    shuffle=True,
    collate_fn=nifti_collate_fn
)

for batch in train_loader:
    rwc, t0_data, tgt_data, t, transform_info, sample_id = batch
    # training loop...

```

