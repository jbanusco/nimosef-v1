import os
import json
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from nimosef.data.io import load_subject_manifest, load_parquet_from_manifest, load_npy_from_manifest


def nifti_collate_fn(batch):
    rwc_list, t0_data_list, tgt_data_list, t_list, transform_info_list, sample_idx = zip(*batch)

    # Sample ID's
    sample_id = [torch.tile(torch.tensor(sample_idx[i]), (rwc_list[i].shape[0],)) for i in range(0, len(sample_idx))]
    sample_id = torch.cat(sample_id, dim=0)

    # Concatenate real-world coordinates (RWC)
    rwc = torch.cat(rwc_list, dim=0)

    # Concatenate t0 data (intensity and segmentation)
    t0_intensity = torch.cat([t0[0] for t0 in t0_data_list], dim=0)
    t0_segmentation = torch.cat([t0[1] for t0 in t0_data_list], dim=0)
    t0_mask = torch.cat([t0[2] for t0 in t0_data_list], dim=0)
    t0_data = (t0_intensity, t0_segmentation, t0_mask)

    # Concatenate target data (intensity and segmentation)    
    tgt_intensity = torch.cat([tgt[0] for tgt in tgt_data_list], dim=0)
    tgt_segmentation = torch.cat([tgt[1] for tgt in tgt_data_list], dim=0)
    tgt_mask = torch.cat([tgt[2] for tgt in tgt_data_list], dim=0)
    tgt_data = (tgt_intensity, tgt_segmentation, tgt_mask)

    # Concatenate normalized time (T)
    t = [torch.tile(torch.tensor(t_list[i]), (rwc_list[i].shape[0],)) for i in range(len(sample_idx))]
    t = torch.cat(t, dim=0)

    centers, affines, indices = zip(*transform_info_list)
    transform_info = {
        "centers": centers,
        "affines": affines,
        "indices": indices
    }

    assert np.isclose(rwc.shape[0], sample_id.shape[0])
    assert np.isclose(rwc.shape[0], t0_intensity.shape[0])
    assert np.isclose(rwc.shape[0], tgt_intensity.shape[0])
    assert np.isclose(rwc.shape[0], t.shape[0])

    return rwc, t0_data, tgt_data, t, transform_info, sample_id


class NiftiDataset(Dataset):
    def __init__(self, dataset_manifest, mode="train", transform=None, use_local_axis=True):
        """
        Args:
            dataset_manifest (str): Path to dataset-level manifest JSON.
            mode (str): One of ['train', 'val', 'test'].
            transform: Optional data augmentation.
        """
        assert mode in ["train", "val", "test"], "Mode must be train, val or test."
        self.mode = mode
        self.transform = transform
        self.use_local_axis = use_local_axis

        # Load dataset manifest
        with open(dataset_manifest, "r") as f:
            split_data = json.load(f)

        self.subject_manifests = split_data[mode]
        self.bbox = tuple(split_data["bbox"]) if split_data["bbox"] is not None else None

        # --- Add patients list (subject IDs) --- and also the metadata
        self.patients = []
        self.subjects = []

        for manifest_path in self.subject_manifests:
            subj_manifest = load_subject_manifest(manifest_path)
            self.patients.append(subj_manifest["subject_id"])

            # Metadata for fast loading (avoid .json load every time)
            # self.subjects.append({
            #     "subject_id": subj_manifest["subject_id"],
            #     "parquet": subj_manifest["parquet"],
            #     "npy": subj_manifest.get("npy", None),  # optional .npy paths
            #     "npz": subj_manifest["parquet"].get("npz", None),  # optional .npz path
            #     "transform": subj_manifest["parquet"]["transform"],  # json path
            # })
            
            # open arrays once, mmap-backed
            df_int, df_seg, df_coords_scaled, df_indices, transform = self._load_data(subj_manifest)
            arrays = {
                "intensity": df_int,
                "segmentation": df_seg,
                "coords_scaled":df_coords_scaled,
                "indices": df_indices,
            }
            
            self.subjects.append({
                "subject_id": subj_manifest["subject_id"],
                "arrays": arrays,
                "transform": transform
            })
            

    def __len__(self):
        return len(self.subject_manifests)

    def _load_data(self, manifest):
        npy_paths = manifest.get("npy", None)
        npz_path = manifest["parquet"]["npz"] if "npz" in manifest else None
        
        if npz_path and os.path.exists(npz_path):            
            # Load from .npz file
            data = np.load(npz_path, allow_pickle=False, mmap_mode="r")
            df_int = data["intensity"]
            df_seg = data["segmentation"]
            df_coords_scaled = data["coords_scaled"]
            df_indices = data["indices"]
            with open(manifest["parquet"]["transform"], "r") as f:
                transform = json.load(f)
        elif npy_paths is not None:
            # Load from .npy files
            df_int, df_seg, df_coords_scaled, df_indices = load_npy_from_manifest(manifest)
            with open(manifest["parquet"]["transform"], "r") as f:
                transform = json.load(f)
        else:
            # Load from .parquet files
            # fallback: parquet        
            df_int, df_seg, df_coords_scaled, df_indices, transform = load_parquet_from_manifest(manifest)
            df_int = df_int.values
            df_seg = df_seg.values
            df_coords_scaled = df_coords_scaled.values
            df_indices = df_indices.values
        
        return df_int, df_seg, df_coords_scaled, df_indices, transform

    def __getitem__(self, idx):        
        # Load data here
        # manifest = self.subjects[idx]
        # df_int, df_seg, df_coords_scaled, df_indices, transform = self._load_data(manifest)

        # Use pre-loaded data
        arrays = self.subjects[idx]["arrays"]
        transform = self.subjects[idx]["transform"]

        df_int = arrays["intensity"]
        df_seg = arrays["segmentation"]
        df_coords_scaled = arrays["coords_scaled"]
        df_indices = arrays["indices"]

        num_frames = df_int.shape[1]
        t0 = 0
        t_frame = np.random.choice(num_frames, 1)[0]
        t = t_frame / num_frames

        rwc = torch.tensor(df_coords_scaled).float()
        indices = torch.tensor(df_indices).float()
        center = torch.tensor(transform["center"]).float()
        affine = torch.tensor(transform["affine"]).float()
        local_axis = torch.tensor(transform["local_axis"]).float()
        
        t0_intensity = torch.tensor(df_int[:, t0]).float()
        t0_seg = torch.tensor(df_seg[:, t0]).long()
        t0_mask = (t0_seg > 0).long()

        tgt_intensity = torch.tensor(df_int[:, t_frame]).float()
        tgt_seg = torch.tensor(df_seg[:, t_frame]).long()
        tgt_mask = (tgt_seg > 0).long()

        t0_data = (t0_intensity, t0_seg, t0_mask)
        tgt_data = (tgt_intensity, tgt_seg, tgt_mask)

        if self.use_local_axis:
            # Combine the local axis with the affine transformation
            affine[:3, :3] = affine[:3, :3] @ local_axis[:3, :3].T
            affine[:3, 3] += local_axis[:3, 3]
        transform_info = (center, affine, indices)

        return rwc, t0_data, tgt_data, t, transform_info, idx