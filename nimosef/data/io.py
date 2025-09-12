import os
import json
import numpy as np
import nibabel as nib
import pandas as pd

# --------------------
# Subject manifests
# --------------------
def save_subject_manifest(manifest_path, subject_id, parquet_paths, nifti_paths, meta, npy_paths=None):
    """Save a JSON manifest for one subject."""
    manifest = {
        "subject_id": subject_id,
        "parquet": parquet_paths,
        "nifti": nifti_paths,
        "meta": meta,
    }
    if npy_paths is not None:
        manifest["npy"] = npy_paths
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)


def load_subject_manifest(manifest_path):
    """Load JSON manifest for one subject."""
    with open(manifest_path, "r") as f:
        return json.load(f)


# --------------------
# Dataset manifest
# --------------------
def save_dataset_manifest(manifest_path, split_dict, bbox):
    """Save dataset-level manifest with splits and global bbox."""
    manifest = {**split_dict, "bbox": bbox}
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)


def load_dataset_manifest(manifest_path):
    """Load dataset-level manifest."""
    with open(manifest_path, "r") as f:
        return json.load(f)


# --------------------
# File loaders
# --------------------
def save_parquet_data( save_path, subj_id, intensity, segmentation, coords_scaled, coords, indices, center, affine, local_axis, ): 
    """ Save subject data into parquet + JSON transform file. 
    Args: save_path (str): directory where to save files. 
    subj_id (str): subject ID (e.g., "sub-0001"). 
    intensity (pd.DataFrame): intensity values [N_vox x T]. 
    segmentation (pd.DataFrame): segmentation labels [N_vox x T]. 
    coords_scaled (pd.DataFrame): scaled RWC coords [N_vox x 3]. 
    coords (pd.DataFrame): raw RWC coords [N_vox x 3]. 
    indices (pd.DataFrame): voxel indices [N_vox x 3]. 
    center (np.ndarray): 3D center in RWC. 
    affine (np.ndarray): affine matrix (4x4). 
    local_axis (np.ndarray): 
    local axis transform (4x4). """ 
    os.makedirs(save_path, exist_ok=True) 
    # save voxelwise data 
    intensity.to_parquet(os.path.join(save_path, f"{subj_id}_intensity.parquet")) 
    segmentation.to_parquet(os.path.join(save_path, f"{subj_id}_segmentation.parquet")) 
    coords_scaled.to_parquet(os.path.join(save_path, f"{subj_id}_coords_scaled_local.parquet")) 
    coords.to_parquet(os.path.join(save_path, f"{subj_id}_coords_local.parquet")) 
    indices.to_parquet(os.path.join(save_path, f"{subj_id}_indices.parquet")) 
    
    # save metadata (JSON) 
    transform_dict = { "center": center.tolist(), "affine": affine.tolist(), "local_axis": local_axis.tolist(), } 
    with open(os.path.join(save_path, f"{subj_id}_transform.json"), "w") as f: 
        json.dump(transform_dict, f, indent=4)


def load_parquet_from_manifest(manifest):
    """Load all parquet files listed in subject manifest."""
    pq = manifest["parquet"]
    df_int = pd.read_parquet(pq["intensity"])
    df_seg = pd.read_parquet(pq["segmentation"])
    df_coords_scaled = pd.read_parquet(pq["coords_scaled"])
    df_indices = pd.read_parquet(pq["indices"])
    with open(pq["transform"], "r") as f:
        transform = json.load(f)
    return df_int, df_seg, df_coords_scaled, df_indices, transform


def load_npy_from_manifest(manifest):
    """Load all parquet files listed in subject manifest."""
    npy = manifest["npy"]
    df_int = np.load(npy["intensity"],  mmap_mode="r")
    df_seg = np.load(npy["segmentation"], mmap_mode="r")
    df_coords_scaled = np.load(npy["coords_scaled"], mmap_mode="r")
    df_indices = np.load(npy["indices"], mmap_mode="r")
    
    return df_int, df_seg, df_coords_scaled, df_indices


def load_nifti(image_path):
    """Load a NIfTI image as numpy array."""
    nii = nib.load(image_path)
    return nii.get_fdata(), nii.affine


def load_segmentation(seg_path):
    """Load a segmentation NIfTI as int numpy array."""
    nii = nib.load(seg_path)
    return nii.get_fdata().astype(np.int64), nii.affine


def save_subject_arrays_npz(out_dir, subj_id, df_int, df_seg, df_coords_scaled, df_indices):
    """
    Save subject arrays into a compressed .npz file for fast runtime loading.
    """
    os.makedirs(out_dir, exist_ok=True)    
    npz_path = os.path.join(out_dir, f"{subj_id}.npz")
    
    np.savez_compressed(
        npz_path,
        intensity=df_int.values.astype(np.float32),
        segmentation=df_seg.values.astype(np.int16),
        coords_scaled=df_coords_scaled.values.astype(np.float32),
        indices=df_indices.values.astype(np.int32),
    )

    return npz_path


def save_subject_arrays_npy(out_dir, subj_id, df_int, df_seg, df_coords_scaled, df_indices):
    """
    Save subject arrays into a compressed .npz file for fast runtime loading.
    """
    os.makedirs(out_dir, exist_ok=True)            
    np.save(os.path.join(out_dir, f"{subj_id}_intensity.npy"), df_int.values.astype(np.float32))
    np.save(os.path.join(out_dir, f"{subj_id}_segmentation.npy"), df_seg.values.astype(np.int16))
    np.save(os.path.join(out_dir, f"{subj_id}_coords_scaled.npy"), df_coords_scaled.values.astype(np.float32))
    np.save(os.path.join(out_dir, f"{subj_id}_indices.npy"), df_indices.values.astype(np.int32))

    np_paths = {'intensity': os.path.join(out_dir, f"{subj_id}_intensity.npy"),
                'segmentation': os.path.join(out_dir, f"{subj_id}_segmentation.npy"),
                'coords_scaled': os.path.join(out_dir, f"{subj_id}_coords_scaled.npy"),
                'indices': os.path.join(out_dir, f"{subj_id}_indices.npy"),
                }
    return np_paths