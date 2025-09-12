import torch
import numpy as np
import torch.nn.functional as F
import scipy.ndimage
from pytorch3d.ops import knn_points


def compute_pairwise_knn(points, k=10):
    points = points.unsqueeze(0) # (N, D)
    knn = knn_points(points, points, K=k, return_nn=False)  # (B, N, D)
    return knn.dists.squeeze(0), knn.idx.squeeze(0) # (N, K)


def gaussian_weights(points, sigma):
    distances, n_idx = compute_pairwise_knn(points, k=10)
    weights = torch.exp(-distances / (2 * sigma**2))
    return weights, n_idx, distances


def compute_signed_distance_with_boundary(mask):
    mask_np = mask.cpu().numpy()

    structure = np.ones([3]*mask.ndim, dtype=np.uint8)
    dilated_mask = scipy.ndimage.binary_dilation(mask_np, structure=structure).astype(mask_np.dtype)

    boundary = dilated_mask - mask_np
    inverted_mask = 1 - boundary
    distance_transform = scipy.ndimage.distance_transform_edt(inverted_mask)

    sdf = torch.from_numpy(distance_transform).to(mask.device).float()
    sdf[mask_np == 1] *= -1
    sdf[boundary == 1] = 0

    return sdf


def compute_sdf(segmentation_logits, labels):
    seg_labels = segmentation_logits.argmax(dim=1)  # (B, X, Y, Z)
    seg_pred = F.one_hot(seg_labels, num_classes=len(labels))
    sdf = torch.zeros_like(seg_pred, dtype=torch.float32)
    for i, label in enumerate(labels):
        if label == 0:
            continue
        mask = seg_pred[:, :, :, :, i].float()
        sdf_label = compute_signed_distance_with_boundary(mask)
        sdf[..., i] = sdf_label
    return sdf


def extract_boundary_points(segmentation, coords, sample_id, k=5):
    coords_extended = torch.cat((coords, sample_id.unsqueeze(-1)), dim=1)
    coords_batch = coords_extended.unsqueeze(0).float()
    knn = knn_points(coords_batch, coords_batch, K=k + 1)
    knn_indices = knn.idx[0, :, 1:]

    seg_expanded = segmentation.unsqueeze(1).expand(-1, k)
    neighbor_seg = seg_expanded.gather(0, knn_indices)
    is_boundary = (seg_expanded != neighbor_seg).any(dim=1)

    boundary_indices = is_boundary.nonzero(as_tuple=True)[0]
    boundary_coords = coords[boundary_indices]
    mean_distance = (coords[knn_indices] - coords.unsqueeze(1)).norm(dim=2).mean()

    return boundary_indices.long(), boundary_coords, mean_distance
