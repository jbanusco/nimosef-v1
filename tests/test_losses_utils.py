import torch
import pytest
from pytorch3d.loss import chamfer_distance

from nimosef.losses.utils import (
    compute_pairwise_knn,
    gaussian_weights,
    compute_signed_distance_with_boundary,
    compute_sdf,
    extract_boundary_points,
)


def test_chamfer_distance_sanity():
    # Two identical point clouds
    pts1 = torch.tensor([[[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]]])
    pts2 = pts1.clone()
    loss, _ = chamfer_distance(pts1, pts2)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    # Shifted version of pts2
    shift = torch.tensor([[[1.0, 0.0, 0.0]]])
    pts2_shifted = pts2 + shift
    loss_shifted, _ = chamfer_distance(pts1, pts2_shifted)
    assert loss_shifted > loss

    # Larger shift → larger Chamfer
    pts2_big_shift = pts2 + torch.tensor([[[5.0, 0.0, 0.0]]])
    loss_big, _ = chamfer_distance(pts1, pts2_big_shift)
    assert loss_big > loss_shifted


def test_compute_pairwise_knn_shapes():
    points = torch.rand(8, 3)
    dists, idx = compute_pairwise_knn(points, k=4)
    assert dists.shape == idx.shape
    assert dists.shape[1] == 4


def test_gaussian_weights_nonnegative():
    points = torch.rand(6, 3)
    weights, n_idx, dists = gaussian_weights(points, sigma=1.0)
    assert torch.all(weights >= 0)
    assert weights.shape == dists.shape


def test_compute_signed_distance_with_boundary_inside_outside():
    mask = torch.zeros((6, 6, 1, 1), dtype=torch.int)
    mask[2:4, 2:4] = 1
    sdf = compute_signed_distance_with_boundary(mask)
    assert sdf.shape == mask.shape
    # Inside should be ≤ 0, outside ≥ 0
    assert (sdf[mask == 1] <= 0).all()
    assert (sdf[mask == 0] >= 0).all()


def test_compute_sdf_multiple_labels():
    logits = torch.randint(0, 2, (1, 1, 6, 6, 6))  # fake logits [B, C, X, Y, Z]
    labels = [0, 1]
    sdf = compute_sdf(logits, labels)
    assert isinstance(sdf, torch.Tensor)
    assert sdf.shape[-1] == len(labels)


def test_extract_boundary_points_detects_boundary():
    N = 12
    coords = torch.rand(N, 3)
    seg = torch.cat([torch.zeros(N//2), torch.ones(N//2)]).long()
    sample_id = torch.zeros(N, dtype=torch.long)
    idx, bcoords, mdist = extract_boundary_points(seg, coords, sample_id, k=3)
    assert idx.ndim == 1
    assert bcoords.shape[1] == 3
    assert mdist > 0


def test_compute_pairwise_knn():
    points = torch.tensor([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    dists, idx = compute_pairwise_knn(points, k=2)
    # print(dists, idx)
    assert dists.shape == (3, 2)
    assert idx.shape == (3, 2)
    # nearest neighbor of (0,0,0) should be (1,0,0) or (0,1,0)
    assert idx[0,1].item() in [1,2]


def test_gaussian_weights():
    points = torch.rand(5, 3)
    weights, n_idx, dists = gaussian_weights(points, sigma=0.5)
    assert weights.shape == dists.shape == n_idx.shape
    assert torch.all(weights >= 0) and torch.all(weights <= 1)


def test_compute_signed_distance_with_boundary_simple():
    mask = torch.zeros((10, 10), dtype=torch.int)
    mask[3:7, 3:7] = 1
    sdf = compute_signed_distance_with_boundary(mask)
    assert sdf.shape == mask.shape
    assert (sdf[mask == 1] <= 0).all()
    assert (sdf[mask == 0] >= 0).all()
    assert torch.any(sdf == 0)  # boundary should exist


def test_compute_sdf():
    # 1 batch, 2 classes (background + fg)
    logits = torch.randn(1, 2, 8, 8, 8)
    sdf = compute_sdf(logits, labels=[0,1])
    assert sdf.shape == (1, 8, 8, 8, 2)
    # class 0 should be all zeros
    assert torch.allclose(sdf[...,0], torch.zeros_like(sdf[...,0])) or torch.all(sdf[...,0] == 0)


def test_extract_boundary_points_square():
    # simple square segmentation in 2D
    coords = torch.tensor([[x,y,0.0] for x in range(5) for y in range(5)], dtype=torch.float32)
    seg = torch.zeros(25, dtype=torch.long)
    seg[6:9] = 1  # make a small block
    sample_id = torch.zeros(25, dtype=torch.long)

    boundary_idx, boundary_coords, mean_dist = extract_boundary_points(seg, coords, sample_id, k=3)
    assert isinstance(boundary_idx, torch.Tensor)
    assert isinstance(boundary_coords, torch.Tensor)
    assert mean_dist > 0
    # boundary points must be fewer than all points
    assert len(boundary_idx) < len(seg)


@pytest.mark.parametrize("dim", [2,3])
def test_sdf_dimensionality(dim):
    # Check 2D vs 3D mask handling
    shape = (10,10) if dim == 2 else (10,10,10)
    mask = torch.zeros(shape, dtype=torch.int)
    slices = tuple(slice(3,7) for _ in range(dim))
    mask[slices] = 1
    sdf = compute_signed_distance_with_boundary(mask)
    assert sdf.shape == shape
