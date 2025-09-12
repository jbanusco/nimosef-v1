import torch
import pytest
from nimosef.models.layers import WIRE, INR
from nimosef.models.nimosef import MultiHeadNetwork


def test_wire_forward_and_grad():
    torch.manual_seed(0)
    N, in_size, out_size = 5, 3, 4
    x = torch.rand(N, in_size, requires_grad=True)

    layer = WIRE(in_size, out_size, is_first=True, trainable=True)
    y = layer(x)

    assert y.shape == (N, out_size)
    y.mean().backward()
    assert x.grad is not None, "WIRE should be differentiable"


def test_inr_forward_shapes_residual_and_nonresidual():
    torch.manual_seed(0)
    coord_size, embed_size, hidden_size, out_size = 3, 4, 8, 6
    N = 10
    coords = torch.rand(N, coord_size)
    embed = torch.rand(N, embed_size)

    # With residual
    inr = INR(coord_size, embed_size, embed_size, embed_size, num_hidden_layers=2, use_residual=True)
    out_res = inr((coords, embed))
    assert out_res.shape == (N, embed_size)

    # Without residual
    inr = INR(coord_size, embed_size, hidden_size, out_size, num_hidden_layers=2, use_residual=False)
    out_nores = inr((coords, embed))
    assert out_nores.shape == (N, out_size)


def test_inr_backward():
    torch.manual_seed(0)
    coord_size, embed_size, out_size = 3, 4, 6
    N = 5
    coords = torch.rand(N, coord_size, requires_grad=True)
    embed = torch.rand(N, embed_size, requires_grad=True)

    input_size = coord_size + 1
    hidden_size = embed_size
    inr = INR(coord_size, embed_size, embed_size, embed_size, num_hidden_layers=2, use_residual=True)
    #NOTE: if use residual the inner size has to be the same as embed_size
    out = inr((coords, embed))

    loss = out.mean()
    loss.backward()

    assert coords.grad is not None
    assert embed.grad is not None


def test_multiheadnetwork_forward_shapes():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 2, 4, 8, 6
    # hidden_size = 16
    hidden_size = latent_size  # For the residual
    model = MultiHeadNetwork(
        num_subjects=num_subjects,
        num_labels=num_labels,
        latent_size=latent_size,
        motion_size=motion_size,
        hidden_size=hidden_size,
        num_res_layers=2,
        linear_head=True,
    )

    N = 20
    coords = torch.rand(N, 3)
    time = torch.rand(N, 1)
    sample_idx = torch.randint(0, num_subjects, (N,))

    seg_pred, intensity_pred, displacement, h = model(coords, time, sample_idx)

    assert seg_pred.shape == (N, num_labels)
    assert torch.allclose(seg_pred.sum(dim=1), torch.ones(N), atol=1e-5), "Segmentation must be softmax normalized"
    assert intensity_pred.shape == (N, 1)
    assert (0 <= intensity_pred).all() and (intensity_pred <= 1).all(), "Intensity must be in [0,1]"
    assert displacement.shape == (N, 3)
    assert h.shape[1] == latent_size


def test_multiheadnetwork_decode_latent():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 3, 4, 8, 6
    # hidden_size = 16
    hidden_size = latent_size  # For the residual
    model = MultiHeadNetwork(num_subjects, num_labels, latent_size, motion_size, hidden_size=hidden_size)

    N = 10
    coords = torch.rand(N, 3)
    time = torch.rand(N, 1)
    h = torch.rand(N, latent_size)
    corr_code = torch.rand(N, model.correction_size)

    latent_t = model.decode_latent(coords, time, h, corr_code)
    assert latent_t.shape == (N, latent_size)


def test_multiheadnetwork_backward_pass():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 2, 3, 8, 6
    # hidden_size = 16
    hidden_size = latent_size  # For the residual
    model = MultiHeadNetwork(num_subjects, num_labels, latent_size, motion_size, hidden_size=hidden_size)

    N = 15
    coords = torch.rand(N, 3, requires_grad=True)
    time = torch.rand(N, 1, requires_grad=True)
    sample_idx = torch.randint(0, num_subjects, (N,))

    seg_pred, intensity_pred, displacement, h = model(coords, time, sample_idx)
    loss = seg_pred.mean() + intensity_pred.mean() + displacement.mean() + h.mean()
    loss.backward()

    # Check gradients flow
    assert coords.grad is not None
    assert time.grad is not None
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def numerical_grad_scalar(f, x, eps=1e-4):
    """Finite-difference gradient for scalar f(x) wrt tensor x."""
    grad = torch.zeros_like(x)
    flat_x = x.view(-1)
    flat_grad = grad.view(-1)

    for i in range(flat_x.numel()):
        orig = flat_x[i].item()
        flat_x[i] = orig + eps
        f_pos = f(x).item()
        flat_x[i] = orig - eps
        f_neg = f(x).item()
        flat_x[i] = orig
        flat_grad[i] = (f_pos - f_neg) / (2 * eps)
    return grad


# ---------------------------
# WIRE
# ---------------------------
def test_wire_numerical_grad():
    torch.manual_seed(0)
    N, in_size, out_size = 4, 3, 5
    x = torch.rand(N, in_size, requires_grad=True)
    layer = WIRE(in_size, out_size, is_first=True, trainable=True)

    def loss_fn(inp):
        return layer(inp).sum()

    # autograd
    y = loss_fn(x)
    y.backward()
    autograd_grad = x.grad.clone()

    # finite difference
    fd_grad = numerical_grad_scalar(loss_fn, x.detach().clone())

    diff = (autograd_grad - fd_grad).abs().mean().item()
    assert diff < 1e-2, f"WIRE gradient mismatch too large: {diff}"


# ---------------------------
# INR
# ---------------------------
def test_inr_numerical_grad():
    torch.manual_seed(0)
    coord_size, embed_size = 3, 4
    N = 5
    coords = torch.rand(N, coord_size, requires_grad=True)
    embed = torch.rand(N, embed_size, requires_grad=True)

    inr = INR(coord_size, embed_size, embed_size, embed_size,
              num_hidden_layers=2, use_residual=True)

    def loss_fn(c, e):
        return inr((c, e)).sum()

    # autograd
    out = inr((coords, embed))
    out.sum().backward()
    autograd_grad_coords = coords.grad.clone()
    autograd_grad_embed = embed.grad.clone()

    # finite difference (coords)
    fd_grad_coords = numerical_grad_scalar(lambda c: loss_fn(c, embed.detach()), coords.detach().clone())
    # finite difference (embed)
    fd_grad_embed = numerical_grad_scalar(lambda e: loss_fn(coords.detach(), e), embed.detach().clone())

    diff_coords = (autograd_grad_coords - fd_grad_coords).abs().mean().item()
    diff_embed = (autograd_grad_embed - fd_grad_embed).abs().mean().item()

    assert diff_coords < 1e-2, f"INR coords grad mismatch: {diff_coords}"
    assert diff_embed < 1e-2, f"INR embed grad mismatch: {diff_embed}"


# ---------------------------
# MultiHeadNetwork
# ---------------------------
def test_multiheadnetwork_numerical_grad():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 2, 3, 4, 2
    hidden_size = latent_size
    model = MultiHeadNetwork(num_subjects, num_labels, latent_size, motion_size, hidden_size=hidden_size)

    N = 6
    coords = torch.rand(N, 3, requires_grad=True)
    time = torch.rand(N, 1, requires_grad=True)
    sample_idx = torch.randint(0, num_subjects, (N,))

    def loss_fn(c, t):
        seg_pred, intensity_pred, displacement, h = model(c, t, sample_idx)
        return seg_pred.sum() + intensity_pred.sum() + displacement.sum() + h.sum()

    # autograd
    out = loss_fn(coords, time)
    out.backward()
    autograd_grad_coords = coords.grad.clone()
    autograd_grad_time = time.grad.clone()

    # finite difference (coords)
    fd_grad_coords = numerical_grad_scalar(lambda c: loss_fn(c, time.detach()), coords.detach().clone())
    # finite difference (time)
    fd_grad_time = numerical_grad_scalar(lambda t: loss_fn(coords.detach(), t), time.detach().clone())

    diff_coords = (autograd_grad_coords - fd_grad_coords).abs().mean().item()
    diff_time = (autograd_grad_time - fd_grad_time).abs().mean().item()

    assert diff_coords < 1e-2, f"MultiHead coords grad mismatch: {diff_coords}"
    assert diff_time < 1e-2, f"MultiHead time grad mismatch: {diff_time}"


def test_multiheadnetwork_numerical_grad_embeddings():
    torch.manual_seed(0)
    num_subjects, num_labels, latent_size, motion_size = 3, 3, 4, 2
    hidden_size = latent_size
    model = MultiHeadNetwork(num_subjects, num_labels, latent_size, motion_size, hidden_size=hidden_size)

    N = 4
    coords = torch.rand(N, 3)
    time = torch.rand(N, 1)
    sample_idx = torch.randint(0, num_subjects, (N,))

    def loss_fn():
        seg_pred, intensity_pred, displacement, h = model(coords, time, sample_idx)
        return seg_pred.sum() + intensity_pred.sum() + displacement.sum() + h.sum()

    # autograd
    loss = loss_fn()
    loss.backward()

    autograd_grad_shape = model.shape_code.weight.grad.clone()
    autograd_grad_corr = model.correction_code.weight.grad.clone()

    # finite difference (shape_code weights)
    shape_weight = model.shape_code.weight.data.clone()
    fd_grad_shape = torch.zeros_like(shape_weight)

    eps = 1e-4
    for i in range(shape_weight.numel()):
        orig = shape_weight.view(-1)[i].item()
        model.shape_code.weight.data.view(-1)[i] = orig + eps
        f_pos = loss_fn().item()
        model.shape_code.weight.data.view(-1)[i] = orig - eps
        f_neg = loss_fn().item()
        model.shape_code.weight.data.view(-1)[i] = orig
        fd_grad_shape.view(-1)[i] = (f_pos - f_neg) / (2 * eps)

    # finite difference (correction_code weights)
    corr_weight = model.correction_code.weight.data.clone()
    fd_grad_corr = torch.zeros_like(corr_weight)

    for i in range(corr_weight.numel()):
        orig = corr_weight.view(-1)[i].item()
        model.correction_code.weight.data.view(-1)[i] = orig + eps
        f_pos = loss_fn().item()
        model.correction_code.weight.data.view(-1)[i] = orig - eps
        f_neg = loss_fn().item()
        model.correction_code.weight.data.view(-1)[i] = orig
        fd_grad_corr.view(-1)[i] = (f_pos - f_neg) / (2 * eps)

    # Compare
    diff_shape = (autograd_grad_shape - fd_grad_shape).abs().mean().item()
    diff_corr = (autograd_grad_corr - fd_grad_corr).abs().mean().item()

    assert diff_shape < 1e-2, f"Shape embedding gradient mismatch too large: {diff_shape}"
    assert diff_corr < 1e-2, f"Correction embedding gradient mismatch too large: {diff_corr}"
