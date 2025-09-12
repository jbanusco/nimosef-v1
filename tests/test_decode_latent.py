import pandas as pd
import torch
import pytest
from nimosef.analysis import save_shape_code


class DummyDataset:
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return (None, None, None, None, None, None)


class DummyModel:
    def __init__(self, *a, **kw): pass
    def to(self, *a, **kw): return self
    def eval(self): return self
    def load_state_dict(self, *a, **kw): return self


def test_decode_latent_calls_generate(tmp_path, monkeypatch):
    # --- Create dummy latent CSV file ---
    latent_file = tmp_path / "latent.csv"
    df = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]], columns=["h1", "h2", "h3", "h4"])
    df.to_csv(latent_file, index=False)

    # --- Fake args ---
    args = type("Args", (), {
        "data_folder": str(tmp_path),
        "split_file": str(tmp_path / "split.json"),
        "experiment_name": "exp",
        "save_folder": str(tmp_path),
        "latent_filename": str(latent_file),  # <- CSV file
        "res_factor_z": 1,
        "reprocess": True,
    })()

    # --- Monkeypatch dependencies ---
    monkeypatch.setattr(save_shape_code, "NiftiDataset", lambda *a, **kw: DummyDataset())
    monkeypatch.setattr(save_shape_code, "MultiHeadNetwork", lambda *a, **kw: DummyModel())
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {})

    called = {}
    def fake_generate(*a, **kw):
        called["ok"] = True
    monkeypatch.setattr(save_shape_code, "generate_reconstructed_images", fake_generate)

    # --- Run ---
    save_shape_code.decode_latent(args)

    # --- Check that generate_reconstructed_images was called ---
    assert "ok" in called, "generate_reconstructed_images was not called"
