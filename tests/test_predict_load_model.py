"""Compatibility tests for training.predict.load_model."""

from pathlib import Path

import torch

from training.predict import load_model


class TinyModel(torch.nn.Module):
    """Minimal model for checkpoint loading tests."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def _write_checkpoint(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_load_model_plain_state_dict(tmp_path, monkeypatch):
    """load_model supports legacy checkpoints saved as plain state_dict."""

    source_model = TinyModel()
    state_dict = source_model.state_dict()
    checkpoint_path = tmp_path / "plain_model.pth"
    _write_checkpoint(checkpoint_path, state_dict)

    monkeypatch.setattr("training.predict.create_model", lambda **_: TinyModel())

    loaded_model = load_model(str(checkpoint_path), device="cpu")

    for name, param in loaded_model.state_dict().items():
        assert torch.equal(param, state_dict[name])
    assert not hasattr(loaded_model, "_temperature")


def test_load_model_calibrated_checkpoint_dict(tmp_path, monkeypatch):
    """load_model supports calibrated checkpoints with nested model_state_dict."""

    source_model = TinyModel()
    state_dict = source_model.state_dict()
    checkpoint_path = tmp_path / "calibrated_model.pth"
    _write_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": state_dict,
            "temperature": 2.5,
        },
    )

    monkeypatch.setattr("training.predict.create_model", lambda **_: TinyModel())

    loaded_model = load_model(str(checkpoint_path), device="cpu")

    for name, param in loaded_model.state_dict().items():
        assert torch.equal(param, state_dict[name])
    assert hasattr(loaded_model, "_temperature")
    assert loaded_model._temperature == 2.5  # noqa: SLF001
