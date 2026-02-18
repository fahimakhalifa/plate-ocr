import pytest


def test_predict_plate_contract_monkeypatch(monkeypatch):
    import src.inference as inf

    # If weights aren't present (CI), skip. Repo policy says weights aren't committed.
    try:
        inf.load_crnn_model()
    except FileNotFoundError:
        pytest.skip("OCR weights not available in CI; skipping inference test.")

    # If weights exist locally, run the contract test as usual:
    import torch
    x = torch.randn(1, 1, 32, 160)
    out = inf.predict_plate(x)
    assert isinstance(out, str)
