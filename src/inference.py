import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch

from src.dataset import decode
from src.model import PlateRecognitionModel
from src.paths import model_asset


def _assets_present() -> bool:
    """Return True if required OCR assets exist locally."""
    return (
        Path(model_asset("char_to_idx.json")).exists()
        and Path(model_asset("plate_model_v1.pth")).exists()
    )


@lru_cache(maxsize=1)
def load_crnn_model() -> Tuple[PlateRecognitionModel, dict, dict]:
    """
    Loads CRNN OCR model + vocab mappings.
    Lazy + cached so imports don't fail in CI (weights are not committed).
    """
    if not _assets_present():
        raise FileNotFoundError(
            "OCR model files not found. Expected:\n"
            f"- {model_asset('plate_model_v1.pth')}\n"
            f"- {model_asset('char_to_idx.json')}\n"
            "Download/place them locally as described in README."
        )

    with open(model_asset("char_to_idx.json"), "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)

    idx_to_char = {int(v): k for k, v in char_to_idx.items()}

    model = PlateRecognitionModel(vocab_size=len(char_to_idx))
    state = torch.load(model_asset("plate_model_v1.pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, idx_to_char, char_to_idx


def predict_plate(tensor_image) -> str:
    """
    tensor_image: torch.Tensor shaped like [1, 1, 32, W] after transforms.
    Returns decoded string.
    """
    model, idx_to_char, _ = load_crnn_model()

    with torch.no_grad():
        logits = model(tensor_image)          # expected: [T, B, C]
        pred = logits.argmax(2).squeeze(1)    # [T]
    return decode(pred, idx_to_char)


if __name__ == "__main__":
    print("inference.py loaded OK")
