# FILE: plate-ocr/src/inference.py
import json

import torch

from src.dataset import decode
from src.model import PlateRecognitionModel
from src.paths import model_asset


def load_crnn_model():
    """
    Loads CRNN OCR model + vocab mappings.
    Kept as a stable API because process_video imports it.
    """
    with open(model_asset("char_to_idx.json"), "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)

    idx_to_char = {int(v): k for k, v in char_to_idx.items()}

    model = PlateRecognitionModel(vocab_size=len(char_to_idx))
    model.load_state_dict(torch.load(model_asset("plate_model_v1.pth"), map_location="cpu"))
    model.eval()
    return model, idx_to_char, char_to_idx


# Keep current module-level behavior: model is loaded on import
_model, _idx_to_char, _char_to_idx = load_crnn_model()


def predict_plate(tensor_image):
    """
    tensor_image: torch.Tensor shaped like [1, 1, 32, W] after transforms.
    Returns decoded string.
    """
    with torch.no_grad():
        logits = _model(tensor_image)         # expected: [T, B, C]
        pred = logits.argmax(2).squeeze(1)    # [T]
    return decode(pred, _idx_to_char)


if __name__ == "__main__":
    print("inference.py loaded OK")
