import torch

from src.dataset import decode


def test_decode_removes_ctc_blanks_and_repeated_characters():
    idx_to_char = {
        1: "A",
        2: "B",
        3: "1",
    }

    prediction = torch.tensor([0, 1, 1, 0, 2, 2, 3, 3, 0])

    assert decode(prediction, idx_to_char) == "AB1"


def test_decode_ignores_unknown_indices():
    idx_to_char = {
        1: "A",
        2: "B",
    }

    prediction = torch.tensor([1, 99, 99, 0, 2])

    assert decode(prediction, idx_to_char) == "AB"


def test_decode_handles_empty_prediction():
    idx_to_char = {
        1: "A",
    }

    prediction = torch.tensor([])

    assert decode(prediction, idx_to_char) == ""
