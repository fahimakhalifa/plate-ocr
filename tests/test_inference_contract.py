import torch


def test_predict_plate_contract_monkeypatch(monkeypatch):
    # Import module
    import src.inference as inf

    # Fake model returns logits shaped [T, B, C] matching your model.forward
    # We'll craft logits so argmax gives indices [1,2,3]
    T, B, C = 3, 1, 5
    logits = torch.zeros((T, B, C))
    logits[0, 0, 1] = 10
    logits[1, 0, 2] = 10
    logits[2, 0, 3] = 10

    class FakeModel:
        def __call__(self, x):
            return logits

    # Patch internals
    inf._model = FakeModel()
    inf._idx_to_char = {1: "A", 2: "B", 3: "C"}

    # Input tensor shape doesn't matter for FakeModel, but keep realistic
    x = torch.zeros((1, 1, 32, 160))
    out = inf.predict_plate(x)
    assert out == "ABC"
