from src.dataset import decode


def test_decode_removes_blanks_and_repeats():
    # CTC-style: remove blanks (0) and collapse consecutive duplicates only
    # pred: 0 1 1 0 2 2 3 0 3 -> 1,2,3,3 -> "ABCC"
    pred = [0, 1, 1, 0, 2, 2, 3, 0, 3]
    idx_to_char = {1: "A", 2: "B", 3: "C"}

    class FakeTensor:
        def __init__(self, xs):
            self.xs = xs
        def __iter__(self):
            for x in self.xs:
                yield FakeItem(x)

    class FakeItem:
        def __init__(self, v):
            self.v = v
        def item(self):
            return self.v

    out = decode(FakeTensor(pred), idx_to_char)
    assert out == "ABCC"