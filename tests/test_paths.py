from src.paths import model_asset, repo_root


def test_repo_root_exists():
    r = repo_root()
    assert r.exists()
    assert (r / "src").exists()

def test_model_asset_resolves_to_string():
    p = model_asset("LP-detection.pt")
    assert isinstance(p, str)

