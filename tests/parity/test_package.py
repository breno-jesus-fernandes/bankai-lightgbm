import bankai_lightgbm


def test_package_targets_pinned_upstream() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    assert (root / "UPSTREAM_VERSION").read_text().strip() == bankai_lightgbm.UPSTREAM_VERSION
