"""Generate convert_model goldens with the upstream CLI built from UPSTREAM_VERSION.

Usage: python scripts/gen_convert_model_fixtures.py --upstream-cli /path/to/lightgbm
Build the CLI from the pinned tag, including its pinned submodules. The wheel alone does
not expose ModelToIfElse. These fixtures are independent of bankai's implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import lightgbm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "tests" / "fixtures" / "convert_model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-cli", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "numerical",
            "categorical",
            "categorical_many",
            "missing",
            "zero_missing",
            "stump",
            "multiclass",
            "linear",
        ],
    )
    args = parser.parse_args()
    pinned = (ROOT / "UPSTREAM_VERSION").read_text().strip()
    if lightgbm.__version__ != pinned.removeprefix("v"):
        raise RuntimeError(f"expected wheel {pinned}, got {lightgbm.__version__}")
    executable = args.upstream_cli.resolve()
    DEST.mkdir(parents=True, exist_ok=True)
    cases = args.cases
    for case in cases:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(128, 3))
        y = 2 * x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=128)
        categorical: list[int] = []
        params: dict[str, object] = {
            "objective": "regression",
            "num_leaves": 4,
            "min_data_in_leaf": 4,
            "min_data_in_bin": 1,
            "verbosity": -1,
            "num_threads": 1,
            "deterministic": True,
            "seed": 42,
        }
        if case == "categorical":
            x[:, 1] = rng.integers(0, 4, size=128)
            y = 3 * (x[:, 1] == 1) + x[:, 0]
            categorical = [1]
            params["max_cat_to_onehot"] = 10
        elif case == "categorical_many":
            x[:, 1] = rng.integers(0, 70, size=128)
            y = 3 * (x[:, 1] % 3 == 1) + x[:, 0]
            categorical = [1]
            params.update(cat_smooth=0, cat_l2=0, min_data_per_group=1)
        elif case == "missing":
            x[rng.random(x.shape) < 0.2] = np.nan
        elif case == "zero_missing":
            x[rng.random(x.shape) < 0.3] = 0
            params["zero_as_missing"] = True
        elif case == "stump":
            y[:] = 1.234567891234567
            params["min_data_in_leaf"] = 128
        elif case == "multiclass":
            y = rng.integers(0, 3, size=128).astype(np.float64)
            params.update(objective="multiclass", num_class=3)
        elif case == "linear":
            params["linear_tree"] = True
        model = lightgbm.train(
            params, lightgbm.Dataset(x, y, categorical_feature=categorical), num_boost_round=2
        )
        model_path = DEST / f"{case}.txt"
        output_path = DEST / f"{case}.cpp"
        if output_path.exists():
            raise FileExistsError(f"refusing to wrap an existing golden: {output_path}")
        model.save_model(model_path)
        subprocess.run(
            [
                str(executable),
                "task=convert_model",
                f"input_model={model_path}",
                f"convert_model={output_path}",
                "verbosity=-1",
            ],
            check=True,
        )
    (DEST / "provenance.json").write_text(
        json.dumps(
            {
                "upstream_version": pinned,
                "wheel_version": lightgbm.__version__,
                "cli_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "cases": sorted(path.stem for path in DEST.glob("*.cpp")),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
