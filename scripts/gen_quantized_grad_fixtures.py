"""Record upstream leaf statistics for nearest-rounded quantized gradients.

The expected statistics come from lightgbm==4.7.0, not a Python quantizer.
Run with the project's uv environment. Existing fixtures are not overwritten.
"""

import json
import platform
from collections.abc import Callable
from pathlib import Path

import lightgbm as lgb
import numpy as np


def fixed_objective(
    gradients: np.ndarray, hessians: np.ndarray
) -> Callable[[np.ndarray, lgb.Dataset], tuple[np.ndarray, np.ndarray]]:
    def objective(_predictions: np.ndarray, _dataset: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        return gradients, hessians

    return objective


def main() -> None:
    if lgb.__version__ != "4.7.0":
        raise RuntimeError("the oracle must be lightgbm==4.7.0")
    cases = []
    for constant in (False, True):
        for bins in (2, 3, 4, 7, 16, 31, 64, 127):
            n = 257
            x = np.arange(n, dtype=np.float64).reshape(-1, 1)
            # Non-power-of-two scales and values near rounding boundaries.
            gradients = ((np.arange(n) % 29) - 14).astype(np.float32) * np.float32(0.37)
            gradients += np.where(np.arange(n) < 121, -1.3, 1.7).astype(np.float32)
            hessians = (
                np.ones(n, dtype=np.float32)
                if constant
                else (0.13 + (np.arange(n) % 17) * 0.19).astype(np.float32)
            )

            params = {
                "objective": "regression" if constant else fixed_objective(gradients, hessians),
                "use_quantized_grad": True,
                "stochastic_rounding": False,
                "quant_train_renew_leaf": False,
                "num_grad_quant_bins": bins,
                "num_leaves": 2,
                "min_data_in_leaf": 1,
                "min_sum_hessian_in_leaf": 0.001,
                "max_bin": 63,
                "learning_rate": 1.0,
                "boost_from_average": False,
                "deterministic": True,
                "num_threads": 1,
                "force_col_wise": True,
                "verbose": -1,
            }
            # Default train() reloads text, whose split_gain is printed at six
            # significant digits. Capture the native score_t gain before that reload.
            model = lgb.train(
                params,
                lgb.Dataset(x, label=-gradients),
                num_boost_round=1,
                keep_training_booster=True,
            )
            tree = model.dump_model()["tree_info"][0]["tree_structure"]
            cases.append(
                {
                    "constant_hessian": constant,
                    "bins": bins,
                    "gradients": gradients.tolist(),
                    "hessians": hessians.tolist(),
                    "left_rows": np.flatnonzero(x[:, 0] <= tree["threshold"]).tolist(),
                    "left_output": tree["left_child"]["leaf_value"],
                    "right_output": tree["right_child"]["leaf_value"],
                    "left_weight": tree["left_child"]["leaf_weight"],
                    "right_weight": tree["right_child"]["leaf_weight"],
                    "split_gain": tree["split_gain"],
                }
            )
    target = Path(__file__).resolve().parents[1] / "tests/fixtures/quantized_gradients.json"
    with target.open("x") as out:
        json.dump(
            {"upstream": lgb.__version__, "platform": platform.platform(), "cases": cases},
            out,
            indent=2,
        )
        out.write("\n")


if __name__ == "__main__":
    main()
