"""``log_evaluation`` goes through the same logger as the native library.

upstream: python-package/lightgbm/callback.py (v4.7.0) _LogEvaluationCallback calls ``_log_info``,
so the line is printed by default and reaches a logger registered with ``register_logger``.
"""

import numpy as np
import pytest

import bankai_lightgbm as lgb
from bankai_lightgbm import basic


def _train_with_log_evaluation(**kwargs):
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    y = (X[:, 0] > 0.5).astype(int)
    train_set = lgb.Dataset(X, y)
    valid_set = lgb.Dataset(X, y, reference=train_set)
    return lgb.train(
        {"objective": "binary", "verbose": -1},
        train_set,
        num_boost_round=4,
        valid_sets=[valid_set],
        callbacks=[lgb.log_evaluation(2)],
    )


@pytest.fixture
def restore_default_logger():
    # Put back the exact logger state, including the default printing logger, not a re-registration.
    saved = (basic._LOGGER, basic._INFO_METHOD_NAME, basic._WARNING_METHOD_NAME)
    yield
    basic._LOGGER, basic._INFO_METHOD_NAME, basic._WARNING_METHOD_NAME = saved


def test_log_evaluation_prints_by_default(capsys):
    _train_with_log_evaluation()
    out = capsys.readouterr().out
    assert "[2]\tvalid_0's binary_logloss: " in out
    assert "[4]\tvalid_0's binary_logloss: " in out


def test_log_evaluation_reaches_registered_logger(restore_default_logger):
    messages: list[str] = []

    class Capture:
        def info(self, msg: str) -> None:
            messages.append(msg)

        def warning(self, msg: str) -> None:
            messages.append(msg)

    lgb.register_logger(Capture())
    _train_with_log_evaluation()
    assert any(m.startswith("[2]\tvalid_0's binary_logloss: ") for m in messages)
    assert any(m.startswith("[4]\tvalid_0's binary_logloss: ") for m in messages)
