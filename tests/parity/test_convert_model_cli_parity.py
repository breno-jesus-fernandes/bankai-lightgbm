"""CLI C++ code generation must match goldens from upstream v4.7.0's CLI."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "convert_model"
CASES = [
    "numerical",
    "categorical",
    "categorical_many",
    "missing",
    "zero_missing",
    "stump",
    "multiclass",
    "linear",
]


def _convert(model: Path, output: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(ROOT / "Cargo.toml"),
            "-p",
            "lgbm-cli",
            "--",
            "task=convert_model",
            f"input_model={model}",
            f"convert_model={output}",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("case", CASES)
def test_convert_model_matches_upstream(case: str, tmp_path: Path) -> None:
    output = tmp_path / "generated.cpp"
    result = _convert(FIXTURES / f"{case}.txt", output, tmp_path)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == (FIXTURES / f"{case}.cpp").read_text()


def test_convert_model_wraps_existing_file_like_upstream(tmp_path: Path) -> None:
    output = tmp_path / "existing.cpp"
    original = "// existing prediction code\n"
    output.write_text(original)
    result = _convert(FIXTURES / "numerical.txt", output, tmp_path)
    assert result.returncode == 0, result.stderr
    golden = (FIXTURES / "numerical.cpp").read_text()
    assert output.read_text() == (
        "#define USE_HARD_CODE 0\n#ifndef USE_HARD_CODE\n"
        + original
        + "\n#else\n"
        + golden
        + "#endif\n"
    )


def test_convert_model_default_output(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(ROOT / "Cargo.toml"),
            "-p",
            "lgbm-cli",
            "--",
            "task=convert_model",
            f"input_model={FIXTURES / 'numerical.txt'}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "gbdt_prediction.cpp").read_text() == (
        FIXTURES / "numerical.cpp"
    ).read_text()


def test_convert_model_missing_input_errors(tmp_path: Path) -> None:
    output = tmp_path / "out.cpp"
    result = _convert(tmp_path / "missing.txt", output, tmp_path)
    assert result.returncode != 0
    assert not output.exists()


def test_convert_model_unwritable_output_errors(tmp_path: Path) -> None:
    result = _convert(FIXTURES / "numerical.txt", tmp_path, tmp_path)
    assert result.returncode != 0


def test_convert_model_preserves_existing_non_utf8_bytes(tmp_path: Path) -> None:
    output = tmp_path / "existing.cpp"
    original = b"// existing source in a legacy encoding: \xff\n"
    output.write_bytes(original)
    result = _convert(FIXTURES / "numerical.txt", output, tmp_path)
    assert result.returncode == 0, result.stderr
    golden = (FIXTURES / "numerical.cpp").read_bytes()
    assert output.read_bytes() == (
        b"#define USE_HARD_CODE 0\n#ifndef USE_HARD_CODE\n"
        + original
        + b"\n#else\n"
        + golden
        + b"#endif\n"
    )
