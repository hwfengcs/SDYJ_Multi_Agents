import tomllib
from pathlib import Path

from SDYJ_Agents import __version__


def test_package_version_matches_pyproject():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert __version__ == metadata["project"]["version"]


def test_manifest_includes_deployment_and_fixture_files():
    manifest_path = Path(__file__).resolve().parents[1] / "MANIFEST.in"
    content = manifest_path.read_text(encoding="utf-8")

    assert "include app.py" in content
    assert "recursive-include docs *" in content
    assert "recursive-include scripts *.py" in content
    assert "benchmarks/fixtures *.jsonl" in content


def test_pyproject_uses_modern_license_metadata():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert metadata["project"]["license"] == "MIT"
    assert "License :: OSI Approved :: MIT License" not in metadata["project"]["classifiers"]
