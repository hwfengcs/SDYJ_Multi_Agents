from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publish_workflow_validates_built_wheel_install():
    workflow = ROOT / ".github" / "workflows" / "publish.yml"
    content = workflow.read_text(encoding="utf-8")

    assert 'python -m pip install -e ".[dev,web]"' in content
    assert "tests/test_trace_viewer.py" in content
    assert "tests/test_hf_entrypoint.py" in content
    assert "tests/test_docker_assets.py" in content
    assert "tests/test_deploy_readiness.py" in content
    assert "Validate wheel install smoke" in content
    assert "python -m venv .wheel-smoke" in content
    assert 'python -m pip install "${WHEEL}[all]"' in content
    assert "sdyj --version" in content
    assert "sdyj --help" in content
    assert "DEEPSEEK_API_KEY=dummy sdyj doctor --provider deepseek" in content
    assert 'resources.files("SDYJ_Agents.prompts")' in content
    assert 'resources.files("SDYJ_Agents.web")' in content
    assert 'resources.files("SDYJ_Agents.benchmarks.fixtures")' in content


def test_release_process_documents_clean_wheel_smoke():
    content = (ROOT / "docs" / "release-process.md").read_text(encoding="utf-8")

    assert "clean" in content
    assert "wheel-install smoke" in content
    assert 'python -m pip install "dist/<built-wheel>.whl[all]"' in content
    assert "DEEPSEEK_API_KEY=dummy sdyj doctor --provider deepseek" in content
    assert "do not use real provider secrets" in content


def test_publish_workflow_skips_pypi_for_github_prereleases():
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "release-process.md").read_text(encoding="utf-8")

    assert "!github.event.release.prerelease" in workflow
    assert "skips PyPI publishing for GitHub Releases marked as prerelease" in docs
