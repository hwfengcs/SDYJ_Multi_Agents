from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_hosted_demo_badges_are_hidden_until_verified():
    for filename in ["README.md", "README_EN.md"]:
        content = _read(filename)

        assert "Hosted demo badges are prepared but hidden until the URLs are verified" in content
        assert "[![Open in Spaces]" in content
        assert "https://huggingface.co/spaces/<owner>/sdyj-multi-agents" in content
        assert "https://hwfengcs.github.io/SDYJ_Multi_Agents/trace-viewer-demo.html" in content


def test_pages_deploy_doc_tracks_placeholder_url_and_checklist():
    content = _read("docs/pages-deploy.md")
    index = _read("docs/index.html")

    assert "https://hwfengcs.github.io/SDYJ_Multi_Agents/" in content
    assert "Treat those as placeholders" in content
    assert "Settings -> Pages" in content
    assert "Source** to **GitHub Actions" in content
    assert "tests/test_trace_viewer.py tests/test_deploy_readiness.py" in content
    assert "pages-deploy.md" in index


def test_hf_spaces_doc_has_predeploy_checks_and_badge_gate():
    content = _read("docs/huggingface-spaces-deploy.md")

    assert "hidden placeholder badge comment" in content
    assert "tests/test_hf_entrypoint.py tests/test_web_smoke.py tests/test_deploy_readiness.py" in content
    assert "DEEPSEEK_API_KEY" in content
    assert "TAVILY_API_KEY" in content
    assert "HF secrets" in content


def test_publish_workflow_has_target_and_tag_gate():
    workflow = _read(".github/workflows/publish.yml")
    docs = _read("docs/release-process.md")

    assert "Validate publish target" in workflow
    assert "PUBLISH_TARGET: ${{ github.event.inputs.target || 'pypi' }}" in workflow
    assert "target not in {\"testpypi\", \"pypi\"}" in workflow
    assert 'expected_tag = f"v{version}"' in workflow
    assert "PyPI publishing must run from the exact package tag" in workflow
    assert "Workflow safety gates" in docs


def test_ghcr_workflow_is_manual_and_no_push_by_default():
    workflow = _read(".github/workflows/ghcr.yml")
    docs = _read("docs/ghcr.md")
    index = _read("docs/index.html")

    assert "workflow_dispatch:" in workflow
    assert "default: false" in workflow
    assert "push: ${{ inputs.publish }}" in workflow
    assert "docker/build-push-action@v6" in workflow
    assert "ghcr.io/hwfengcs/sdyj-multi-agents" in workflow
    assert "publish=false" in docs
    assert "publish=true" in docs
    assert "ghcr.md" in index
