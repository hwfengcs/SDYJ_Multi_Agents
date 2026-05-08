from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dockerfile_installs_package_and_exposes_streamlit():
    content = _read("Dockerfile")

    assert "FROM python:3.12-slim" in content
    assert "ARG SDYJ_EXTRAS=web" in content
    assert 'python -m pip install ".[${SDYJ_EXTRAS}]"' in content
    assert "USER sdyj" in content
    assert "EXPOSE 8501" in content
    assert '"streamlit", "run", "app.py"' in content


def test_dockerignore_keeps_secrets_and_artifacts_out_of_context():
    content = _read(".dockerignore")
    required_patterns = {
        ".env",
        "*.env",
        ".env.*",
        "!.env.example",
        ".git/",
        "outputs/",
        "dist/",
        "build/",
        "*.egg-info/",
        "tests/",
    }

    patterns = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert required_patterns <= patterns


def test_compose_passes_expected_runtime_environment_without_values():
    content = _read("docker-compose.yml")

    assert "image: sdyj:0.6" in content
    assert '"8501:8501"' in content
    assert "DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:-}" in content
    assert "TAVILY_API_KEY: ${TAVILY_API_KEY:-}" in content
    assert "OUTPUT_DIR: /app/outputs" in content
    assert "sdyj_outputs:/app/outputs" in content


def test_docker_docs_and_manifest_are_linked():
    docs = _read("docs/docker.md")
    index = _read("docs/index.html")
    manifest = _read("MANIFEST.in")

    assert "docker build -t sdyj:0.6 ." in docs
    assert "docker build --build-arg SDYJ_EXTRAS=all -t sdyj:0.6-all ." in docs
    assert "docker compose up --build" in docs
    assert "docker run --rm -e DEEPSEEK_API_KEY=dummy sdyj:0.6 sdyj doctor --provider deepseek" in docs
    assert "docker compose config" in docs
    assert "docker.md" in index
    assert "include Dockerfile" in manifest
    assert "include docker-compose.yml" in manifest
    assert "include .dockerignore" in manifest
