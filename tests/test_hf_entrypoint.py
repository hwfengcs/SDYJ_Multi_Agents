import importlib.util
from pathlib import Path

from SDYJ_Agents.web.app import main as web_main


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_app_entrypoint_delegates_to_web_app():
    root = Path(__file__).resolve().parents[1]
    module = _load_module_from_path("sdyj_hf_root_app", root / "app.py")

    assert callable(module.main)
    assert module.main is web_main


def test_streamlit_app_entrypoint_delegates_to_web_app():
    root = Path(__file__).resolve().parents[1]
    module = _load_module_from_path("sdyj_hf_streamlit_app", root / "streamlit_app.py")

    assert callable(module.main)
    assert module.main is web_main


def test_huggingface_space_readme_template_is_streamlit():
    template_path = Path(__file__).resolve().parents[1] / "docs" / "huggingface-space" / "README.md"
    content = template_path.read_text(encoding="utf-8")

    assert "sdk: streamlit" in content
    assert "app_file: app.py" in content
    assert "SDYJ Multi Agents" in content


def test_huggingface_requirements_include_runtime_dependencies():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")

    required = [
        "langgraph",
        "openai",
        "anthropic",
        "google-genai",
        "tavily-python",
        "arxiv",
        "rich",
        "streamlit",
        "httpx",
        "requests",
        "pydantic",
        "python-dotenv",
        "jinja2",
    ]

    for package in required:
        assert package in requirements
