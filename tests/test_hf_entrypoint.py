from pathlib import Path


def test_root_app_entrypoint_delegates_to_web_app():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    content = app_path.read_text(encoding="utf-8")

    assert "from SDYJ_Agents.web.app import main" in content
    assert "main()" in content


def test_huggingface_space_readme_template_is_streamlit():
    template_path = Path(__file__).resolve().parents[1] / "docs" / "huggingface-space" / "README.md"
    content = template_path.read_text(encoding="utf-8")

    assert "sdk: streamlit" in content
    assert "app_file: app.py" in content
    assert "SDYJ Multi Agents" in content
