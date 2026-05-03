from pathlib import Path


def test_root_app_entrypoint_delegates_to_web_app():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    content = app_path.read_text(encoding="utf-8")

    assert "from SDYJ_Agents.web.app import main" in content
    assert "main()" in content
