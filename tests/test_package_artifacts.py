from importlib import resources


def test_runtime_package_artifacts_are_readable():
    prompt_dir = resources.files("SDYJ_Agents.prompts")
    web_dir = resources.files("SDYJ_Agents.web")
    fixture_dir = resources.files("SDYJ_Agents.benchmarks.fixtures")

    report_prompt = prompt_dir / "rapporteur_revise.md"
    trace_viewer = web_dir / "trace_viewer.html"
    gaia_fixture = fixture_dir / "gaia_mini.jsonl"

    assert report_prompt.is_file()
    assert "evidence" in report_prompt.read_text(encoding="utf-8").lower()
    assert trace_viewer.is_file()
    assert "SDYJ Trace Viewer" in trace_viewer.read_text(encoding="utf-8")
    assert gaia_fixture.is_file()
    assert "gaia-mini-001" in gaia_fixture.read_text(encoding="utf-8")
