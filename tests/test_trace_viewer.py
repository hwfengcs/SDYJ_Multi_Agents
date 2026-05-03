from pathlib import Path


def test_static_trace_viewer_contains_core_hooks():
    viewer_path = Path(__file__).resolve().parents[1] / "SDYJ_Agents" / "web" / "trace_viewer.html"
    html = viewer_path.read_text(encoding="utf-8")

    assert "SDYJ Trace Viewer" in html
    assert "function parseTraceText" in html
    assert "function normalizeEvents" in html
    assert "events.jsonl" in html
    assert "trace.json" in html
    assert "data-event-id" in html


def test_docs_trace_viewer_entrypoint_points_to_viewer():
    entrypoint = Path(__file__).resolve().parents[1] / "docs" / "trace-viewer-demo.html"
    html = entrypoint.read_text(encoding="utf-8")

    assert "../SDYJ_Agents/web/trace_viewer.html" in html
    assert "SDYJ_Agents/web/trace_viewer.html" in html
    assert "location.replace(target)" in html


def test_pages_workflow_publishes_trace_viewer():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pages.yml"
    content = workflow.read_text(encoding="utf-8")

    assert "actions/deploy-pages" in content
    assert "SDYJ_Agents/web/trace_viewer.html" in content
    assert "cp -R docs/." in content
