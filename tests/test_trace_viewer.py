from html.parser import HTMLParser
from pathlib import Path
import shutil
from urllib.parse import urlparse


class _HrefParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


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


def test_docs_index_links_pages_artifacts():
    index = Path(__file__).resolve().parents[1] / "docs" / "index.html"
    html = index.read_text(encoding="utf-8")

    assert "trace-viewer-demo.html" in html
    assert "benchmark-results-public.md" in html
    assert "huggingface-spaces-deploy.md" in html


def test_docs_index_local_links_exist_in_pages_site():
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    html = (docs_dir / "index.html").read_text(encoding="utf-8")
    parser = _HrefParser()
    parser.feed(html)

    missing = []
    for href in parser.hrefs:
        parsed = urlparse(href)
        if parsed.scheme or parsed.netloc or parsed.path.startswith("#"):
            continue
        target = (docs_dir / parsed.path).resolve()
        if not target.exists():
            missing.append(href)

    assert not missing


def test_pages_artifact_layout_matches_workflow_copy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    docs_dir = root / "docs"
    site_dir = tmp_path / "_site"
    viewer_target = site_dir / "SDYJ_Agents" / "web" / "trace_viewer.html"

    shutil.copytree(docs_dir, site_dir)
    viewer_target.parent.mkdir(parents=True)
    shutil.copy2(root / "SDYJ_Agents" / "web" / "trace_viewer.html", viewer_target)

    assert (site_dir / "index.html").is_file()
    assert (site_dir / "trace-viewer-demo.html").is_file()
    assert viewer_target.is_file()

    html = (site_dir / "index.html").read_text(encoding="utf-8")
    parser = _HrefParser()
    parser.feed(html)

    missing = []
    for href in parser.hrefs:
        parsed = urlparse(href)
        if parsed.scheme or parsed.netloc or parsed.path.startswith("#"):
            continue
        target = (site_dir / parsed.path).resolve()
        if not target.exists():
            missing.append(href)

    assert not missing
