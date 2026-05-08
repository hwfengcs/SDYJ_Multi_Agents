# GitHub Pages Deployment

The repository already contains a Pages workflow that publishes `docs/` plus
the static Trace Viewer. This page is the checklist for turning that prepared
asset into a verified public URL without overstating what has been deployed.

## Prepared Assets

- Workflow: `.github/workflows/pages.yml`.
- Landing page: `docs/index.html`.
- Redirect entrypoint: `docs/trace-viewer-demo.html`.
- Viewer payload: `SDYJ_Agents/web/trace_viewer.html`.

The expected URL shape after repository settings are enabled is:

```text
https://hwfengcs.github.io/SDYJ_Multi_Agents/
https://hwfengcs.github.io/SDYJ_Multi_Agents/trace-viewer-demo.html
```

Treat those as placeholders until the workflow has run and both pages are
opened successfully in a browser.

## Local Static Checks

Run these before enabling or re-running Pages:

```bash
python -m pytest tests/test_trace_viewer.py tests/test_deploy_readiness.py
python -m ruff check tests/test_trace_viewer.py tests/test_deploy_readiness.py
```

These checks verify that `docs/index.html` only links to existing local docs,
the Pages workflow copies the packaged Trace Viewer, and the README badge
placeholders remain hidden until the URLs are verified.

## Repository Settings

1. Open GitHub repository **Settings -> Pages**.
2. Set **Build and deployment -> Source** to **GitHub Actions**.
3. Save the setting.
4. Run the **Pages** workflow manually, or merge a docs change to `main`.

## Public URL Verification

After the workflow succeeds:

1. Open the Pages root URL.
2. Open `trace-viewer-demo.html` from the root page.
3. Drag in a real `trace.json` or `events.jsonl` from `outputs/runs/<run-id>/`.
4. Confirm the timeline, event filters, and event detail panel render.
5. Record the verified URL and date in `docs/live-run-notes.md`.
6. Unhide the README / README_EN Trace Viewer badge only after the URL works.

If the site returns 404, keep the README badge hidden and record the blocker
instead of claiming a hosted demo.
