# Live Run Notes

This file records real-provider smoke tests and external blockers for the v0.6
release track. Do not paste API keys or raw secrets here.

## 2026-05-03 - DeepSeek provider smoke

- Status: partial pass.
- Provider/model: `deepseek` / `deepseek-v4-flash`.
- Query: `How should RAG agents be evaluated for reliability?`
- Command:

```bash
sdyj research "How should RAG agents be evaluated for reliability?" \
  --provider deepseek \
  --model deepseek-v4-flash \
  --max-iterations 1 \
  --auto-approve \
  --output-dir outputs/live_smoke \
  --output-format markdown
```

- Source run id: `20260503_032501_475f234e`.
- Replay run id: `20260503_033029_0a3985e9`.
- Trace bundle: `outputs/live_smoke/runs/20260503_032501_475f234e/`.
- Replay bundle: `outputs/live_smoke/runs/20260503_033029_0a3985e9/`.
- Report path: `outputs/live_smoke/research_report_20260503_112644.md`.

Validation:

```bash
sdyj inspect-run 20260503_032501_475f234e --output-dir outputs/live_smoke --timeline
sdyj replay 20260503_032501_475f234e --output-dir outputs/live_smoke
sdyj diff-runs 20260503_032501_475f234e 20260503_033029_0a3985e9 --output-dir outputs/live_smoke
```

Observed trace signals:

- `inspect-run` shows 7 LLM calls, 4 tool calls, verifier events, and cost metrics.
- Cost estimate: `total_tokens=22794`, `total_cost_usd=0.009217`.
- Verifier result: `overall_quality=0.77`, `should_revise=false`, weakest dimension `coverage`.
- Replay completed without live API calls after hardening error-tool replay.
- `diff-runs` shows expected mode/provider/runtime/cost deltas between `research` and `replay`.

External blockers:

- `TAVILY_API_KEY` is missing or placeholder in local `.env`, so the full DeepSeek + Tavily live search path is still blocked.
- The run used real DeepSeek LLM calls and arXiv retrieval, while Tavily calls were recorded as `source unavailable or unsupported`.
- Next live verification should rerun the same command with a real `TAVILY_API_KEY`, then confirm Tavily result batches appear in the trace.

Fixes discovered during this run:

- Windows GBK console rendering could fail after report generation and before trace saving; fixed by saving artifacts before terminal rendering and by configuring console streams to replace unencodable characters.
- Deterministic replay did not preserve missing-source tool failures, which could trigger an extra reflection call and exhaust recorded LLM responses; fixed by replaying missing-source calls as `None` so Researcher records the same unavailable-source outcome.

## 2026-05-03 - MCP filesystem demo smoke

- Status: pass for local filesystem MCP; GitHub MCP blocked by missing `GITHUB_PERSONAL_ACCESS_TOKEN`.
- Dry-run command:

```bash
python examples/mcp_demos/mcp_filesystem_demo.py --root .
```

- Dependency check:

```bash
python examples/mcp_demos/mcp_filesystem_demo.py --root . --check
```

Observed `npx=true` and `mcp_python_sdk=true`.

- Real list-tools command:

```bash
python examples/mcp_demos/mcp_filesystem_demo.py --root . --list-tools
```

The server exposed `search_files`, `read_text_file`, `list_directory`,
`directory_tree`, and related filesystem tools. The demo now defaults to:

```bash
MCP_TOOL_NAME=search_files
MCP_TOOL_ARGS_JSON={"path":"<repo-root>","pattern":"{query}"}
```

Direct `MCPClient.search("README.md")` through `search_files` returned one
normalized result for `README.md`.

External blockers:

- GitHub MCP `--check` reports `GITHUB_PERSONAL_ACCESS_TOKEN=false`; do not run
  `--list-tools` for GitHub until a token is configured.

## 2026-05-03 - Hosted demo / release preparation

- Hugging Face Spaces code-side preparation: root `app.py` exists, `requirements.txt` now includes `streamlit>=1.36.0`, and the Space README template lives at `docs/huggingface-space/README.md`.
- GitHub Pages code-side preparation: `docs/index.html` links Trace Viewer, benchmark docs, public benchmark status, Spaces deploy docs, and release process docs. The workflow still requires repository Settings -> Pages -> Source: GitHub Actions.
- Release packaging preparation: `MANIFEST.in` explicitly includes docs, examples, root Streamlit entrypoints, env examples, MCP config examples, prompts, web HTML, and benchmark fixtures.

External blockers:

- HF Space URL is not available until the Space is created and secrets are configured on Hugging Face.
- GitHub Pages URL is not available until Pages is enabled in repository settings and the workflow is run.
- TestPyPI/PyPI URLs require Trusted Publisher setup and a publish workflow run.

## 2026-05-03 - Follow-up preflight

- Status: local verification pass; full Tavily live-search smoke remains blocked.
- Provider/model target: `deepseek` / `deepseek-v4-flash`.

Validation:

```bash
sdyj doctor --provider deepseek
python -m pytest
python -m ruff check SDYJ_Agents tests examples
```

Observed results:

- `sdyj doctor --provider deepseek` reports a usable DeepSeek key.
- `sdyj doctor --provider deepseek` reports `TAVILY_API_KEY` as missing for live web search.
- Local `.env` check found `TAVILY_API_KEY` empty or placeholder; no secret value was printed or recorded.
- `python -m pytest`: `109 passed, 1 xfailed`.
- `python -m ruff check SDYJ_Agents tests examples`: all checks passed.

External blocker:

- Full DeepSeek + Tavily + arXiv live research is still blocked until a real `TAVILY_API_KEY` is configured. Continue with local HF Spaces, Pages, packaging, and benchmark preparation.

## 2026-05-08 - Follow-up preflight

- Status: local verification pass; full Tavily live-search smoke remains blocked.
- Provider/model target: `deepseek` / `deepseek-v4-flash`.

Validation:

```bash
git status --short --branch
git log --oneline -12
sdyj doctor --provider deepseek
python -m pytest
python -m ruff check SDYJ_Agents tests examples
```

Observed results:

- Branch is `feat/v0.6-self-verifying`.
- Recent local commits include `docs(live): record Tavily smoke blocker`, `test(deploy): cover hosted entrypoints`, `ci(release): add package smoke gates`, and `feat(benchmark): label public smoke artifacts`.
- `sdyj doctor --provider deepseek` reports a usable DeepSeek key.
- `sdyj doctor --provider deepseek` reports `TAVILY_API_KEY` as missing for live web search.
- Local `.env` check found `TAVILY_API_KEY` missing; no secret value was printed or recorded.
- `python -m pytest`: `114 passed, 1 xfailed`.
- `python -m ruff check SDYJ_Agents tests examples`: all checks passed.

External blocker:

- Full DeepSeek + Tavily + arXiv live research is still blocked until a real `TAVILY_API_KEY` is configured. Continue with local HF Spaces, Pages, packaging, benchmark, and Docker preparation.
