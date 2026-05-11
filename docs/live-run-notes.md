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

## 2026-05-08 - Docker packaging preflight

- Status: Docker assets added; local container runtime verification is blocked.

Commands attempted:

```bash
docker build -t sdyj:0.6 .
docker run --rm sdyj:0.6 sdyj --help
```

Observed results:

- Both commands failed before project build/run because `docker` is not installed or not on `PATH` on this machine.
- Static Docker asset validation passed through `tests/test_docker_assets.py`.
- Related docs/package tests passed through `tests/test_trace_viewer.py`, `tests/test_version.py`, and `tests/test_package_artifacts.py`.
- `python -m ruff check tests/test_docker_assets.py`: all checks passed.

Follow-up:

- Rerun the same Docker build and CLI smoke on a Docker-enabled host.

## 2026-05-08 - Hosted release readiness follow-up

- Status: local release/hosting readiness improved; external live smoke remains
  blocked by missing Tavily.
- `.env` check: `TAVILY_API_KEY=missing`. No secret value was printed or
  recorded.

Validation:

```bash
sdyj doctor --provider deepseek
python -m pytest
python -m ruff check SDYJ_Agents tests examples
python -m pytest tests/test_deploy_readiness.py tests/test_trace_viewer.py tests/test_publish_workflow.py tests/test_docker_assets.py tests/test_hf_entrypoint.py tests/test_web_smoke.py
python -m ruff check tests/test_deploy_readiness.py tests/test_trace_viewer.py tests/test_publish_workflow.py tests/test_docker_assets.py tests/test_hf_entrypoint.py tests/test_web_smoke.py
```

Observed results:

- `sdyj doctor --provider deepseek`: DeepSeek key present; Tavily missing; MCP
  source not configured.
- `python -m pytest`: `121 passed, 1 xfailed`.
- `python -m ruff check SDYJ_Agents tests examples`: all checks passed.
- Deploy readiness targeted tests: `25 passed, 1 xfailed`; targeted ruff passed.

Local release-readiness changes:

- Added `docs/pages-deploy.md` with the GitHub Pages settings checklist,
  unverified URL placeholders, and public Trace Viewer verification criteria.
- Added hidden README / README_EN hosted-demo badge placeholders so badges are
  ready but not exposed before real URLs are verified.
- Added `docs/ghcr.md` and a manual-only `.github/workflows/ghcr.yml` Docker
  image workflow. The workflow builds by default and pushes only when a
  maintainer explicitly runs it with `publish=true`.
- Added a PyPI target/tag gate to `publish.yml`, requiring PyPI publishing to
  run from the exact `v<pyproject version>` tag while still allowing TestPyPI
  branch-based alpha smoke checks.
- Hardened PyPI workflow behavior so GitHub Releases marked as prerelease do
  not publish to PyPI, and release smoke now includes hosted/Pages/Docker
  static readiness tests.

External blockers:

- Full DeepSeek + Tavily + arXiv live research remains blocked until a usable
  `TAVILY_API_KEY` is configured.
- GitHub Pages and Hugging Face Spaces still require platform-side enablement
  and verified public URLs before badges should be unhidden.
- GHCR image publishing still requires an explicit manual workflow run with
  maintainer approval; no image was pushed in this local batch.

## 2026-05-08 - GAIA / Hugging Face access preflight

- Status: real public GAIA score remains blocked; no score is claimed.
- Target suite: GAIA Level 1 via `gaia-benchmark/GAIA`.

Planned command once Hugging Face login and dataset access are confirmed:

```bash
sdyj benchmark external \
  --suite gaia \
  --source hf \
  --hf-dataset gaia-benchmark/GAIA \
  --hf-config 2023_level1 \
  --split validation \
  --limit 5 \
  --predictions outputs/gaia_predictions.jsonl \
  --output-dir outputs/public_benchmarks
```

Current local fallback:

```bash
sdyj benchmark external --suite gaia --source local --limit 3 --output-dir outputs/public_benchmarks --fail-under 1.0
```

Observed results:

- Synthetic GAIA-style smoke remains a harness/artifact-layout check only.
- Latest local smoke run id: `gaia_20260508_120004`; accuracy `1.0000`,
  prediction coverage `1.0000`, missing predictions `0`.
- Offline benchmark gate with determinism repeats passed:
  `outputs/verify_benchmark_gate/eval_reports/eval_summary_20260508_115841.json`.
- The committed smoke artifact set now includes summary, manifest,
  predictions, and graded JSONL rows under `docs/public-benchmark-artifacts/`.
- Real GAIA validation/test access still requires Hugging Face login and any
  dataset access/terms confirmation. No Hugging Face token was read, printed,
  or recorded.

## 2026-05-08 - MCP no-secret check matrix

- Status: filesystem prerequisite check passes; GitHub prerequisite check is
  blocked by missing token and fails safely.

Commands:

```bash
python examples/mcp_demos/mcp_filesystem_demo.py --root . --check
python examples/mcp_demos/mcp_github_demo.py --token-env GITHUB_PERSONAL_ACCESS_TOKEN --check
```

Observed results:

- Filesystem check returned `npx=true` and `mcp_python_sdk=true`.
- GitHub check returned non-zero with `GITHUB_PERSONAL_ACCESS_TOKEN=false`.
- The GitHub check printed only boolean prerequisite status; no token value was
  read from `.env`, printed, or recorded.

Follow-up:

- Do not run GitHub `--list-tools` until
  `GITHUB_PERSONAL_ACCESS_TOKEN` is configured in the environment.

## 2026-05-08 - Docker smoke command correction

- Status: local Docker runtime remains blocked because `docker` is not
  installed or not on `PATH`.
- Documentation now uses a dummy-key no-network doctor smoke command:

```bash
docker run --rm -e DEEPSEEK_API_KEY=dummy sdyj:0.6 sdyj doctor --provider deepseek
```

Follow-up on a Docker-enabled host:

```bash
docker build -t sdyj:0.6 .
docker build --build-arg SDYJ_EXTRAS=all -t sdyj:0.6-all .
docker run --rm sdyj:0.6 sdyj --help
docker run --rm -e DEEPSEEK_API_KEY=dummy sdyj:0.6 sdyj doctor --provider deepseek
docker compose config
docker compose run --rm sdyj sdyj --help
```

## 2026-05-08 - Final local release-readiness verification

- Status: local verification pass; external platform/runtime blockers remain
  documented.
- Branch: `feat/v0.6-self-verifying`.
- Recent local commits in this batch:
  - `571bdc4 ci(release): harden hosted publish readiness`
  - `49a8712 feat(benchmark): audit public smoke artifacts`

Final validation:

```bash
python -m pytest
python -m ruff check SDYJ_Agents tests examples
python -m build
python -m twine check dist/*
git status --short --branch
git log --oneline -10
```

Observed results:

- `python -m pytest`: `134 passed, 1 xfailed`.
- `python -m ruff check SDYJ_Agents tests examples`: all checks passed.
- `python -m build`: built `sdyj_multi_agents-0.6.0a1.tar.gz` and
  `sdyj_multi_agents-0.6.0a1-py3-none-any.whl`.
- `python -m twine check dist/*`: both artifacts passed.
- `git status --short --branch`: clean, branch ahead of origin by 2 commits
  before this verification-note commit.

Remaining blockers:

- Full DeepSeek + Tavily + arXiv live research still requires a usable
  `TAVILY_API_KEY`.
- Docker runtime smoke still requires Docker installed/on `PATH`.
- Hugging Face Spaces, GitHub Pages, GHCR, TestPyPI, and PyPI still require
  platform-side setup or manual workflow execution before public URLs/packages
  can be claimed.

## 2026-05-08 - Consolidated release-check preflight

- Status: pass for all required local gates; external blockers remain honest.
- Command:

```bash
sdyj release-check --provider deepseek --output-dir outputs\release_readiness
```

Observed results:

- `doctor`: pass; DeepSeek key present, Tavily missing, MCP source not configured.
- `pytest`: `140 passed, 1 xfailed`.
- `ruff`: all checks passed.
- Synthetic external benchmark smoke: pass.
  - Run id: `gaia_20260508_124747`.
  - Artifact root: `outputs/release_readiness/external_benchmarks/gaia_20260508_124747/`.
- Offline benchmark determinism gate: pass.
  - Summary: `outputs/release_readiness/eval_reports/eval_summary_20260508_124748.json`.
- MCP filesystem `--check`: pass.
- `python -m build`: pass.
- `python -m twine check dist/*`: pass.

Also verified the script wrapper:

```bash
python scripts\release_readiness.py --dry-run --json
```

Final post-documentation validation:

```bash
python -m pytest
# 140 passed, 1 xfailed

python -m ruff check SDYJ_Agents tests examples
# All checks passed

python -m build
# built sdyj_multi_agents-0.6.0a1.tar.gz and sdyj_multi_agents-0.6.0a1-py3-none-any.whl

python -m twine check dist/*
# PASSED
```

External blockers reported by the new preflight:

- `TAVILY_API_KEY=missing`; full DeepSeek + Tavily + arXiv live smoke remains blocked.
- `docker=missing`; Docker runtime build/run/compose smoke still needs a Docker-enabled host.
- `GITHUB_PERSONAL_ACCESS_TOKEN=missing`; GitHub MCP `--list-tools` remains blocked.

No API key or raw secret value was printed or recorded.

## 2026-05-08 - Benchmark failure-analysis artifacts

- Status: local benchmark auditability improved; no public GAIA score is claimed.
- Scope:
  - Internal `sdyj benchmark run` summaries now include `failure_analysis`
    rollups for failed thresholds, with root-cause buckets such as
    `planner_gap`, `citation_gap`, `tool_error`, `trace_gap`, and
    `verifier_gap`.
  - External `sdyj benchmark external` runs now write
    `failure_analysis.json` and `failure_analysis.md` beside `summary.json`,
    `manifest.jsonl`, `predictions.jsonl`, and `graded.jsonl`.

Validation:

```bash
python -m pytest tests/test_external_benchmark.py tests/test_evaluation.py tests/test_cli.py
# 24 passed

python -m ruff check SDYJ_Agents\evaluation SDYJ_Agents\benchmarks\external_runner.py SDYJ_Agents\cli\main.py tests\test_evaluation.py tests\test_external_benchmark.py tests\test_cli.py
# All checks passed

sdyj benchmark run --max-scenarios 1 --max-iterations 2 --fail-under 0.75 --determinism-repeats 2 --output-dir outputs\verify_failure_analysis_gate
# average score 1.0000, passed; summary: outputs\verify_failure_analysis_gate\eval_reports\eval_summary_20260508_132634.json

sdyj benchmark external --suite gaia --source local --limit 3 --output-dir outputs\public_benchmarks_failure_analysis --fail-under 1.0
# run id: gaia_20260508_132633, accuracy 1.0000, passed

python -m pytest
# 142 passed, 1 xfailed

python -m ruff check SDYJ_Agents tests examples
# All checks passed

python -m build
# built sdyj_multi_agents-0.6.0a1.tar.gz and sdyj_multi_agents-0.6.0a1-py3-none-any.whl

python -m twine check dist/*
# PASSED

sdyj release-check --provider deepseek --output-dir outputs\release_readiness_failure_analysis
# required gates passed; Tavily, Docker runtime, and GitHub MCP token remain BLOCKED
```

Committed public smoke artifacts now include:

- `docs/public-benchmark-artifacts/gaia-smoke-failure-analysis.json`
- `docs/public-benchmark-artifacts/gaia-smoke-failure-analysis.md`

External blockers remain unchanged: real GAIA Level 1 still requires Hugging
Face login / dataset access and real predictions; Tavily live research still
requires `TAVILY_API_KEY`; Docker runtime smoke still requires a Docker-enabled
host.

## 2026-05-11 - Benchmark regression-analysis artifacts

- Status: local benchmark comparison auditability improved; external blockers
  remain unchanged.
- Environment preflight:
  - Branch: `feat/v0.6-self-verifying`.
  - Latest commit before this batch: `4facae3 feat(benchmark): add failure analysis artifacts`.
  - `sdyj doctor --provider deepseek`: DeepSeek key present, Tavily missing,
    MCP source not configured.
  - Local `.env` check: `TAVILY_API_KEY=missing`; no secret value was printed or
    recorded.
  - `python -m pytest`: `142 passed, 1 xfailed`.
  - `python -m ruff check SDYJ_Agents tests examples`: all checks passed.

Scope:

- `sdyj benchmark run --compare-summary` and `sdyj benchmark compare` now write
  `comparison.regression_analysis` with:
  - compared / new / missing scenario counts;
  - feature/context changes such as verifier, reflection, plan refinement, and
    parallel-tool flags;
  - per-metric mean deltas, regression counts, improvement counts, and mapped
    root causes;
  - top metric regressions and improvements.
- Missing baseline scenarios now fail the comparison gate instead of being
  silently ignored.
- `sdyj benchmark compare --json` now writes raw stdout JSON so long Windows
  paths remain machine-parseable.

Validation:

```bash
python -m pytest tests/test_evaluation.py tests/test_cli.py
# 19 passed

python -m ruff check SDYJ_Agents\evaluation\runner.py SDYJ_Agents\cli\main.py tests\test_evaluation.py tests\test_cli.py
# All checks passed

sdyj benchmark run --scenario agent_reliability_hard --max-iterations 2 --output-dir outputs\compare_analysis_smoke_baseline
sdyj benchmark run --scenario agent_reliability_hard --max-iterations 2 --output-dir outputs\compare_analysis_smoke_candidate --compare-summary <baseline-summary>
sdyj benchmark compare <baseline-summary> <candidate-summary> --json
# JSON parsed with python -m json.tool; comparison_passed=True, metric_regressions=0, missing_scenarios=0

python -m pytest
# 145 passed, 1 xfailed

python -m ruff check SDYJ_Agents tests examples
# All checks passed

python -m build
# built sdyj_multi_agents-0.6.0a1.tar.gz and sdyj_multi_agents-0.6.0a1-py3-none-any.whl

python -m twine check dist/*
# PASSED
```

External blockers remain unchanged:

- `TAVILY_API_KEY` is still missing, so full DeepSeek + Tavily + arXiv live
  research was not run.
- Real GAIA Level 1 still requires Hugging Face login / dataset access and real
  predictions.
- Docker runtime smoke still requires a Docker-enabled host.
- HF Spaces / GitHub Pages / GHCR / TestPyPI / PyPI still require platform-side
  setup or manual workflow execution.
