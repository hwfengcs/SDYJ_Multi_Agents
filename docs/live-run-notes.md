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
