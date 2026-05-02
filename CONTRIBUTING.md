# Contributing

Thanks for improving SDYJ Multi Agents. The project welcomes small, focused
pull requests that improve reliability, evaluation, documentation, or tool
support.

## Development Setup

```bash
python -m pip install -e ".[dev]"
copy .env.example .env
```

Real API keys are only needed for live smoke tests. Unit tests use fake LLM and
fake search implementations.

## Checks

Run these before opening a PR:

```bash
pytest
ruff check SDYJ_Agents tests
```

## Pull Request Guidelines

- Keep changes focused.
- Add or update tests for behavior changes.
- Do not commit `.env`, generated files in `outputs/`, or real API keys.
- Update README or docs when a user-facing command changes.
- Prefer small extension points over large rewrites.

## Good First Issues

- Add a new canned scenario test.
- Improve report citation formatting.
- Add source deduplication.
- Add a new retrieval adapter.
- Improve `docs/evaluation.md` with concrete metrics.
