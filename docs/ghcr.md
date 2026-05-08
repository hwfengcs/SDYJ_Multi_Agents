# GHCR Container Publishing

The repository includes a manual GitHub Actions workflow for building the
Docker image and optionally pushing it to GitHub Container Registry. It is a
release-preparation asset; it does not publish an image unless a maintainer
explicitly runs it with `publish=true`.

## Workflow

```text
.github/workflows/ghcr.yml
```

Image name:

```text
ghcr.io/hwfengcs/sdyj-multi-agents
```

The workflow always builds the image with Buildx. It pushes only when the
manual `publish` input is set to `true`; the default is `false`, so a dry run
can validate Dockerfile changes without creating a public package.

## Local Checks

Run static checks on machines without Docker:

```bash
python -m pytest tests/test_docker_assets.py tests/test_deploy_readiness.py
python -m ruff check tests/test_docker_assets.py tests/test_deploy_readiness.py
```

Run runtime checks on a Docker-enabled host:

```bash
docker build -t sdyj:0.6 .
docker run --rm sdyj:0.6 sdyj --help
docker run --rm sdyj:0.6 sdyj doctor --provider deepseek
docker compose run --rm sdyj sdyj --help
```

`sdyj doctor` is a no-network preflight. It checks whether provider/search
environment variables are present without printing their values.

## Publish Checklist

1. Confirm `python -m pytest`, `python -m ruff check SDYJ_Agents tests examples`,
   `python -m build`, and `python -m twine check dist/*` are green.
2. Confirm Docker runtime smoke passes on a Docker-enabled host.
3. Run **Container Image** from GitHub Actions with `publish=false`.
4. If the dry run passes and the release owner approves, rerun with
   `publish=true`.
5. Record the resulting package URL in `docs/live-run-notes.md`.

Do not add a public GHCR badge to the README until an image has actually been
pushed and pulled successfully.
