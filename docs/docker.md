# Docker

SDYJ can run as a local container for quick demos, hosted VM deployments, or
repeatable smoke tests. The Docker image installs the package and starts the
same root `app.py` Streamlit entrypoint used by Hugging Face Spaces.

## Build

```bash
docker build -t sdyj:0.6 .
```

The default image installs the `web` extra. To include every optional runtime
dependency, build with:

```bash
docker build --build-arg SDYJ_EXTRAS=all -t sdyj:0.6 .
```

## CLI Smoke

```bash
docker run --rm sdyj:0.6 sdyj --help
docker run --rm -e DEEPSEEK_API_KEY=dummy sdyj:0.6 sdyj doctor --provider deepseek
```

`sdyj doctor` is a no-network preflight. It reports whether provider/search
environment variables are configured without printing secret values. The dummy
DeepSeek key above is only for no-network CLI smoke; use real keys only through
your shell environment, `.env`, or a deployment secret manager.

## Docker-Enabled Host Checklist

On a host with Docker installed, verify both the default web image and the full
optional-runtime image:

```bash
docker build -t sdyj:0.6 .
docker build --build-arg SDYJ_EXTRAS=all -t sdyj:0.6-all .
docker run --rm sdyj:0.6 sdyj --help
docker run --rm -e DEEPSEEK_API_KEY=dummy sdyj:0.6 sdyj doctor --provider deepseek
docker compose config
docker compose run --rm sdyj sdyj --help
```

Record the result in `docs/live-run-notes.md`. If Docker is not installed or
not on `PATH`, record that as an external runtime blocker and keep using the
static Docker asset tests.

## Run the Web UI

Create a local `.env` from `.env.example`, then fill the provider keys you want
to use. At minimum, a live DeepSeek + Tavily run needs:

```text
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=...
TAVILY_API_KEY=...
```

Start the Streamlit app:

```bash
docker run --rm \
  --env-file .env \
  -p 8501:8501 \
  -v sdyj_outputs:/app/outputs \
  sdyj:0.6
```

Open http://localhost:8501. Run bundles are written to `/app/outputs` inside
the container and persisted in the `sdyj_outputs` Docker volume.

## Docker Compose

`docker-compose.yml` reads variables from the same local `.env` file that the
CLI uses, but the file itself is never copied into the image.

```bash
docker compose up --build
```

To run a one-off CLI command through Compose:

```bash
docker compose run --rm sdyj sdyj --help
```

## Secret Hygiene

`.dockerignore` excludes `.env`, local output bundles, build artifacts, caches,
and virtual environments from the Docker build context. Keep real API keys in
your shell environment, your local `.env`, or your deployment platform's secret
manager. Do not bake them into the image.

## Relationship to Hugging Face Spaces

The HF Spaces path still uses the native Streamlit SDK and `requirements.txt`.
Docker is an additional deployment path for users who prefer containers; it does
not replace the current Spaces workflow.
