# Conda Setup

SDYJ uses Conda as the default local development environment.

## Create the Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate sdyj
```

If the environment already exists, update it in place:

```bash
conda env update -n sdyj -f environment.yml --prune
conda activate sdyj
```

The environment installs the local package in editable mode, so the `sdyj`
console command is available after activation.

The environment file resolves packages from conda-forge only. If an older
Anaconda/Miniconda installation stops before solving because its global
`defaults` channels require terms acceptance, either accept/remove those global
channels or use a Miniforge installation for this project.

## Configure Secrets

Copy the example environment file and fill in at least one LLM provider key:

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Recommended first provider:

```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Run

```bash
sdyj research "How should RAG agents be evaluated for reliability?"
sdyj benchmark run --max-scenarios 1 --max-iterations 2
streamlit run streamlit_app.py
```

## Verify

```bash
pytest
ruff check SDYJ_Agents tests
```

## Dependency Policy

- `environment.yml` is the default developer environment.
- `pyproject.toml` remains the package metadata for the `sdyj` console script,
  extras, and PyPI builds.
- `requirements.txt` is kept as a lightweight runtime list for hosted platforms
  that still expect a pip requirements file.
- When adding a dependency, update `pyproject.toml` first, then mirror it in
  `environment.yml` when it should be present in the Conda development
  environment.
