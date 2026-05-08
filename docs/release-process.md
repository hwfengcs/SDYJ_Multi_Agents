# Release Process

## One-time PyPI setup (Trusted Publishers, no API tokens)

This project uses PyPI's [Trusted Publishers](https://docs.pypi.org/trusted-publishers/) flow,
which authenticates the GitHub Actions workflow with short-lived OIDC tokens
instead of long-lived API tokens. You need to do this once per index.

### TestPyPI

1. Sign in at https://test.pypi.org/.
2. Go to **Account settings → Publishing → Add a new pending publisher**.
3. Fill in:
   - PyPI Project Name: `sdyj-multi-agents`
   - Owner: `hwfengcs`
   - Repository name: `SDYJ_Multi_Agents`
   - Workflow name: `publish.yml`
   - Environment name: `testpypi`
4. Save. Once the project actually exists on TestPyPI, the publisher becomes
   active automatically.

### PyPI

Repeat the same steps at https://pypi.org/, but use environment name `pypi`.

### Configure GitHub environments

In the repo, go to **Settings → Environments**, create two environments named
`testpypi` and `pypi`. Optional but recommended: require manual approval before
the `pypi` environment runs, so a release cannot be pushed to PyPI by accident.

### Workflow safety gates

`.github/workflows/publish.yml` performs local release gates before any upload:

- validates the requested target is only `testpypi` or `pypi`;
- allows TestPyPI dry publishing from a branch for alpha smoke checks;
- skips PyPI publishing for GitHub Releases marked as prerelease;
- requires PyPI publishing to run from the exact tag that matches
  `pyproject.toml`, for example `v0.6.0a1` when the package version is
  `0.6.0a1`;
- builds the sdist and wheel, runs `twine check`, and installs the built wheel
  in a clean virtual environment before publishing.

These gates do not replace Trusted Publisher setup or GitHub environment
approval; they catch local release mistakes earlier in the workflow.

## Cutting a release

### Pre-release (alpha / beta) → TestPyPI

```bash
# 1. Bump version in pyproject.toml, e.g. 0.6.0a1 -> 0.6.0a2
# 2. Commit and push the bump
git commit -am "chore: bump version to 0.6.0a2"
git push

# 3. Manually run the publish workflow targeting TestPyPI
gh workflow run publish.yml -f target=testpypi
```

After it succeeds, verify on TestPyPI:

```bash
conda create -n sdyj-release-test python=3.12 pip
conda activate sdyj-release-test
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  sdyj-multi-agents
sdyj --help
```

### Stable release → PyPI

Do not use a GitHub Release marked **This is a pre-release** for PyPI. The
workflow intentionally skips PyPI publishing for prerelease events; alpha and
beta validation should go through the manual TestPyPI target first.

```bash
# 1. Bump version to a stable number (e.g. 0.6.0)
# 2. Commit and push
git commit -am "chore: release 0.6.0"
git push origin main

# 3. Tag and create a GitHub Release (this triggers the workflow automatically)
git tag v0.6.0
git push origin v0.6.0
gh release create v0.6.0 \
  --title "v0.6.0 — Self-Verifying Deep Research Agent" \
  --notes-file docs/release-notes/v0.6.md
```

## Local smoke test before tagging

```bash
conda env update -n sdyj -f environment.yml --prune
conda activate sdyj
pytest
ruff check SDYJ_Agents tests examples
sdyj benchmark run --max-scenarios 1 --max-iterations 2 --fail-under 0.75
python -m build
python -m twine check dist/*
python -m pip install --force-reinstall dist/sdyj_multi_agents-*.whl
sdyj --version
sdyj doctor --provider deepseek
```

The publish workflow runs the same release gate plus an additional clean
wheel-install smoke before publishing:

```bash
python -m build
python -m twine check dist/*
python -m venv .wheel-smoke
source .wheel-smoke/bin/activate
python -m pip install "dist/<built-wheel>.whl[all]"
sdyj --version
sdyj --help
DEEPSEEK_API_KEY=dummy sdyj doctor --provider deepseek
```

The dummy key is only used to prove the installed console script and no-network
doctor command work from the wheel; do not use real provider secrets in release
smoke logs.

Run a live research smoke only after provider/search keys are present:

```bash
sdyj research "smoke test query" --auto-approve --provider deepseek --max-iterations 1
```

Local development and CI use Conda. `pyproject.toml` remains the source of
package metadata for PyPI builds and the `sdyj` console entry point.

## Rollback

PyPI does not allow re-uploading the same version. If a bad release escapes:

1. Yank the bad version on PyPI (does not delete, but stops new installs).
2. Bump the version (e.g. `0.6.0` → `0.6.1`) and cut a fresh release.
