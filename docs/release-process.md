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
  --notes-file docs/release-notes/v0.6.0.md
```

## Local smoke test before tagging

```bash
conda env update -n sdyj -f environment.yml --prune
conda activate sdyj
python -m build
python -m pip install --force-reinstall dist/sdyj_multi_agents-*.whl
sdyj research "smoke test query" --auto-approve --provider deepseek
```

Local development and CI use Conda. `pyproject.toml` remains the source of
package metadata for PyPI builds and the `sdyj` console entry point.

## Rollback

PyPI does not allow re-uploading the same version. If a bad release escapes:

1. Yank the bad version on PyPI (does not delete, but stops new installs).
2. Bump the version (e.g. `0.6.0` → `0.6.1`) and cut a fresh release.
