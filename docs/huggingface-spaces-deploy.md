# Deploying SDYJ to Hugging Face Spaces

The Web UI in `SDYJ_Agents/web/app.py` runs on Hugging Face Spaces with a
single one-time setup. The deployed Space gives users a live, no-install demo
they can try in 60 seconds — the highest-leverage way to convert "saw the repo"
into "starred and shared the repo".

This guide assumes you already have a Hugging Face account.

## 1. Create the Space

1. Go to https://huggingface.co/new-space.
2. **Owner**: your account (or org).
3. **Space name**: `sdyj-multi-agents` (or any name you like).
4. **License**: MIT.
5. **Select the Space SDK**: **Streamlit**.
6. **Hardware**: CPU basic (free) is enough for the MVP.
7. **Visibility**: Public.

Click **Create Space**. Hugging Face creates an empty Git repo.

## 2. Configure the Space metadata

Spaces reads a YAML frontmatter from the repo's `README.md`. Use
[`docs/huggingface-space/README.md`](huggingface-space/README.md) as the
Space README template. You can paste it in the web UI editor, "Files" tab.
In the Hugging Face Space repository, this template must be saved as the root
`README.md`; leaving it under `docs/` will not activate the Space metadata.

The template starts with:

```yaml
---
title: SDYJ Multi Agents
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
license: mit
short_description: Self-verifying, replayable multi-agent research framework.
---

# SDYJ Multi Agents — live demo

This Space hosts the Streamlit Web UI for SDYJ Multi Agents.

Source code: https://github.com/hwfengcs/SDYJ_Multi_Agents
Release notes: https://github.com/hwfengcs/SDYJ_Multi_Agents/blob/main/docs/release-notes/v0.6.md
```

Save. The frontmatter is what tells HF to use Streamlit + `app.py`
as the entrypoint.

## 3. Push the code

The simplest pattern is to mirror this repo into the Space repo:

```bash
# In a fresh directory
git clone https://huggingface.co/spaces/<your-username>/sdyj-multi-agents space
cd space

# Pull code from this repo (preserve histories — or do a fresh copy if you
# don't care about history matching).
git remote add upstream https://github.com/hwfengcs/SDYJ_Multi_Agents.git
git fetch upstream main
git merge upstream/main --allow-unrelated-histories

# Make sure the README.md frontmatter from step 2 is preserved (the merge may
# overwrite it; keep your Space's README.md and only port over the body).

git push origin main
```

Hugging Face will start a build. Watch the **Logs** tab in the Space UI; the
first build takes 2–3 minutes because it installs all dependencies.

## 4. Set the Space secrets

The app refuses to run when no LLM API key is set. In the Space UI:

1. Open **Settings → Variables and secrets**.
2. Add the following secrets (use **New secret**, not "variable", so the
   values are not exposed in build logs):

   - `DEEPSEEK_API_KEY` — recommended, cheapest provider.
   - `TAVILY_API_KEY` — required for web search.
   - Optional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` — only
     if you want to expose those providers in the dropdown.

3. Restart the Space (Settings → Factory rebuild) so the new secrets are
   picked up by the running container.

## 5. Update the README badge

Once the Space URL is live (e.g. `https://huggingface.co/spaces/<you>/sdyj-multi-agents`),
edit `README.md` and `README_EN.md` to add a badge near the top:

```markdown
[![Open in Spaces](https://img.shields.io/badge/🤗_Spaces-Try_demo-blue)](https://huggingface.co/spaces/<you>/sdyj-multi-agents)
```

That badge is the single highest-conversion element for new visitors — keep
it above the fold.

The root README files already contain a hidden placeholder badge comment. Keep
it hidden until the real Space URL has been opened, a demo query has completed,
and the report / evidence / trace / LLM cost tabs have rendered. If the Space
build is still failing or secrets are missing, record the blocker in
`docs/live-run-notes.md` instead of exposing a dead badge.

## Pre-deploy checklist

Run these before pushing to the Space repo:

```bash
python -m pytest tests/test_hf_entrypoint.py tests/test_web_smoke.py tests/test_deploy_readiness.py
python -m ruff check tests/test_hf_entrypoint.py tests/test_web_smoke.py tests/test_deploy_readiness.py
sdyj doctor --provider deepseek
```

Expected local state:

- root `app.py` imports `SDYJ_Agents.web.app.main`;
- `requirements.txt` includes Streamlit for the HF build environment;
- Space README frontmatter is copied to the Space repo root as `README.md`;
- `DEEPSEEK_API_KEY` and `TAVILY_API_KEY` are configured as HF secrets, not
  variables;
- the README badge remains hidden until the public URL is verified.

## Cost notes

- The app prints per-call token + USD cost in the **LLM cost** tab. With
  DeepSeek + a 3-iteration research run, expect roughly **$0.02–0.05** per
  query. This is the upper bound that will land on your provider account
  while the Space is public.
- HF's free CPU Space tier rate-limits queries through the inference quota
  but does not bill you for users; LLM usage cost is paid against the keys
  you set in step 4.

## Local dry run before pushing

```bash
conda env update -n sdyj -f environment.yml --prune
conda activate sdyj
streamlit run streamlit_app.py
# Open http://localhost:8501 — same UI that runs on Spaces.
```

Run the no-network deployment preflight too:

```bash
sdyj doctor --provider deepseek
```

Local development uses the Conda environment in `environment.yml`. Hugging Face
Spaces still reads `requirements.txt` during hosted builds, so that lightweight
runtime file is intentionally kept for deployment compatibility.

If the local dry run works and the Space build logs come back green, the
demo is live.
