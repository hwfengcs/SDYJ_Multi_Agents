# syntax=docker/dockerfile:1

FROM python:3.12-slim

ARG SDYJ_EXTRAS=web

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    OUTPUT_DIR=/app/outputs \
    OUTPUT_FORMAT=markdown

WORKDIR /app

RUN groupadd --system sdyj \
    && useradd --system --gid sdyj --create-home --home-dir /home/sdyj sdyj

COPY pyproject.toml README.md README_EN.md LICENSE MANIFEST.in ./
COPY SDYJ_Agents ./SDYJ_Agents

RUN python -m pip install --upgrade pip \
    && if [ -n "$SDYJ_EXTRAS" ]; then \
        python -m pip install ".[${SDYJ_EXTRAS}]"; \
    else \
        python -m pip install .; \
    fi

COPY app.py streamlit_app.py requirements.txt environment.yml mcp_config.json.example ./
COPY docs ./docs
COPY examples ./examples

RUN mkdir -p /app/outputs \
    && chown -R sdyj:sdyj /app /home/sdyj

USER sdyj

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
