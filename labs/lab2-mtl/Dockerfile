# Static data preparation happens at build time; the code is copied last so that
# editing a source file does not invalidate the download and preprocessing layers.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

# git is needed for the commit hash in run_meta.json, curl for manual checks
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.10.7

WORKDIR /app

# dependency layer: changes only when pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# prepare reads only configs/ and src/, so it is cached independently of tests
COPY configs ./configs
COPY src ./src
RUN uv sync --frozen

# The official PTB-XL 1.0.3 archive is mirrored on the Hugging Face Hub because
# physionet.org is not reachable from every network. Use --set data.prefer_mirror=false
# to download from PhysioNet instead. The raw archive is dropped after preprocessing.
RUN python -m ecgmtl.cli prepare --set data.prefer_mirror=true --set data.root=/app/data \
 && rm -rf /app/data/raw

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/outputs /app/report \
 && chown -R app:app /app
USER app

CMD ["python", "-m", "ecgmtl.cli", "all", "--config", "configs/default.yaml"]
