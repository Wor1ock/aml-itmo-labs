# syntax=docker/dockerfile:1

# ---- dependency layer -------------------------------------------------------
FROM python:3.12-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY --from=deps /app/src /app/src
COPY configs ./configs
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 app \
    && mkdir -p data cache results report \
    && chown -R app:app /app
USER app

ENTRYPOINT ["python", "-m", "fusion_slu.cli"]
CMD ["report", "--config", "configs/default.yaml"]

# ---- test / development target ---------------------------------------------
FROM deps AS dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app
RUN uv sync --frozen
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN useradd --create-home --uid 1000 app \
    && mkdir -p data cache results report \
    && chown -R app:app /app
USER app

ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q"]
