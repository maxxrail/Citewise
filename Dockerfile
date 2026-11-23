# syntax=docker/dockerfile:1

# ---- build: resolve deps with uv into /app/.venv ----
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- runtime: slim image, non-root user ----
FROM python:3.12-slim AS runtime
RUN groupadd --system citewise && useradd --system --gid citewise --home /app citewise
WORKDIR /app
COPY --from=build --chown=citewise:citewise /app /app
COPY --chown=citewise:citewise alembic.ini ./
COPY --chown=citewise:citewise migrations ./migrations
COPY --chown=citewise:citewise data/brands ./data/brands
# Mock fixtures ship in the image so `docker compose up` in mock mode serves recorded answers.
COPY --chown=citewise:citewise tests/fixtures ./tests/fixtures
# Explicit paths: don't rely on the package's install location to find repo data.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 CITEWISE_MODE=mock     FIXTURES_DIR=/app/tests/fixtures BRANDS_DIR=/app/data/brands
USER citewise
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "citewise.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
