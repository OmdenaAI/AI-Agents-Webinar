# One image, four services. Which tool server a container runs is chosen
# by its command, so all four restart independently from the same build.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first, so editing source does not reinstall the world.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    MCP_HOST=0.0.0.0
