# MDAPFlow-MCP Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy uv installation script and install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install dependencies and build package
RUN uv sync --frozen

# Create non-root user
RUN useradd --create-home --shell /bin/bash mdapflow
USER mdapflow

# Expose port for HTTP transport (optional)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uv", "run", "mdapflow-mcp"]