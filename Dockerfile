# REBUILD_TIMESTAMP: 2026-04-07 23:30 (Phase2 Fix: /health=healthy, /metadata, /schema, /mcp, httpx)
FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to container
COPY . .

# Install the package and its dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir uvicorn fastapi

# Set environment variables for better logging and robustness
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/.cache
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the FastAPI server
# Running from /app ensures server.app:app resolves correctly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
