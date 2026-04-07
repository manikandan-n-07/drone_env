# Use official lightweight Python image
# REBUILD_TIMESTAMP: 2026-04-07 16:30 (Force Refresh)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git for building packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to container
COPY . .

# Install the package and its dependencies
# pip install -e . uses the pyproject.toml in the current directory
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir uvicorn fastapi

# Expose the API port (standard for HF Spaces or custom)
EXPOSE 8000

# Set environment variables for better logging
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/.cache

# Command to run the FastAPI server
# This calls the main function in server/app.py which starts uvicorn
CMD ["python", "server/app.py", "--host", "0.0.0.0", "--port", "8000"]
