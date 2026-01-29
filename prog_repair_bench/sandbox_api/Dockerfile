FROM python:3.12-slim

# Avoid bytecode & keep logging output visible
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
ENV POETRY_VERSION=2.1.3
RUN pip install poetry==${POETRY_VERSION}

ENV POETRY_PYTHON_PATH=/usr/local/bin/python3

# Configure poetry
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_VIRTUALENVS_IN_PROJECT=false

# Copy project files first for better caching
COPY pyproject_cloud.toml ./pyproject.toml
COPY README.md ./README.md
# Copy only the necessary files for the API
COPY prog_repair_bench/ prog_repair_bench/

# TODO Restore
RUN poetry config sandbox TOKEN

# Install dependencies
RUN poetry install --extras="local"

EXPOSE 8000

# This command would be replaced by the building script
CMD ["python", "prog_repair_bench/sandbox_api/run_api.py"]