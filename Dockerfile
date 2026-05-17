FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Keep CPU image aligned with the current requirements.txt only.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt

COPY . /app

# Default to a lightweight command; override at runtime as needed.
CMD ["python", "benchmarks/edge_based_centralized.py", "--help"]
