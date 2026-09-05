FROM python:3.13-slim

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

RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip vim && \
    rm -rf /var/lib/apt/lists/* && \
    if [ -f /app/topologies/zoo.zip ]; then cd /app/topologies && unzip -o zoo.zip; fi && \
    cd /app && python protos/__init__.py

# Default to a lightweight command; override at runtime as needed.
CMD ["python", "-m", "benchmarks.edge_based_centralized", "--help"]
