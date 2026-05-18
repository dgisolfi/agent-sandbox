FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS agent
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent/ .
RUN mkdir -p /logs
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "app.py"]
CMD ["list"]

FROM base AS sandbox
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    g++ \
    build-essential \
    libncurses5-dev \
    libncursesw5-dev \
    && rm -rf /var/lib/apt/lists/*
COPY sandbox/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY sandbox/*.py .
RUN mkdir -p /workspace
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
