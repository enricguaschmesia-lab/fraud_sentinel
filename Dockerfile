FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md NEXT_STEPS.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY examples ./examples
COPY data ./data
COPY artifacts ./artifacts

CMD ["python", "-m", "fraud_sentinel.cli", "train"]

