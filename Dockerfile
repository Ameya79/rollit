FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md demo.py ./
COPY rollit/ ./rollit/

RUN pip install --no-cache-dir .

CMD ["python", "demo.py"]
