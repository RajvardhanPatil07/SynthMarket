FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY synthmarket /app/synthmarket

RUN python -m pip install --upgrade pip && \
    python -m pip install .[api]

RUN useradd --create-home --shell /usr/sbin/nologin synthmarket && \
    chown -R synthmarket:synthmarket /app
USER synthmarket

EXPOSE 8000

CMD ["uvicorn", "synthmarket.api:app", "--host", "0.0.0.0", "--port", "8000"]
