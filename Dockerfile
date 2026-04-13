FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY quantflow/ quantflow/
COPY config/ config/

RUN pip install --no-cache-dir -e ".[ml,llm,dashboard,live]"

EXPOSE 8501

CMD ["quantflow", "--help"]
