FROM python:3.12.5-slim
WORKDIR /app
COPY ./pyproject.toml ./poetry.lock ./
RUN pip install poetry
RUN poetry install
COPY ./backend/clustering_service /app
EXPOSE 8002
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
