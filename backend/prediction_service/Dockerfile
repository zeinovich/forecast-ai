FROM python:3.10.10-slim
WORKDIR /app
COPY ./pyproject.toml ./poetry.lock ./
RUN pip install poetry
RUN poetry install
COPY ./backend/prediction_service /app
EXPOSE 8001
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
