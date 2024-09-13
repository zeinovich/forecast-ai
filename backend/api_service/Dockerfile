FROM python:3.10.10-slim
WORKDIR /app
COPY ./backend/api_service/requirements.txt /app
RUN pip install -r requirements.txt

COPY ./backend/api_service/ .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "89.169.165.189", "--port", "8000"]
