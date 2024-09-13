FROM python:3.10.10-slim
WORKDIR /app
COPY ./backend/prediction_service/requirements.txt /app
RUN pip install -r requirements.txt

COPY ./backend/prediction_service/ .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "89.169.165.189", "--port", "8001"]
