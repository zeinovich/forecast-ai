FROM python:3.10.10-slim
WORKDIR /app
COPY .backend/clustering_service/requirements.txt /app
RUN pip install -r requirements.txt

COPY .backend/clustering_service/ .
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
