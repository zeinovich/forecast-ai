FROM python:3.10.10-slim
WORKDIR /app
COPY ./frontend/requirements.txt /app
RUN pip install -r requirements.txt

COPY ./frontend/ .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
