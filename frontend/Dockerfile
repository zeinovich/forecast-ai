FROM python:3.10.10-slim
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
