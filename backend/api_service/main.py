from fastapi import FastAPI
import requests
import os

app = FastAPI()

@app.post("/forecast/")
async def get_forecast(payload: dict):
    """
    Основная точка входа для управления задачами.
    Перенаправляет запрос на Prediction-сервис.
    """
    target_name = payload["target_name"]
    date_name = payload["date_name"]
    segment_name = payload["segment_name"]
    data = payload["data"]
    target_segment_names = payload["target_segment_names"]
    horizon = payload["horizon"]
    granularity = payload["granularity"]
    model = payload["model"]
    metric = payload["metric"]
    top_k_features = payload["top_k_features"]

    prediction_url = f"http://{os.getenv('PREDICTION_SERVICE_HOST')}:{os.getenv('PREDICTION_SERVICE_PORT')}/predict/"
    response = requests.post(prediction_url, json={
        "target_name": target_name,
        "date_name": date_name,
        "segment_name": segment_name,
        "data": data,
        "target_segment_names": target_segment_names,
        "horizon": horizon,
        "granularity": granularity,
        "model": model,
        "metric": metric,
        "top_k_features": top_k_features
    })

    return response.json()


@app.post("/clusterize/")
async def get_clusters_dataset(payload: dict):
    """
    Эндпоинт для кластеризации данных.
    Перенаправляет запрос на Clustering-сервис.
    """
    data = payload["data"]

    clustering_url = f"http://{os.getenv('CLUSTERING_SERVICE_HOST')}:{os.getenv('CLUSTERING_SERVICE_PORT')}/clasterize/"
    response = requests.post(clustering_url, json={
        "data": data,
    })

    return response.json()