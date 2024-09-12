from typing import Dict
import requests

from fastapi import FastAPI

app = FastAPI()


PREDICTION_SERVICE_HOST = "forecast"
CLUSTERING_SERVICE_HOST = "cluster"
PREDICTION_SERVICE_PORT = 8001
CLUSTERING_SERVICE_PORT = 8002

TIMEOUT = 300


@app.post("/forecast/")
async def get_forecast(payload: Dict[str, str]) -> Dict[str, str]:
    """
    Основная точка входа для управления задачами.
    Перенаправляет запрос на Prediction-сервис.
    """
    prediction_url = (
        f"http://{PREDICTION_SERVICE_HOST}:{PREDICTION_SERVICE_PORT}/predict/"
    )
    response = requests.post(
        prediction_url,
        json=payload,
        timeout=TIMEOUT,
    )

    return response.json()


@app.post("/clusterize/")
async def get_clusters_dataset(payload: Dict[str, str]) -> Dict[str, str]:
    """
    Эндпоинт для кластеризации данных.
    Перенаправляет запрос на Clustering-сервис.
    """
    clustering_url = (
        f"http://{CLUSTERING_SERVICE_HOST}:{CLUSTERING_SERVICE_PORT}/clusterize/"
    )
    response = requests.post(
        clustering_url,
        json=payload,
        timeout=TIMEOUT,
    )

    return response.json()
