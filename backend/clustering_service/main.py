from fastapi import FastAPI
import hdbscan
from tslearn.metrics import cdist_dtw
import pandas as pd
from utils import decode_dataframe, encode_dataframe

app = FastAPI()


@app.post("/clusterize/")
async def clusterize(payload: dict):
    """
    Эндпоинт кластеризации данных.
    Использует расстояния DTW и кластеризацию HDBSCAN.
    Пример ноутбука: https://github.com/zeinovich/dream-team/blob/romanov/notebooks/v0.3-romanov-clustering_ts.ipynb
    """
    # Декодируем данные из base64 в CSV
    data = payload["data"]

    df = decode_dataframe(data)[["date", "item_id", "cnt"]]

    items = df["item_id"].unique()
    items_ts = {}

    for item in items:
        items_ts[item] = df[df["item_id"] == item]["cnt"].to_list()

    items_ts = pd.DataFrame.from_dict(items_ts, orient="index")

    # Подготовка данных (временные ряды) для кластеризации
    # предполагается, что данные представляют временные ряды

    # Рассчитываем матрицу расстояний DTW
    distance_matrix = cdist_dtw(items_ts)

    # Применяем HDBSCAN для кластеризации
    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2)
    labels = clusterer.fit_predict(distance_matrix)

    # Добавляем метки кластеров к DataFrame
    items_ts["cluster"] = labels
    items_ts.index.name = "item_id"
    items_ts = items_ts.reset_index()

    # Кодируем результат обратно в base64
    encoded_result = encode_dataframe(items_ts[["item_id", "cluster"]])

    return {
        "encoded_dataframe": encoded_result,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
    }
