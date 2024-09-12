from fastapi import FastAPI
import hdbscan
from tslearn.metrics import cdist_dtw

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
    df = decode_dataframe(data)

    # Подготовка данных (временные ряды) для кластеризации
    time_series = (
        df.to_numpy()
    )  # предполагается, что данные представляют временные ряды

    # Рассчитываем матрицу расстояний DTW
    distance_matrix = cdist_dtw(time_series)

    # Применяем HDBSCAN для кластеризации
    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2)
    labels = clusterer.fit_predict(distance_matrix)

    # Добавляем метки кластеров к DataFrame
    df["cluster"] = labels

    # Кодируем результат обратно в base64
    encoded_result = encode_dataframe(df)

    return {
        "encoded_dataframe": encoded_result,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
    }
