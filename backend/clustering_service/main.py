from fastapi import FastAPI
import pandas as pd
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from utils import auto_determine_clusters

app = FastAPI()

@app.post("/clasterize/")
async def clusterize(payload: dict):
    """
    Эндпоинт кластеризации данных.
    Определяет оптимальное количество кластеров и проводит кластеризацию.
    """
    data = payload["data"]

    decoded_data = base64.b64decode(data)
    df = pd.read_csv(BytesIO(decoded_data))

    n_clusters = auto_determine_clusters(df)

    model = KMeans(n_clusters=n_clusters)
    df['cluster'] = model.fit_predict(df)

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    encoded_result = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {"encoded_dataframe": encoded_result, "n_clusters": n_clusters}
