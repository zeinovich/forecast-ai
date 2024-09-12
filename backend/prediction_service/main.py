import base64
from typing import Dict
import pickle
import logging

from fastapi import FastAPI
import pandas as pd

from pipeline import preprocess_data, predict_with_model, calculate_average_forecast

app = FastAPI()


def encode_dataframe(df: pd.DataFrame):
    pickled = pickle.dumps(df)
    pickled_b64 = base64.b64encode(pickled)
    hug_pickled_str = pickled_b64.decode("utf-8")
    return hug_pickled_str


@app.post("/predict/")
async def predict(payload: dict) -> dict:
    """
    Интерфейс предсказания.
    Получает данные, предобрабатывает их и вызывает модель для предсказания и доверительных интервалов.
    """
    # target_name = payload["target_name"]
    # date_name = payload["date_name"]
    # segment_name = payload["segment_name"]
    data = payload["data"]
    target_segment_names = payload["target_segment_names"]
    horizon = payload["horizon"]
    granularity = payload["granularity"]
    model_name = payload["model"]
    top_k_features = payload["top_k_features"]
    no_history = payload["no_history"]

    data = pickle.loads(base64.b64decode(data.encode()))

    df = data[data["segment"].isin(target_segment_names)]

    df = preprocess_data(df, granularity)
    prediction_df, metrics_df = predict_with_model(
        df,
        # target_segment_names,
        horizon // granularity,
        model_name,
        top_k_features=top_k_features,
    )

    if no_history is not None:
        preds = calculate_average_forecast(
            data,
            horizon // granularity,
            granularity,
            no_history,
        )
        prediction_df = pd.concat(
            [prediction_df, preds],
            axis=0,
        )

    encoded_predictions = encode_dataframe(prediction_df)
    encoded_metrics = encode_dataframe(metrics_df)

    return {
        "encoded_predictions": encoded_predictions,
        "encoded_metrics": encoded_metrics,
    }
