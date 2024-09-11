from fastapi import FastAPI
import pandas as pd
import base64

# from io import BytesIO
import pickle
from pipeline import preprocess_data, predict_with_model

app = FastAPI()


def encode_dataframe(df: pd.DataFrame):
    pickled = pickle.dumps(df)
    pickled_b64 = base64.b64encode(pickled)
    hug_pickled_str = pickled_b64.decode("utf-8")
    return hug_pickled_str


@app.post("/predict/")
async def predict(payload: dict):
    """
    Интерфейс предсказания.
    Получает данные, предобрабатывает их и вызывает модель для предсказания и доверительных интервалов.
    """
    target_name = payload["target_name"]
    date_name = payload["date_name"]
    segment_name = payload["segment_name"]
    data = payload["data"]
    target_segment_names = payload["target_segment_names"]
    horizon = payload["horizon"]
    granularity = payload["granularity"]
    model_name = payload["model"]
    metric = payload["metric"]
    top_k_features = payload["top_k_features"]

    print(target_segment_names)

    # decoded_data = base64.b64decode(data)
    # df = pd.read_csv(BytesIO(decoded_data))

    df = pickle.loads(base64.b64decode(data.encode()))

    df = preprocess_data(df, target_name, date_name, segment_name, granularity)
    prediction_df = predict_with_model(
        df,
        target_segment_names,
        horizon // granularity,
        model_name,
        metric,
        top_k_features=top_k_features,
    )

    prediction_df = prediction_df[prediction_df["segment"].isin(target_segment_names)]

    # buffer_pred = BytesIO()
    # prediction_df.to_csv(buffer_pred, index=False)
    # encoded_predictions = base64.b64encode(buffer_pred.getvalue()).decode("utf-8")

    # buffer_metrics = BytesIO()
    # encoded_metrics = base64.b64encode(buffer_metrics.getvalue()).decode("utf-8")
    encoded_predictions = encode_dataframe(prediction_df)
    # encoded_metrics = encode_dataframe(metrics_df)

    return {
        "encoded_predictions": encoded_predictions,
        "encoded_metrics": 0,
    }
