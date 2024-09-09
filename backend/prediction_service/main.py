from fastapi import FastAPI
import pandas as pd
import base64
from io import BytesIO
from pipeline import preprocess_data, predict_with_model

app = FastAPI()


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

    decoded_data = base64.b64decode(data)
    df = pd.read_csv(BytesIO(decoded_data))

    df = preprocess_data(df, target_name, date_name, segment_name, granularity)

    prediction_df, metrics_df = predict_with_model(
        df,
        target_segment_names,
        horizon,
        model_name,
        metric,
        top_k_features=top_k_features,
    )

    buffer_pred = BytesIO()
    prediction_df.to_csv(buffer_pred, index=False)
    encoded_predictions = base64.b64encode(buffer_pred.getvalue()).decode("utf-8")

    buffer_metrics = BytesIO()
    metrics_df.to_csv(buffer_metrics, index=False)
    encoded_metrics = base64.b64encode(buffer_metrics.getvalue()).decode("utf-8")

    return {
        "encoded_predictions": encoded_predictions,
        "encoded_metrics": encoded_metrics,
    }
