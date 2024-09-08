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
    Возвращает закодированный датафрейм с колонками ["date", "pred", "upper", "lower"].
    """
    target_name = payload["target_name"]
    date_name = payload["date_name"]
    segment_name = payload["segment_name"]
    data = payload["data"]
    horizon = payload["horizon"]
    granularity = payload["granularity"]
    model = payload["model"]
    metric = payload["metric"]

    decoded_data = base64.b64decode(data)
    df = pd.read_csv(BytesIO(decoded_data))

    df = preprocess_data(df, target_name, date_name, segment_name, granularity)

    prediction_dates, predictions, upper_bound, lower_bound, metric_value = predict_with_model(df, horizon, model, metric)

    df[date_name] = pd.to_datetime(df['timestamp'])
    df[segment_name] = df['segment']
    df[target_name] = df['target']
    df.drop(columns=['timestamp', 'segment', 'target'], inplace=True)

    result_df = pd.DataFrame({
        "date": prediction_dates,
        "pred": predictions,
        "upper": upper_bound,
        "lower": lower_bound,
        "metric_score": metric_value
    })

    buffer = BytesIO()
    result_df.to_csv(buffer, index=False)
    encoded_result = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {"encoded_dataframe": encoded_result}
