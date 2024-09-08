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
    target = payload["target"]
    date = payload["date"]
    data = payload["data"]
    horizon = payload["horizon"]
    granularity = payload["granularity"]
    model = payload["model"]
    metric = payload["metric"]

    decoded_data = base64.b64decode(data)
    df = pd.read_csv(BytesIO(decoded_data))

    df = preprocess_data(df, target, date, granularity)

    predictions, upper_bound, lower_bound = predict_with_model(df, horizon, model, metric)

    future_dates = pd.date_range(df[date].max(), periods=horizon + 1, freq="D")[1:]
    result_df = pd.DataFrame({
        "date": future_dates,
        "pred": predictions,          # Предсказанные значения
        "upper": upper_bound,         # Верхняя граница доверительного интервала
        "lower": lower_bound          # Нижняя граница доверительного интервала
    })

    buffer = BytesIO()
    result_df.to_csv(buffer, index=False)
    encoded_result = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {"encoded_dataframe": encoded_result}
