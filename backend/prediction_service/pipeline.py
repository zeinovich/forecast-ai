import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame, target: str, date: str, granularity: str) -> pd.DataFrame:
    """
    Предобработка данных.
    Агрегируем данные по неделям или месяцам и заполняем пропуски.
    """
    df[date] = pd.to_datetime(df[date])

    if granularity == "week":
        df['week_no'] = df[date].dt.isocalendar().week
        df = df.groupby(['year', 'month', 'week_no'])[target].sum().reset_index() # проверить поля
    elif granularity == "month":
        df['month'] = df[date].dt.month
        df = df.groupby(['year', 'month'])[target].sum().reset_index()

    df = generate_features(df, target, date)

    return df

def generate_features(df: pd.DataFrame, target: str, date: str) -> pd.DataFrame:
    """
    Функция для генерации признаков.
    """
    pass
    return df

def predict_with_model(df: pd.DataFrame, horizon: int, model: str, metric: str):
    """
    Интерфейс предсказания через выбранную модель.
    Модели подключаются отдельно.
    
    :param df: данные для предсказания
    :param horizon: горизонт предсказания
    :param model: модель, которая будет использована
    :param metric: метрика для оценки предсказания
    :return: предсказанные значения
    """
    match model:
        case "example_model":
            return example_model_predict(df, horizon)
        case _:
            raise NotImplementedError(f"Model {model} is not implemented yet.")

def example_model_predict(df: pd.DataFrame, horizon: int):
    """
    Пример функции для предсказаний.
    Возвращаем случайные данные для предсказаний и доверительных интервалов.
    """
    predictions = np.random.rand(horizon) * 100
    std_dev = np.std(predictions) * 0.1
    upper_bound = predictions + std_dev
    lower_bound = predictions - std_dev
    return predictions, upper_bound, lower_bound
