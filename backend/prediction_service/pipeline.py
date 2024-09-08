from etna.datasets import TSDataset
from etna.metrics import SMAPE, MAE, MSE
from etna.models import (
    AutoARIMAModel,
    ProphetModel,
    SARIMAXModel,
    CatBoostMultiSegmentModel,
    LinearPerSegmentModel,
    ElasticMultiSegmentModel,
    HoltModel,
    SeasonalMovingAverageModel
)
from etna.pipeline import Pipeline
from etna.transforms import ModelOutliersTransform, DateFlagsTransform, LagTransform, MovingAverageTransform, FourierTransform
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame, target_name: str, date_name: str, segment_name: str, granularity: str) -> TSDataset:
    """
    Предобработка данных.
    Агрегируем данные по неделям или месяцам и заполняем пропуски.
    """
    df.drop(columns=['weekday', 'wday', 'month', 'year'], inplace=True)
    df[date_name] = pd.to_datetime(df[date_name])
    label_encoder = LabelEncoder()

    df['event_name_1_encoded'] = label_encoder.fit_transform(df['event_name_1'].fillna('0'))
    df['event_type_1_encoded'] = label_encoder.fit_transform(df['event_type_1'].fillna('0'))
    df['event_name_2_encoded'] = label_encoder.fit_transform(df['event_name_2'].fillna('0'))
    df['event_type_2_encoded'] = label_encoder.fit_transform(df['event_type_2'].fillna('0'))
    df.drop(columns=['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], inplace=True)

    df['timestamp'] = pd.to_datetime(df[date_name])
    df['segment'] = df[segment_name]
    df['target'] = df[target_name]
    
    df.drop(columns=[date_name, segment_name, target_name], inplace=True)

    df = generate_features(df)
    ts_dataset = TSDataset(df, freq='D')
    if granularity == 'weekly':
        ts_dataset = ts_dataset.to_period('W')
    elif granularity == 'monthly':
        ts_dataset = ts_dataset.to_period('M')
    df = remove_outliners(df)
    df = generate_features_etna(df)
    return df

def remove_outliners(df: TSDataset):
    prophet_model = ProphetModel()
    outliers_transform = ModelOutliersTransform(model=prophet_model, in_column='target', threshold=3)
    df.fit_transform([outliers_transform])
    return df

def generate_features_etna(df: TSDataset):
    """
    Генерация стандартных признаков для временных рядов с использованием Etna.
    
    Признаки включают:
    - Признаки на основе даты (год, месяц, день недели и т.д.)
    - Лаги целевой переменной
    - Скользящее среднее
    - Гармоники Фурье для моделирования сезонности
    
    Параметры:
    ts_dataset: TSDataset - исходные данные временных рядов.
    
    Возвращает:
    TSDataset - временной ряд с добавленными признаками.
    """
    date_flags_transform = DateFlagsTransform()

    lag_transform = LagTransform(in_column="target", lags=[7, 14, 30])
    ma_transform = MovingAverageTransform(in_column="target", window=7)
    fourier_transform = FourierTransform(period=365.25, order=3)
    df.fit_transform([
        date_flags_transform,
        lag_transform,
        ma_transform,
        fourier_transform
    ])
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция для генерации признаков.
    """
    
    transformed_df = df.copy()
    transformed_df['is_event_day'] = transformed_df[['event_type', 'event_type2']].notna().any(axis=1)
    transformed_df['is_double_event'] = transformed_df[['event_type', 'event_type2']].notna().all(axis=1)
    transformed_df['is_start_of_month'] = transformed_df['date'].dt.day <= 5
    transformed_df['is_end_of_month'] = transformed_df['date'].dt.day >= (transformed_df['date'].dt.days_in_month - 4)
    # transformed_df['average_sales_per_weekday']
    # transformed_df['average_sales_per_month']
    # transformed_df['has_cashback']
    # transformed_df['store_avg_sales']
    # transformed_df['store_item_popularity']
    # transformed_df['store_weekday_impact']
    # transformed_df['store_cashback_activity']
    # transformed_df['store_price_level']
    # transformed_df['store_seasonal_sales_variation']
    # transformed_df['is_discounted']
    # что-то про скидки добавить TODO
    # ...
    return transformed_df

def predict_with_model(df: TSDataset, horizon: int, model: str, metric: str):
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
        case "auto_arima":
            model_instance = AutoARIMAModel()
        case "prophet":
            model_instance = ProphetModel()
        case "sarimax":
            model_instance = SARIMAXModel()
        case "catboost":
            model_instance = CatBoostMultiSegmentModel()
        case "linear":
            model_instance = LinearPerSegmentModel()
        case "elastic_net":
            model_instance = ElasticMultiSegmentModel()
        case "holt":
            model_instance = HoltModel()
        case "seasonal_moving_average":
            model_instance = SeasonalMovingAverageModel()
        case _:
            raise NotImplementedError(f"Model {model} is not implemented yet.")
        
    pipeline = Pipeline(model=model_instance, horizon=horizon)
    pipeline.fit(df)
    forecast_ts = pipeline.forecast(prediction_interval=True)
    forecast_df = forecast_ts.df
    predictions = forecast_df['target'].values
    upper_bound = forecast_df['target_upper'].values if 'target_upper' in forecast_df.columns else None
    lower_bound = forecast_df['target_lower'].values if 'target_lower' in forecast_df.columns else None

    prediction_dates = forecast_df.index.get_level_values('timestamp').values
    if metric:
        metric_instance = get_metric_instance(metric)
        metric_value = metric_instance(y_true=df[:, :, 'target'], y_pred=forecast_df[:, :, 'target'])

    return prediction_dates, predictions, upper_bound, lower_bound, metric_value

def get_metric_instance(metric: str):
    """
    Возвращает экземпляр метрики на основе её названия.
    
    :param metric: название метрики
    :return: объект метрики
    """
    match metric:
        case "smape":
            return SMAPE()
        case "mae":
            return MAE()
        case "mse":
            return MSE()
        case _:
            raise NotImplementedError(f"Metric {metric} is not implemented yet.")
