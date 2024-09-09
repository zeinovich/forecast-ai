import importlib
from etna.datasets import TSDataset
from etna.metrics import SMAPE, MAE, MSE, MAPE
from etna.pipeline import Pipeline
from etna.transforms import (
    DensityOutliersTransform,
    DateFlagsTransform,
    LagTransform,
    RobustScalerTransform,
    FourierTransform,
    SegmentEncoderTransform,
    TreeFeatureSelectionTransform
)
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

    df['timestamp'] = df[date_name]
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

    ts_dataset.df = ts_dataset.df.applymap(lambda x: 1 if x is True else (0 if x is False else x))
    ts_dataset.df = ts_dataset.df.fillna(0)
    return df

def remove_outliners(df: TSDataset):
    outliers_transform = DensityOutliersTransform(in_column='target')
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
    fourier_transform = FourierTransform(period=365.25, order=3)
    scaler_transform = RobustScalerTransform(in_column="target")
    segment_encoder = SegmentEncoderTransform()
    df.fit_transform([
        date_flags_transform,
        lag_transform,
        fourier_transform,
        scaler_transform,
        segment_encoder
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

def import_model_class(model_name: str):
    """
    Импортирует модель по её названию. Если модель из нейросетевого модуля (nn), она импортируется из etna.models.nn.
    """
    if model_name.startswith('nn.'):
        module_path = f"etna.models.nn.{model_name}"
    else:
        module_path = f"etna.models.{model_name}"

    module_name, class_name = module_path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

def predict_with_model(df: TSDataset, target_segment_names: list[str], horizon: int, model_name: str, metric: bool, top_k_features: int):
    """
    Интерфейс предсказания через выбранную модель.
    Модели подключаются отдельно.
    
    :param df: данные для предсказания
    :param target_segment_names: сегменты для которых неодходимо предсказать
    :param horizon: горизонт предсказания
    :param model: модель, которая будет использована
    :param metric: необходимо ли возвращать значения метрик
    :return: предсказанные значения
    """
    model_class = import_model_class(model_name)
    model_instance = model_class()

    transform = TreeFeatureSelectionTransform(
        model='catboost',
        top_k=top_k_features,
    )
        
    pipeline = Pipeline(model=model_instance, transforms=[transform], horizon=horizon)
    pipeline.fit(df)
    
    target_segments = df[:, target_segment_names, :]
    target_ts = TSDataset(target_segments, freq=df.freq)
    forecast_ts = pipeline.forecast(ts=target_ts, prediction_interval=True)
    forecast_df = forecast_ts.df.loc[:, pd.IndexSlice[:, ['target', 'target_0.025', 'target_0.975']]]

    if metric:
        metrics_df, _, _ = pipeline.backtest(ts=df, metrics=[MAE(), MSE(), MAPE(), SMAPE()], n_folds=3, aggregate_metrics=True)

    return forecast_df, metrics_df