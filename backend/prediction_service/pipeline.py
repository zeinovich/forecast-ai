import importlib

from etna.datasets import TSDataset
from etna.metrics import SMAPE, MAE, MSE
from etna.pipeline import Pipeline
from etna.transforms import (
    DateFlagsTransform,
    LagTransform,
    RobustScalerTransform,
    FourierTransform,
    SegmentEncoderTransform,
    TreeFeatureSelectionTransform,
)

import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    granularity: str,
) -> TSDataset:
    """
    Предобработка данных.
    Агрегируем данные по неделям или месяцам и заполняем пропуски.
    """
    # df.drop(columns=["weekday", "wday", "month", "year"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = generate_features(df)

    ts_dataset = aggregate(df, granularity)

    return ts_dataset


def aggregate(df: pd.DataFrame, granularity: int) -> TSDataset:
    return TSDataset(df, freq="D")


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция для генерации признаков. [TODO]
    """
    return df


def import_model_class(model_name: str):
    """
    Импортирует модель по её названию. Если модель из нейросетевого модуля (nn), она импортируется из etna.models.nn.
    """
    if model_name.startswith("nn."):
        module_path = f"etna.models.nn.{model_name}"
    else:
        module_path = f"etna.models.{model_name}"

    module_name, class_name = module_path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    return model_class

def calculate_average_forecast(df: TSDataset, horizon: int, average_segments: list[str] = None):
    """
    Рассчитывает средний прогноз по выбранным или всем сегментам.

    :param df: TSDataset с данными
    :param horizon: горизонт прогнозирования
    :param average_segments: список сегментов для усреднения (если None, используются все сегменты)
    :return: DataFrame с средним прогнозом
    """
    all_data = df.to_pandas(flatten=True)
    
    if average_segments:
        mean_data = all_data[all_data['segment'].isin(average_segments)]
    else:
        mean_data = all_data
    
    mean_target = mean_data['target'].mean()
    
    last_date = all_data['timestamp'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    
    avg_forecast = pd.DataFrame({
        'timestamp': future_dates,
        'segment': ['sku_without_history'] * horizon,
        'target': [mean_target] * horizon,
        'target_0.025': [mean_target * 0.9] * horizon,  # Примерный доверительный интервал
        'target_0.975': [mean_target * 1.1] * horizon
    })
    
    return avg_forecast
    

def predict_with_model(
    df: TSDataset,
    target_segment_names: list[str],
    horizon: int,
    model_name: str,
    metric: bool,
    top_k_features: int,
    average_segments: list[str] = None
):
    """
    Интерфейс предсказания через выбранную модель.

    :param df: данные для предсказания
    :param target_segment_names: сегменты для которых необходимо предсказать
    :param horizon: горизонт предсказания
    :param model_name: название модели, которая будет использована
    :param metric: необходимо ли возвращать значения метрик
    :param top_k_features: количество лучших признаков для отбора
    :param average_segments: список сегментов для усреднения прогноза товаров без истории
    :return: предсказанные значения и метрики
    """
    tfs_transform = TreeFeatureSelectionTransform(
        model="random_forest",
        top_k=top_k_features,
    )
    date_flags_transform = DateFlagsTransform(
        is_weekend=True, day_number_in_month=True, day_number_in_week=True
    )
    lag_transform = LagTransform(
        in_column="target", lags=list(range(horizon, 2 * horizon + 1, 1))
    )
    fourier_transform = FourierTransform(period=365.25, order=3)
    scaler_transform = RobustScalerTransform(in_column="target")
    segment_encoder = SegmentEncoderTransform()

    transforms = [
        date_flags_transform,
        lag_transform,
        fourier_transform,
        scaler_transform,
        segment_encoder,
        tfs_transform,
    ]

    if not model_name:
        raise ValueError("Should provide model_name")
    
    model_class = import_model_class(model_name)
    model = model_class()
    pipeline = Pipeline(model=model, transforms=transforms, horizon=horizon)

    pipeline.fit(df)

    forecast_ts = pipeline.forecast(prediction_interval=True)
    forecast_df = forecast_ts.df.loc[
        :, pd.IndexSlice[:, ["target", "target_0.025", "target_0.975"]]
    ]

    forecast_df = pd.melt(forecast_df, ignore_index=False).reset_index()
    forecast_df = forecast_df.pivot(
        index=["segment", "timestamp"], columns="feature", values="value"
    ).reset_index()

    forecast_df = forecast_df[
        ["timestamp", "segment", "target", "target_0.025", "target_0.975"]
    ]

    # Добавляем прогноз для товаров без истории
    avg_forecast = calculate_average_forecast(df, horizon, average_segments)
    forecast_df = pd.concat([forecast_df, avg_forecast], ignore_index=True)

    if metric:
        metrics_df, _, _ = pipeline.backtest(
            ts=df,
            metrics=[MAE(), MSE(), SMAPE()],
            n_folds=3,
            aggregate_metrics=True,
        )
        metrics_df.loc['sku_without_history'] = np.nan
    else:
        metrics_df = pd.DataFrame(
            data=["no metric passed"], columns=["metrics"], index=["segment"]
        )
        metrics_df.loc['sku_without_history'] = "no metric passed"

    return forecast_df, metrics_df
