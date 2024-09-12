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


def predict_with_model(
    df: TSDataset,
    horizon: int,
    model_name: str,
    top_k_features: int,
):
    """
    Интерфейс предсказания через выбранную модель.
    Модели подключаются отдельно.

    :param TSDataset df: данные для предсказания
    :param target_segment_names: сегменты для которых неодходимо предсказать
    :param horizon: горизонт предсказания
    :param model: модель, которая будет использована
    :param metric: необходимо ли возвращать значения метрик
    :return: предсказанные значения
    """

    tfs_transform = TreeFeatureSelectionTransform(
        model="random_forest",
        top_k=top_k_features,
    )
    date_flags_transform = DateFlagsTransform(
        is_weekend=True,
        day_number_in_month=True,
        day_number_in_week=True,
        week_number_in_month=True,
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

    if model_name == "" or not model_name:
        raise ValueError("Should provide model_name")
    else:
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

    metrics_df, _, _ = pipeline.backtest(
        ts=df,
        metrics=[MAE(), MSE(), SMAPE()],
        n_folds=3,
        aggregate_metrics=True,
    )

    return forecast_df, metrics_df
