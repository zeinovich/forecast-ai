import importlib
from etna.auto import Auto
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
    TreeFeatureSelectionTransform,
)
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(
    df: pd.DataFrame,
    data_future: pd.DataFrame = None,
    columns_types: dict = None,  # словарь с типами колонок, например {"event_type": "categorical", "is_holiday": "bool"}
    target_name: str = "target",
    date_name: str = "timestamp",
    segment_name: str = "segment",
    granularity: str = "D",
    is_template: bool = False,
):
    """
    Предобработка данных.
    Агрегируем данные по неделям или месяцам и заполняем пропуски.
    """

    # Обработка признаков
    df[date_name] = pd.to_datetime(df[date_name])
    label_encoder = LabelEncoder()
    for column, col_type in columns_types.items():
        if column in df.columns:
            if col_type == "categorical":
                df[column] = label_encoder.fit_transform(df[column].fillna("0"))
                if data_future is not None and not data_future.empty:
                    data_future[column] = label_encoder.transform(data_future[column].fillna("0"))
            elif col_type == "bool":
                df[column] = df[column].apply(lambda x: 1 if x is True else 0)
                if data_future is not None and not data_future.empty:
                    data_future[column] = data_future[column].apply(lambda x: 1 if x is True else 0)
    
    if data_future is None or data_future.empty:
        df = df[[date_name, segment_name, target_name]]
        columns_types = {target_name: columns_types.get(target_name, 'numeric')}
    else:
        empty_columns = [col for col in data_future.columns if data_future[col].isnull().all() and col != target_name]
        if empty_columns:
            columns_types = {col: col_type for col, col_type in columns_types.items() if col not in empty_columns}
            df.drop(columns=empty_columns, inplace=True)
            if data_future is not None and not data_future.empty:
                data_future.drop(columns=empty_columns, inplace=True)

    df["timestamp"] = df[date_name]
    df["segment"] = df[segment_name]
    df["target"] = df[target_name]
    df.drop(columns=[date_name, segment_name, target_name], inplace=True)

    if data_future is not None and not data_future.empty:
        data_future[date_name] = pd.to_datetime(data_future[date_name])
        data_future["timestamp"] = data_future[date_name]
        data_future["segment"] = data_future[segment_name]
        data_future["target"] = data_future[target_name]
        data_future.drop(columns=[date_name, segment_name, target_name], inplace=True)

    first_segment = df['segment'].unique()[0]
    df_first_segment = df[df['segment'] == first_segment]
    detected_granularity = pd.infer_freq(df_first_segment['timestamp'])

    if detected_granularity != granularity:
        agg_dict = {'target': "sum"}
        for column, col_type in columns_types.items():
            if col_type == "categorical":
                agg_dict[column] = lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
            elif col_type == "bool":
                agg_dict[column] = "mean"
            elif col_type == "numeric":
                agg_dict[column] = "mean"

        df = df.groupby([pd.Grouper(key="timestamp", freq=granularity), "segment"]).agg(agg_dict).reset_index()
        if data_future is not None and not data_future.empty:
            data_future = data_future.groupby([pd.Grouper(key="timestamp", freq=granularity), "segment"]).agg(agg_dict).reset_index()
    
    if is_template and data_future is not None and not data_future.empty:
        df, data_future = generate_features(df, data_future, empty_columns)

    ts_dataset = TSDataset(df, freq=granularity)
    ts_dataset_future = TSDataset(data_future, freq=granularity)
    
    ts_dataset.df = ts_dataset.df.fillna(0)
    return ts_dataset, ts_dataset_future


def generate_features(df: pd.DataFrame, df_future: pd.DataFrame, empty_columns: list) -> pd.DataFrame:
    """
    Функция для генерации признаков на основе событий и статистики.
    """

    def add_event_features(data):
        """Добавляет признаки события для переданного DataFrame."""
        if all(col not in empty_columns for col in ['event_type_1', 'event_type_2']):
            data['is_event_day'] = data[['event_type_1', 'event_type_2']].notna().any(axis=1).astype(int)
            data['is_double_event'] = data[['event_type_1', 'event_type_2']].notna().all(axis=1).astype(int)
        return data

    def add_popularity_features(data):
        """Добавляет признак популярности товара для переданного DataFrame."""
        if all(col not in empty_columns for col in ['store_id', 'segment', 'target']):
            data['store_item_popularity'] = data.groupby(['store_id', 'segment'])['target'].transform('mean')
        return data

    df = add_event_features(df)
    df_future = add_event_features(df_future)
    
    df = add_popularity_features(df)
    df_future = add_popularity_features(df_future)

    return df, df_future


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
    ts_origin: TSDataset,
    future_ts_origin: TSDataset,
    target_segment_names: list[str],
    horizon: int,
    model_name: str,
    metric: bool,
    top_k_percent_features: float,
):
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

    df = TSDataset(ts_origin.df.copy(), freq=ts_origin.freq)
    future_df = TSDataset(future_ts_origin.df.copy(), freq=future_ts_origin.freq)
    date_flags_transform = DateFlagsTransform(
        is_weekend=True, day_number_in_month=True, day_number_in_week=True
    )
    lag_transform = LagTransform(
        in_column="target", lags=list(range(horizon, 2 * horizon + 1, 1))
    )
    fourier_transform = FourierTransform(period=365.25, order=3)
    outliner_transform = DensityOutliersTransform(in_column="target")
    all_features = df.df.columns.get_level_values(1).unique()
    num_total_features = len(all_features)
    top_k_features = max(1, int(num_total_features * top_k_percent_features))
    tfs_transform = TreeFeatureSelectionTransform(
        model="catboost",
        top_k=top_k_features,
    )
    transforms = [
        date_flags_transform,
        lag_transform,
        fourier_transform,
        outliner_transform,
        tfs_transform,
    ]

    df.fit_transform(transforms)

    if model_name == "" or not model_name:
        auto = Auto(
            target_metric=SMAPE(), horizon=horizon, backtest_params=dict(n_folds=5)
        )
        pipeline = auto.fit(ts=df, tune_size=0)
    else:
        model_class = import_model_class(model_name)
        model = model_class()
        pipeline = Pipeline(model=model, horizon=horizon)

    pipeline.fit(df)

    df.df = df[:, target_segment_names, :]
    future = df.make_future(future_steps=horizon, transforms=transforms)
    future_df = future_df.df.reindex(future.df.index)
    future_df1 = future.df.copy()
    future_df1.update(future_df)
    future = TSDataset(df=future_df1, freq=future.freq)
    forecast_ts = model.forecast(ts=future, prediction_interval=True)
    # forecast_ts = pipeline.forecast(ts=future, prediction_interval=True)
    forecast_ts.inverse_transform(transforms)
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

    metrics_df = pd.DataFrame()
    if metric:
        # ts_origin.fit_transform(transforms)
        metrics = [MAE(), MSE(), MAPE(), SMAPE()]
        pipeline_metrics = Pipeline(model=model, horizon=horizon)
        metrics_df, _, _ = pipeline_metrics.backtest(ts=ts_origin, metrics=metrics, n_folds=3, aggregate_metrics=True)

    return forecast_df, metrics_df
