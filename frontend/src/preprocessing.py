from datetime import timedelta
from typing import Dict, Any
import pickle
import base64

import pandas as pd


# [TODO] - zeinovich - move to preprocessing pipeline
# [TODO] - zeinovich AutoML - how to pass DF to backend
def default_merge(
    dates: pd.DataFrame, sales: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges input dataframes. Works only for standard train format.
    See README.md for dataframe structure (section "Data")

    Args:
        dates (pd.DataFrame): Dates dataframe
        sales (pd.DataFrame): Sales dataframe
        prices (pd.DataFrame): Prices dataframe

    Returns:
        pd.DataFrame: Merged dataframe
    """
    dates["date"] = pd.to_datetime(dates["date"])

    merged_sales_dates = pd.merge(sales, dates, on="date_id", how="left")
    final_merged_data = pd.merge(
        merged_sales_dates, prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )

    return final_merged_data


# [TODO] - zeinovich - adapt for different datasets
def default_prepare_datasets(dates_file, sales_file, prices_file):
    dates = pd.read_csv(dates_file)
    sales = pd.read_csv(sales_file)
    prices = pd.read_csv(prices_file)

    sales_df = default_merge(dates, sales, prices)

    return sales_df, dates


# Function to validate that the horizon is greater than the granularity
def validate_horizon_vs_granularity(horizon, granularity):
    """Checks if horizon >= granularity"""
    horizons = {"1-day": 1, "1-week": 7, "1-month": 30}
    granularities = {"1-day": 1, "1-week": 7, "1-month": 30}
    h_int = horizons[horizon]
    g_int = granularities[granularity]

    return h_int >= g_int, h_int, g_int


def filter_by_time_window(
    df: pd.DataFrame, date_column: str, window: str
) -> pd.DataFrame:
    """
    Filters dataframe by specified column and time window.
    Goes back from tail of dataframe

    Args:
        df (pd.DataFrame): DataFrame
        date_column (str): Date column name
        window (str): Time window. Choice: ["1-week", "1-month", "3-month", "1-year", "All"]

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    latest_date = df[date_column].max()

    if window == "1-week":
        start_date = latest_date - timedelta(weeks=1)
    elif window == "1-month":
        start_date = latest_date - timedelta(weeks=4)
    elif window == "3-month":
        start_date = latest_date - timedelta(weeks=12)
    elif window == "1-year":
        start_date = latest_date - timedelta(weeks=52)
    else:
        return df

    return df[df[date_column] >= start_date]


def encode_dataframe(df: pd.DataFrame) -> Dict[Any, Any]:
    pickled = pickle.dumps(df)
    pickled_b64 = base64.b64encode(pickled)
    hug_pickled_str = pickled_b64.decode("utf-8")
    return hug_pickled_str


def decode_dataframe(data) -> pd.DataFrame:
    return pickle.loads(base64.b64decode(data.encode()))
