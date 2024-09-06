"""
Streamlit app

App has 3 pages: Input, [TODO] Prediction, [TODO] AutoML

To start app run: `streamlit run ./frontend/app.py`

Author: zeinovich
"""

from datetime import timedelta
import requests  # dev

import streamlit as st
import pandas as pd
import numpy as np  # dev

from plots import sku_plot, add_events, forecast_plot

BACKEND_URL = "http://localhost:8000/forecast"
MODELS_URL = "http://localhost:8000/models"
TIMEOUT = 60  # HTTP timeout in seconds


# Function to validate that the horizon is greater than the granularity
def validate_horizon_vs_granularity(horizon, granularity):
    """Checks if horizon >= granularity"""
    horizons = {"1-day": 1, "1-week": 7, "1-month": 30}
    granularities = {"1-day": 1, "1-week": 7, "1-month": 30}
    h_int = horizons[horizon]
    g_int = granularities[granularity]

    return h_int >= g_int, h_int, g_int


# [TODO] - zeinovich - make adaptive form for file download
def upload_data():
    dates_file = st.sidebar.file_uploader("Upload dates CSV", type="csv")
    sales_file = st.sidebar.file_uploader("Upload Sales CSV", type="csv")
    prices_file = st.sidebar.file_uploader("Upload Prices CSV", type="csv")

    css = """
        <style>
            [data-testid='stFileUploader'] {
                width: max-content;
            }
            [data-testid='stFileUploader'] section {
                padding: 0;
                float: left;
            }
            [data-testid='stFileUploader'] section > input + div {
                display: none;
            }
            [data-testid='stFileUploader'] section + div {
                float: right;
                padding-top: 0;
            }

        </style>
    """

    st.sidebar.markdown(css, unsafe_allow_html=True)

    return dates_file, sales_file, prices_file


# [TODO] - zeinovich - move to preprocessing pipeline
# [TODO] - zeinovich AutoML - how to pass DF to backend
def merge(
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

    sales["SKU"] = sales["item_id"].apply(lambda x: x[-3:])
    dates["date"] = pd.to_datetime(dates["date"])

    sales_df = pd.merge(
        left=sales,
        right=dates[["date_id", "wm_yr_wk", "date"]],
        how="left",
        left_on="date_id",
        right_on="date_id",
        suffixes=("", ""),
    )

    # merge on (wm_yr_wk, item_id) to get price for particular week
    sales_df = pd.merge(
        left=sales_df,
        right=prices[["item_id", "wm_yr_wk", "sell_price"]],
        how="left",
        left_on=("wm_yr_wk", "item_id"),
        right_on=("wm_yr_wk", "item_id"),
        suffixes=("", ""),
    )

    return sales_df


# [TODO] - zeinovich - adapt for different datasets
def prepare_datasets(dates_file, sales_file, prices_file):
    dates = pd.read_csv(dates_file)
    sales = pd.read_csv(sales_file)
    prices = pd.read_csv(prices_file)

    sales_df = merge(dates, sales, prices)

    return sales_df, dates


def get_forecast_settings(sku_list, store_list):
    # Assuming SKU and Store columns exist in sales and prices

    # SKU and Store selection
    sku = st.sidebar.selectbox("Select SKU", sku_list)
    store = st.sidebar.selectbox("Select Store", store_list)

    st.sidebar.subheader("Forecast Settings")
    horizon = st.sidebar.selectbox(
        "Select Forecast Horizon", ["1-day", "1-week", "1-month"]
    )
    # [TODO] - zeinovich - add aggregation of target
    granularity = st.sidebar.selectbox(
        "Select Granularity",
        # ["1-day", "1-week", "1-month"],
        ["1-day"],
    )
    # Check that the horizon is greater than granularity
    valid, h_int, g_int = validate_horizon_vs_granularity(horizon, granularity)
    if not valid:
        st.sidebar.error("Forecast horizon must be greater than granularity.")
        st.stop()

    # models_list = requests.post(MODELS_URL, timeout=TIMEOUT)
    models_list = ["XGBoost"]
    model = st.sidebar.selectbox("Select Model", models_list)

    return sku, store, h_int, g_int, model


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


def main():
    """Main"""
    st.title("Demand Forecasting")

    # File upload section in the sidebar
    st.sidebar.subheader("Upload CSV Files")

    dates_file, sales_file, prices_file = upload_data()

    # Load the uploaded data
    if dates_file and sales_file and prices_file:
        sales_df, dates = prepare_datasets(dates_file, sales_file, prices_file)
    else:
        st.sidebar.warning("Please upload all three CSV files (dates, Sales, Prices).")
        st.stop()

    # [TODO] - zeinovich - make adaptive form for target cols selection
    # [TODO] - zeinovich - place value if no store selection
    # make it look more like passing list of lists
    # name selectboxes as their columns
    sku_list = sales_df["SKU"].unique()
    store_list = sorted(sales_df["store_id"].unique())
    sku, store, horizon, granularity, model = get_forecast_settings(
        sku_list, store_list
    )

    # Filter the data based on selected SKU and Store
    filtered_sales = sales_df[
        (sales_df["SKU"] == sku) & (sales_df["store_id"] == store)
    ]

    if len(filtered_sales) == 0:
        st.warning(f"SKU {sku} has never been sold in store {store}")
        st.stop()

    # Plotting the sales data
    st.subheader(f"Sales for SKU {sku} at Store {store}")

    # time_window = st.radio(
    #     "Choose time window",
    #     ["1-week", "1-month", "3-month", "1-year", "All"],
    #     horizontal=True,
    # )

    # # Filter data by the selected time window
    cutoff = st.sidebar.selectbox(
        "Choose time window",
        ["1-week", "1-month", "3-month", "1-year", "All"],
        index=2,
    )
    filtered_sales = filter_by_time_window(filtered_sales, "date", cutoff)
    filtered_dates = filter_by_time_window(dates, "date", cutoff)

    event_dates = filtered_dates[filtered_dates["event_type_1"].notna()][
        ["date", "event_name_1", "event_type_1"]
    ]
    sales_plot = sku_plot(
        filtered_sales,
        x="date",
        y="cnt",
        title="Sales over Time",
        labels={"date": "Date", "cnt": "Items Sold"},
    )
    sales_plot = add_events(event_dates, sales_plot)
    sales_st = st.empty()
    sales_st.plotly_chart(sales_plot)

    # Plotting the price data
    price_plot = sku_plot(
        filtered_sales,
        x="date",
        y="sell_price",
        title="Prices Over Time",
        labels={"date": "Date", "sell_price": "Sell Price"},
    )
    price_plot = add_events(event_dates, price_plot)
    st.plotly_chart(price_plot)

    # Button to trigger the forecast request
    if st.sidebar.button("Get Forecast"):
        # Create payload with forecast settings
        payload = {
            "sku": sku,
            "store": store,
            "horizon": horizon,
            "granularity": granularity,
            "model": model,
        }

        # Send request to the backend (example backend port assumed to be 8000)
        # Update this with the correct backend URL
        # response = requests.post(BACKEND_URL, json=payload, timeout=TIMEOUT)

        response = np.random.normal(
            filtered_sales["cnt"].mean(),
            filtered_sales["cnt"].std(),
            size=horizon // granularity,
        )
        response = pd.DataFrame(response, columns=["predicted"])
        response["date"] = [
            filtered_dates["date"].tolist()[-1] + timedelta(days=1) * (i + 1)
            for i in range(horizon // granularity)
        ]
        response["upper"] = np.random.normal(
            1.2 * response["predicted"],
            0.5,
            size=len(response),
        )
        response["lower"] = np.random.normal(
            0.8 * response["predicted"],
            0.5,
            size=len(response),
        )
        response["lower"] = np.min(response[["lower", "upper"]].to_numpy(), axis=1)
        response["upper"] = np.max(response[["lower", "upper"]].to_numpy(), axis=1)
        st.session_state["response"] = response

    else:
        st.stop()

    # Process the response
    # if response and response.status_code == 200:
    if st.session_state["response"] is not None:
        # [TODO] - zeinovich - postprocessing of response
        # forecast_data = pd.DataFrame(response.json())
        forecast_data = response
        st.success("Forecast generated successfully!")

        # Display the forecast data
        st.write(forecast_data)

    else:
        st.error("Failed to get forecast. Please check your settings and try again.")
        st.stop()

    # Model selection (this is a simple list of boosting algorithms for now)
    # [TODO] - zeinovich - postprocessing of response

    # # [TODO] - zeinovich - check if historical is present
    # # and preprocess it
    sales_plot = forecast_plot(
        forecast_data,
        sales_plot,
        add_trace={"row": 1, "col": 1},
        scatter_args={"line": {"color": "Black", "dash": "dash"}},
    )
    sales_st.plotly_chart(sales_plot)


if __name__ == "__main__":
    main()
