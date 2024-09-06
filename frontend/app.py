"""
Streamlit app

App has 3 pages: Input, [TODO] Prediction, [TODO] AutoML

To start app run: `streamlit run ./frontend/app.py`

Author: zeinovich
"""

from datetime import timedelta
import requests

import streamlit as st
import pandas as pd

from .plots import sku_plot, add_events, forecast_plot

BACKEND_URL = "http://localhost:8000/forecast"
MODELS_URL = "http://localhost:8000/models"
TIMEOUT = 60  # HTTP timeout in seconds


# Function to validate that the horizon is greater than the granularity
def validate_horizon_vs_granularity(horizon, granularity):
    """Checks if horizon >= granularity"""
    horizons = {"1-day": 1, "1-week": 7, "1-month": 30}
    granularities = {"1-day": 1, "1-week": 7, "1-month": 30}

    return horizons[horizon] >= granularities[granularity]


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
    # Page selection in the sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page", ["Input Page", "Forecast", "Other Pages (Coming Soon)"]
    )

    # Logic for the Input Page
    if page == "Input Page":
        st.title("Demand Forecasting - Input Page")

        # File upload section in the sidebar
        st.sidebar.header("Upload CSV Files")

        # [TODO] - make adaptive form for file download
        dates_file = st.sidebar.file_uploader("Upload dates CSV", type="csv")
        sales_file = st.sidebar.file_uploader("Upload Sales CSV", type="csv")
        prices_file = st.sidebar.file_uploader("Upload Prices CSV", type="csv")

        if "dates_file" not in st.session_state:
            st.session_state["dates_file"] = dates_file
            st.session_state["sales_file"] = sales_file
            st.session_state["prices_file"] = prices_file

        # Load the uploaded data
        if (
            st.session_state["dates_file"]
            and st.session_state["sales_file"]
            and st.session_state["prices_file"]
        ):
            st.session_state["dates_file"].seek(0)
            st.session_state["sales_file"].seek(0)
            st.session_state["prices_file"].seek(0)

            dates = pd.read_csv(st.session_state["dates_file"])
            sales = pd.read_csv(st.session_state["sales_file"])
            prices = pd.read_csv(st.session_state["prices_file"])

            sales_df = merge(dates, sales, prices)

            # Assuming SKU and Store columns exist in sales and prices
            # [TODO] - zeinovich - make adaptive form for target cols selection
            sku_list = sales_df["SKU"].unique()
            store_list = sorted(sales_df["store_id"].unique())

            # SKU and Store selection
            sku = st.sidebar.selectbox("Select SKU", sku_list)
            store = st.sidebar.selectbox("Select Store", store_list)

            st.session_state["sku"] = sku
            st.session_state["store"] = store

            # Filter the data based on selected SKU and Store
            filtered_sales = sales_df[
                (sales_df["SKU"] == sku) & (sales_df["store_id"] == store)
            ]

            if len(filtered_sales) == 0:
                st.warning(f"SKU {sku} has never been sold in store {store}")
                st.stop()

            # Plotting the sales data
            st.subheader(f"Sales for SKU {sku} at Store {store}")

            st.markdown("### Select Time Window for Plots")

            time_window = st.radio(
                "Choose time window",
                ["1-week", "1-month", "3-month", "1-year", "All"],
            )

            # Filter data by the selected time window
            filtered_sales = filter_by_time_window(filtered_sales, "date", time_window)
            filtered_dates = filter_by_time_window(dates, "date", time_window)

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
            st.plotly_chart(sales_plot)

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

        else:
            st.sidebar.warning(
                "Please upload all three CSV files (dates, Sales, Prices)."
            )

    elif page == "Forecast":
        st.title("Forecast Page")

        # Display the selected forecast settings
        st.subheader("Forecast Settings")

        # [TODO] - zeinovich - place value if no store selection
        if not ("sku" in st.session_state and "store" in st.session_state):
            st.warning("You didn't select SKU or/and Store")
            st.stop()

        sku = st.session_state["sku"]
        store = st.session_state["store"]

        # Forecast horizon and granularity
        st.header(f"Forecast Settings for SKU {sku} in store {store}")
        horizon = st.selectbox(
            "Select Forecast Horizon", ["1-day", "1-week", "1-month"]
        )
        granularity = st.selectbox("Select Granularity", ["1-day", "1-week", "1-month"])

        # Check that the horizon is greater than granularity
        if not validate_horizon_vs_granularity(horizon, granularity):
            st.error("Forecast horizon must be greater than granularity.")
            st.stop()

        # Model selection (this is a simple list of boosting algorithms for now)
        # [TODO] - zeinovich - postprocessing of response
        models_list = requests.post(MODELS_URL, timeout=TIMEOUT)
        models_list = ["XGBoost"]
        model = st.selectbox("Select Model", models_list)

        # Button to trigger the forecast request
        if st.button("Get Forecast"):
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
            response = requests.post(BACKEND_URL, json=payload, timeout=TIMEOUT)

            # Process the response
            if response.status_code == 200:
                # [TODO] - zeinovich - postprocessing of response
                forecast_data = pd.DataFrame(response.json())
                st.success("Forecast generated successfully!")

                # Display the forecast data
                st.write(forecast_data)

            else:
                st.error(
                    "Failed to get forecast. Please check your settings and try again."
                )
                st.stop()

            # [TODO] - zeinovich - check if historical is present
            # and preprocess it
            fig = forecast_plot(forecast_data, historical=None)

            st.plotly_chart(fig)

    # Placeholder for future pages
    elif page == "Other Pages (Coming Soon)":
        st.title("Coming Soon")
        st.write("More features and pages will be added here in the future.")


if __name__ == "__main__":
    main()
