"""
Streamlit app

App has 3 pages: Input, [TODO] Prediction, [TODO] AutoML

To start app run: `streamlit run ./frontend/app.py`

Author: zeinovich
"""

from datetime import timedelta
from typing import Dict
import requests

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly import subplots

_pallette = [
    "#5ba300",
    "#89ce00",
    "#0073e6",
    "#e6308a",
    "#b51963",
]


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


def add_events(event_dates: pd.DataFrame, plot: go.Figure) -> go.Figure:
    """
    Used to highlight events on a plot

    Args:
        event_dates (pd.DataFrame): DataFrame with columns `[date, event_name_1]`
        plot (go.Figure): Plotly go.Figure

    Returns:
        go.Figure: Plotly go.Figure with highlighted events
    """
    # Gets max Y value directly from Figure
    # to not throw around dataframes
    top = max(trace["y"].max() for trace in plot.data if "y" in trace)

    cmap = {t: c for t, c in zip(event_dates["event_type_1"].unique(), _pallette)}

    for _, row in event_dates.iterrows():
        event_date = row["date"]
        event_label = row["event_name_1"]
        event_type = row["event_type_1"]

        # Add vertical rectangle for the event
        plot.add_vrect(
            x0=event_date - timedelta(days=0.5),
            x1=event_date + timedelta(days=0.5),
            fillcolor=cmap[event_type],
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        # Add annotation label for the event
        # Add annotation label for the event with vertical text and no arrow
        plot.add_annotation(
            x=event_date,
            y=top * 1.1,
            text=event_label,
            showarrow=False,
            textangle=-90,  # Rotate text to vertical
            valign="middle",  # Vertically center the text in the rectangle
            xshift=0,  # Shift text slightly for better alignment
            bgcolor=cmap[event_type],
            font=dict(color="black"),
            opacity=0.8,
        )

    return plot


def make_plot(
    df: pd.DataFrame, x: str, y: str, title: str, labels: Dict[str, str]
) -> go.Figure:
    """
    Plots line plot from data. Data distribution is appended to the left of a line plot

    Args:
        df (pd.DataFrame): Data
        x (str): X-axis column
        y (str): Y-axis column (also used for histogram)
        title (str): Figure title
        labels (Dict[str, str]): Column names, i.e. {"sell_price": "Sell Price"}

    Returns:
        go.Figure: Plotly plot
    """
    # Create 2 subplots with shared Y-axis
    # Left subplot - Line plot (80% width)
    # Right subplot - histogram of values in y column (20% width)
    plot = subplots.make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2]
    )

    plot.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="lines",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Add vertical histogram
    plot.add_trace(
        go.Histogram(
            y=df[y],
            histfunc="count",
            histnorm="percent",
            opacity=0.6,
            marker=dict(color="LightCoral"),
            orientation="h",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    min_y, max_y = df[y].min(), df[y].max()
    min_x, max_x = df[x].min(), df[x].max()

    # Add dashed horizontal lines for min and max Y with annotations
    plot.add_shape(
        type="line",
        x0=min_x,
        x1=max_x,
        y0=min_y,
        y1=min_y,
        line=dict(color="Green", width=2, dash="dash"),
        name=f"Min: {min_y}",
        showlegend=True,
    )

    plot.add_shape(
        type="line",
        x0=min_x,
        x1=max_x,
        y0=max_y,
        y1=max_y,
        line=dict(color="Red", width=2, dash="dash"),
        name=f"Max: {max_y}",
        showlegend=True,
    )

    plot.update_layout(title=title, xaxis_title=labels[x], yaxis_title=labels[y])
    # Set right sublot x-axis label directly
    plot["layout"]["xaxis2"]["title"] = "Fraction, %"
    plot.update_layout(showlegend=True)

    return plot


def main():
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
            sales_plot = make_plot(
                filtered_sales,
                x="date",
                y="cnt",
                title="Sales over Time",
                labels={"date": "Date", "cnt": "Items Sold"},
            )
            sales_plot = add_events(event_dates, sales_plot)
            st.plotly_chart(sales_plot)

            # Plotting the price data
            price_plot = make_plot(
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

        # Model selection (this is a simple list of boosting algorithms for now)
        model = st.selectbox(
            "Select Model", ["XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]
        )

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
            backend_url = "http://localhost:8000/forecast"  # Update this with the correct backend URL
            response = requests.post(backend_url, json=payload, timeout=60)

            # Process the response
            if response.status_code == 200:
                forecast_data = pd.DataFrame(response.json())
                st.success("Forecast generated successfully!")

                # Display the forecast data
                st.write(forecast_data)

            else:
                st.error(
                    "Failed to get forecast. Please check your settings and try again."
                )
                st.stop()

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=forecast_data["date"],
                    y=forecast_data["predicted"],
                    mode="lines",
                    name="Predicted Demand",
                    line=dict(color="blue"),
                )
            )

            # Prediction intervals (Uncertainty bounds) shaded area
            fig.add_trace(
                go.Scatter(
                    x=forecast_data["date"],
                    y=forecast_data["upper"],
                    mode="lines",
                    line=dict(width=0),
                    name="Upper Bound",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_data["date"],
                    y=forecast_data["lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    name="Uncertainty Bounds",
                    showlegend=True,
                )
            )

            fig.update_layout(
                title="Predicted Demand with Uncertainty Bounds",
                xaxis_title="Date",
                yaxis_title="Demand",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )

            fig.add_trace(
                go.Scatter(
                    x=filtered_sales["date"],
                    y=filtered_sales["cnt"],
                    labels={"date": "Date", "cnt": "Items Sold"},
                )
            )

            st.plotly_chart(fig)

    # Placeholder for future pages
    elif page == "Other Pages (Coming Soon)":
        st.title("Coming Soon")
        st.write("More features and pages will be added here in the future.")


if __name__ == "__main__":
    main()
