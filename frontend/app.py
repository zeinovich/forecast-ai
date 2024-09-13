"""
Streamlit app

App has 3 pages: Input, [TODO] Prediction, [TODO] AutoML

To start app run: `streamlit run ./frontend/app.py`

Author: zeinovich
"""

from typing import Dict, Any

import requests

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import pandas as pd

from src.plots import forecast_plot, add_events, add_minmax
from src.preprocessing import (
    default_prepare_datasets,
    validate_horizon_vs_granularity,
    filter_by_time_window,
    encode_dataframe,
    decode_dataframe,
)

_pallette = [
    "#5ba300",
    "#89ce00",
    "#0073e6",
    "#e6308a",
    "#b51963",
]

FORECAST_URL = "http://forecast_api:8000/forecast"
CLUSTER_URL = "http://forecast_api:8000/clusterize"
TIMEOUT = 300  # HTTP timeout in seconds

DATES = "./data/raw/shop_sales_dates.csv"

_models = [
    "prophet.ProphetModel",
    "NaiveModel",
    "CatBoostMultiSegmentModel",
    "ElasticMultiSegmentModel",
]

_metrics = [
    "MSE",
    "RMSE",
    "MAE",
    "MAPE",
]


# [TODO] - zeinovich - make adaptive form for file download
def upload_standard_data():
    dates_file = "./data/shop_sales_dates.csv"
    sales_file = "./data/shop_sales.csv"
    prices_file = "./data/shop_sales_prices.csv"

    return dates_file, sales_file, prices_file


def upload_data(expander: DeltaGenerator):
    sales_file = expander.file_uploader("Upload Sales CSV", type="csv")

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

    return sales_file


def aggregate(df: pd.DataFrame, granularity: int) -> pd.DataFrame:
    """
    Aggregates data by given granularity (1 = day, 7 = week, 30 = month) for each segment.

    Parameters:
    - df: DataFrame with 'timestamp', 'segment', and 'target' columns.
    - granularity: integer defining the granularity (1 - day, 7 - week, 30 - month).

    Returns:
    - Aggregated DataFrame.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if granularity == 1:
        freq = "D"  # day
    elif granularity == 7:
        freq = "W"  # week
    elif granularity == 30:
        freq = "M"  # month
    else:
        raise ValueError("Granularity must be 1 (day), 7 (week), or 30 (month).")

    df.set_index("timestamp", inplace=True)

    aggregated_df = df.groupby("segment").resample(freq)["target"].sum().reset_index()

    return aggregated_df


def reset_forecast():
    if "response" in st.session_state:
        del st.session_state["response"]


def get_dataset_features(df: pd.DataFrame, is_standard_format: bool):
    expander = st.sidebar.expander("DFU settings", expanded=True)

    if not is_standard_format:
        segment_name = expander.selectbox(
            "Select ID column", df.columns.tolist(), on_change=reset_forecast
        )
        target_name = expander.selectbox(
            "Select target column", df.columns.tolist(), on_change=reset_forecast
        )
        date_name = expander.selectbox(
            "Select date column", df.columns.tolist(), on_change=reset_forecast
        )

    else:
        segment_name = "item_id"
        target_name = "cnt"
        date_name = "date"

    unique_segments = df[segment_name].unique().tolist()

    segments = expander.multiselect(
        f"Select {segment_name}", sorted(unique_segments), on_change=reset_forecast
    )

    return target_name, date_name, segment_name, segments


def add_new_segments(df: pd.DataFrame) -> Dict[Any, Any]:
    add_new = st.sidebar.expander("New IDs")

    name_ = add_new.text_input(label="Name of new ID")

    unique_segments = ["all"] + sorted(df["segment"].unique().tolist())

    similar = add_new.multiselect("Select similar IDs", unique_segments, default=None)

    if name_ is None or similar is None:
        add_new.warning("Fill name")
        st.stop()

    if similar == ["all"]:
        similar = 0

    no_history = {"name": name_, "similar": similar}

    return no_history


def get_forecast_settings(forecast_expander: DeltaGenerator):
    # Assuming SKU and Store columns exist in sales and prices

    # SKU and Store selection
    # [TODO] - zeinovich - multiSKU
    horizon = forecast_expander.selectbox(
        "Select Forecast Horizon",
        ["1-day", "1-week", "1-month"],
        on_change=reset_forecast,
    )
    # [TODO] - zeinovich - add aggregation of target
    granularity = forecast_expander.selectbox(
        "Select Granularity",
        ["1-day", "1-week", "1-month"],
        on_change=reset_forecast,
    )
    # Check that the horizon is greater than granularity
    valid, h_int, g_int = validate_horizon_vs_granularity(horizon, granularity)

    if not valid:
        forecast_expander.error("Forecast horizon must be greater than granularity.")
        st.stop()

    # Model selection (this is a simple list of boosting algorithms for now)

    model = forecast_expander.selectbox(
        "Select Model",
        _models,
        on_change=reset_forecast,
    )

    top_k_features = forecast_expander.select_slider(
        "Top K features", list(range(1, 21, 1)), on_change=reset_forecast
    )
    top_k_features = (
        top_k_features if not forecast_expander.checkbox("Use all features") else 10**6
    )

    # metric = forecast_expander.selectbox(
    #     "Select Metric",
    #     _metrics,
    #     on_change=reset_forecast,
    # )

    return {
        "horizon": h_int,
        "granularity": g_int,
        "model": model,
        # "metric": metric,
        "top_k_features": top_k_features,
    }


def process_forecast_table(df: pd.DataFrame, date_name: str):
    df = df.round(2)
    df[date_name] = df[date_name].dt.strftime("%m/%d/%Y")
    cols = df.drop(date_name, axis=1).columns.tolist()
    df = df[[date_name] + cols]
    return df


def main():
    """Main"""
    st.set_page_config(layout="wide")
    st.title("Demand Forecasting")

    # File upload section in the sidebar
    upload_expander = st.sidebar.expander("Upload data", expanded=True)

    is_standard_format = (
        upload_expander.radio("Standard Format", ["Yes", "No"]) == "Yes"
    )

    if is_standard_format:
        dates_file, sales_file, prices_file = upload_standard_data()

        # Load the uploaded data
        if dates_file and sales_file and prices_file:
            sales_df, dates = default_prepare_datasets(
                dates_file, sales_file, prices_file
            )
        else:
            st.sidebar.warning(
                "Please upload all three CSV files (dates, Sales, Prices)."
            )
            st.stop()

    else:
        sales_file = upload_data(upload_expander)

        if sales_file:
            sales_df = pd.read_csv(sales_file)
            dates = pd.read_csv(DATES)
        else:
            upload_expander.warning("Please upload CSV file")
            st.stop()

    target_name, date_name, segment_name, segments = get_dataset_features(
        sales_df, is_standard_format
    )
    sales_df = sales_df.rename(
        {target_name: "target", segment_name: "segment", date_name: "timestamp"}, axis=1
    )

    no_history = add_new_segments(sales_df)

    forecast_expander = st.sidebar.expander("Forecast Settings", expanded=True)
    forecast_settings = get_forecast_settings(forecast_expander)
    num_steps_for_clusterization = forecast_expander.select_slider(
        "Clusterization history", list(range(1, 13, 1)), value=3
    )

    # Filter the data based on selected SKU and Store
    filtered_sales = sales_df.copy()

    filtered_sales = filtered_sales[filtered_sales["segment"].isin(segments)]

    if len(filtered_sales) == 0:
        st.warning(
            f'History for column "{segment_name}" doesn\'t have value "{segments}"'
        )
        sales_st = st.empty()

    if len(filtered_sales) > 0:
        filtered_sales = filtered_sales.sort_values(by="timestamp")

    forecast = forecast_expander.button("Get Forecast")

    # Button to trigger the forecast request
    if forecast:
        # Create payload with forecast settings
        payload = {
            "target_segment_names": segments,
            "data": encode_dataframe(sales_df),
            "horizon": forecast_settings.get("horizon", None),
            "granularity": forecast_settings.get("granularity", None),
            "model": forecast_settings.get("model", None),
            "top_k_features": forecast_settings.get("top_k_features", None),
            "no_history": no_history,
        }
        cluster_payload = {
            "data": encode_dataframe(
                filter_by_time_window(
                    sales_df, "timestamp", "1-month", num_steps_for_clusterization
                )
            )
        }
        # Send request to the backend (example backend port assumed to be 8000)
        # Update this with the correct backend URL
        response = requests.post(FORECAST_URL, json=payload, timeout=TIMEOUT)
        clusters_response = requests.post(
            CLUSTER_URL, json=cluster_payload, timeout=TIMEOUT
        )
        st.session_state["response"] = response
        st.session_state["clusters_response"] = clusters_response

    elif "response" in st.session_state:
        pass

    else:
        st.stop()

    # Process the response
    if (
        "response" in st.session_state
        and st.session_state["response"].status_code == 200
    ):
        # append last history point to prediction
        response = st.session_state["response"].json()
        forecast_data = decode_dataframe(response["encoded_predictions"])
        metrics_data = decode_dataframe(response["encoded_metrics"])
        forecast_data = forecast_data.rename(
            {
                "timestamp": "date",
                "target": "predicted",
                "target_0.025": "lower",
                "target_0.975": "upper",
            },
            axis=1,
        )

        table = st.expander("Forecast Table")
        # Display the forecast data
        forecast_data_for_display = process_forecast_table(forecast_data, "date")
        table.data_editor(forecast_data_for_display, width=640, hide_index=True)

        mtable = st.expander("Metrics Table")
        # Display the forecast data
        mtable.data_editor(metrics_data, width=480, hide_index=True)

        if no_history is not None:
            segments.append(no_history["name"])

    else:
        st.error("Failed to get forecast. Please check your settings and try again.")
        st.stop()

    plots_section = st.expander("Plots", expanded=True)
    plots_section.subheader(f"Forecast for {', '.join(segments)}")
    cutoff = plots_section.selectbox(
        "Display history",
        ["1-week", "1-month", "3-month", "1-year", "All"],
        index=2,
    )
    sales_st = plots_section.empty()
    sales_plot = None

    # Filter data by the selected time window
    if len(filtered_sales) > 0:
        sales_for_display = filter_by_time_window(filtered_sales, "timestamp", cutoff)
        dates_for_display = filter_by_time_window(dates, date_name, cutoff)

        # Используем granularity из forecast_settings
        # Агрегируем данные
        granularity = forecast_settings["granularity"]
        sales_for_display = aggregate(sales_for_display, granularity)

        sales_for_display["upper"] = sales_for_display["target"]
        sales_for_display["lower"] = sales_for_display["target"]
        sales_for_display = sales_for_display.rename(
            {"timestamp": "date", "target": "predicted"}, axis=1
        )

        sales_for_display = sales_for_display.sort_values(by="date")

        event_dates = dates_for_display[dates_for_display["event_type_1"].notna()][
            ["date", "event_name_1", "event_type_1"]
        ]

        sales_plot = None

        for segment, c in zip(segments, _pallette):
            seg_hist = sales_for_display[sales_for_display["segment"] == segment]

            if len(seg_hist) == 0:
                continue

            sales_plot = (
                forecast_plot(
                    data=seg_hist,
                    segment=segment,
                    trace_name=segment,
                    fig=sales_plot,
                    scatter_args={"line": {"color": c}},
                    plot_ci=False,
                )
                if len(seg_hist) > 0
                else sales_plot
            )

            forecast_data.loc[len(forecast_data), :] = seg_hist.iloc[-1, :]

        forecast_data = forecast_data.sort_values(by="date")

        sales_plot = add_events(event_dates, sales_plot)

        min_y, max_y = (
            sales_for_display["predicted"].min(),
            sales_for_display["predicted"].max(),
        )

        min_x, max_x = (sales_for_display[date_name].min(), forecast_data["date"].max())

        sales_plot = add_minmax(sales_plot, min_y, max_y, min_x, max_x)

    for segment, c in zip(segments, _pallette):
        forecast_seg = forecast_data[forecast_data["segment"] == segment]

        trace_name = (
            segment
            if no_history is not None
            and no_history["name"] == segment
            and no_history["name"] != ""
            else None
        )

        sales_plot = forecast_plot(
            data=forecast_seg,
            segment=segment,
            trace_name=trace_name,
            fig=sales_plot,
            scatter_args={"line": {"color": c, "dash": "dash"}},
            plot_ci=len(segments) == 1,  # plot CI only if forecast for 1 segment
        )

    sales_st.plotly_chart(sales_plot)

    cluster_section = st.expander("Clusterization")

    if (
        "clusters_response" in st.session_state
        and st.session_state["clusters_response"].status_code == 200
    ):
        response = st.session_state["clusters_response"].json()
        clusters_df = decode_dataframe(response["encoded_dataframe"])

        clusters_df = clusters_df.groupby("cluster").agg(list)
        clusters_df["size"] = clusters_df["segment"].apply(len)

        cluster_section.data_editor(clusters_df)


if __name__ == "__main__":
    main()
