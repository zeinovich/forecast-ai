"""
Streamlit app

App has 3 pages: Input, [TODO] Prediction, [TODO] AutoML

To start app run: `streamlit run ./frontend/app.py`

Author: zeinovich
"""

from datetime import timedelta
import requests  # dev

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import pandas as pd
import numpy as np  # dev

from plots import sku_plot, add_events, add_minmax, forecast_plot
from preprocessing import (
    default_prepare_datasets,
    validate_horizon_vs_granularity,
    filter_by_time_window,
    encode_dataframe,
    decode_dataframe,
)

BACKEND_URL = "http://localhost:8000/forecast"
TIMEOUT = 300  # HTTP timeout in seconds

DATES = "./data/raw/shop_sales_dates.csv"

_models = ["linear.LinearPerSegmentModel", "AutoARIMAModel", "prophet.ProphetModel"]

_metrics = [
    "MSE",
    "RMSE",
    "MAE",
    "MAPE",
]


# [TODO] - zeinovich - make adaptive form for file download
def upload_standard_data(expander: DeltaGenerator):
    dates_file = expander.file_uploader("Upload dates CSV", type="csv")
    sales_file = expander.file_uploader("Upload Sales CSV", type="csv")
    prices_file = expander.file_uploader("Upload Prices CSV", type="csv")

    css = """
        <style>
            [data-testid='stFileUploader'] {
                width: max-container;
            }
            [data-testid='stFileUploader'] section {
                padding: 0;
                float: left;
            }
            [data-testid='stFileUploader'] section > input + div {
                display: none;
            }
            [data-testid='stFileUploader'] section + div {
                float: left;
                padding-top: 0;
            }

        </style>
    """

    st.sidebar.markdown(css, unsafe_allow_html=True)

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


def reset_forecast():
    if "response" in st.session_state:
        del st.session_state["response"]


def get_dataset_features(df: pd.DataFrame):
    expander = st.sidebar.expander("DFU settings", expanded=True)
    segment_name = expander.selectbox(
        "Select ID column", df.columns.tolist(), on_change=reset_forecast
    )
    unique_segments = df[segment_name].unique().tolist()

    segments = expander.multiselect(
        f"Select {segment_name}", sorted(unique_segments), on_change=reset_forecast
    )

    target_name = expander.selectbox(
        "Select target column", df.columns.tolist(), on_change=reset_forecast
    )
    date_name = expander.selectbox(
        "Select date column", df.columns.tolist(), on_change=reset_forecast
    )

    return target_name, date_name, segment_name, segments


def get_forecast_settings(forecast_expander):
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

    metric = forecast_expander.selectbox(
        "Select Metric",
        _metrics,
        on_change=reset_forecast,
    )

    return h_int, g_int, model, metric


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
        dates_file, sales_file, prices_file = upload_standard_data(upload_expander)

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

    # [TODO] - zeinovich - make adaptive form for target cols selection
    # [TODO] - zeinovich - place value if no store selection
    target_name, date_name, segment_name, segments = get_dataset_features(sales_df)

    forecast_expander = st.sidebar.expander("Forecast Settings", expanded=True)
    horizon, granularity, model, metric = get_forecast_settings(forecast_expander)

    # Filter the data based on selected SKU and Store
    filtered_sales = sales_df.copy()

    filtered_sales = filtered_sales[filtered_sales[segment_name].isin(segments)]

    if len(filtered_sales) == 0:
        st.warning(
            f'History for column "{segment_name}" doesn\'t have value "{segments}"'
        )
        # [TODO] - zeinovich - how render plots with no history
        sales_st = st.empty()

    if len(filtered_sales) > 0:
        filtered_sales = filtered_sales.sort_values(by=date_name)

    # Button to trigger the forecast request
    if forecast_expander.button("Get Forecast"):
        # Create payload with forecast settings
        payload = {
            "target_name": target_name,
            "date_name": date_name,
            "segment_name": segment_name,
            "target_segment_names": segments,
            "data": encode_dataframe(sales_df),
            "horizon": horizon,
            "granularity": granularity,
            "model": model,
            "metric": metric,
            "top_k_features": 10,
        }

        # Send request to the backend (example backend port assumed to be 8000)
        # Update this with the correct backend URL
        # [TODO] - zeinovich - postprocessing of response
        response = requests.post(BACKEND_URL, json=payload, timeout=TIMEOUT)
        # response = np.random.normal(
        #     filtered_sales["cnt"].mean(),
        #     filtered_sales["cnt"].std(),
        #     size=horizon // granularity,
        # )
        # response = pd.DataFrame(response, columns=["predicted"])
        # response["date"] = [
        #     filtered_sales["date"].tolist()[-1] + timedelta(days=1) * (i + 1)
        #     for i in range(horizon // granularity)
        # ]
        # response["upper"] = np.random.normal(
        #     1.2 * response["predicted"],
        #     0.5,
        #     size=len(response),
        # )
        # response["lower"] = np.random.normal(
        #     0.8 * response["predicted"],
        #     0.5,
        #     size=len(response),
        # )
        # response["lower"] = np.min(response[["lower", "upper"]].to_numpy(), axis=1)
        # response["upper"] = np.max(response[["lower", "upper"]].to_numpy(), axis=1)
        st.session_state["response"] = response.json()

    elif "response" in st.session_state:
        pass

    else:
        st.stop()

    # Process the response
    if response.status_code == 200:
        # if st.session_state["response"] is not None:
        # [TODO] - zeinovich - postprocessing of response
        # append last history point to prediction
        forecast_data = decode_dataframe(response["encoded_predictions"])
        forecast_metrics = decode_dataframe(response["encoded_metrics"])
        # forecast_data = st.session_state["response"]
        # st.success("Forecast generated successfully!")

        table = st.expander("Forecast Table")
        # Display the forecast data
        forecast_data_for_display = process_forecast_table(forecast_data, date_name)
        table.data_editor(forecast_data_for_display, use_container_width=True)

    else:
        st.error("Failed to get forecast. Please check your settings and try again.")
        st.stop()

    # # [TODO] - zeinovich - check if historical is present
    # # and preprocess it
    # Plotting the sales data
    # [TODO] - zeinovich - how to print out
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
        sales_for_display = filter_by_time_window(filtered_sales, date_name, cutoff)
        dates_for_display = filter_by_time_window(dates, date_name, cutoff)

        event_dates = dates_for_display[dates_for_display["event_type_1"].notna()][
            ["date", "event_name_1", "event_type_1"]
        ]
        # [TODO] - multiSKU
        sales_plot = sku_plot(
            sales_for_display,
            x=date_name,
            y=target_name,
            title="Sales over Time",
        )
        sales_plot = add_events(event_dates, sales_plot)
        forecast_data.loc[len(forecast_data)] = pd.Series(
            [
                filtered_sales[date_name].iloc[-1],
                filtered_sales[target_name].iloc[-1],
                filtered_sales[target_name].iloc[-1],
                filtered_sales[target_name].iloc[-1],
            ],
            index=["date", "predicted", "upper", "lower"],
        )

        forecast_data = forecast_data.sort_values("date")

        min_y, max_y = (
            sales_for_display[target_name].min(),
            sales_for_display[target_name].max(),
        )

        min_x, max_x = (sales_for_display[date_name].min(), forecast_data["date"].max())

        sales_plot = add_minmax(sales_plot, min_y, max_y, min_x, max_x)

    sales_plot = forecast_plot(
        forecast_data,
        sales_plot,
        scatter_args={"line": {"color": "Green", "dash": "dash"}},
    )

    sales_st.plotly_chart(sales_plot)


if __name__ == "__main__":
    main()
