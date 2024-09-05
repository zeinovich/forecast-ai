import streamlit as st
import pandas as pd
import plotly.express as px


# Function to validate that the horizon is greater than the granularity
def validate_horizon_vs_granularity(horizon, granularity):
    horizons = {"1-day": 1, "1-week": 7, "1-month": 30}
    granularities = {"1-day": 1, "1-week": 7, "1-month": 30}

    return horizons[horizon] > granularities[granularity]


# [TODO] - zeinovich - move to preprocessing pipeline
# [TODO] - zeinovich AutoML - how to pass DF to backend
def read_merge(dates_file, sales_file, prices_file):
    dates = pd.read_csv(dates_file)
    sales = pd.read_csv(sales_file)
    prices = pd.read_csv(prices_file)

    sales["SKU"] = sales["item_id"].apply(lambda x: x[-3:])

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


def main():
    # Page selection in the sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page", ["Input Page", "Other Pages (Coming Soon)"]
    )

    # Logic for the Input Page
    if page == "Input Page":
        st.title("Demand Forecasting - Input Page")

        # File upload section in the sidebar
        st.sidebar.header("Upload CSV Files")
        dates_file = st.sidebar.file_uploader("Upload Dates CSV", type="csv")
        sales_file = st.sidebar.file_uploader("Upload Sales CSV", type="csv")
        prices_file = st.sidebar.file_uploader("Upload Prices CSV", type="csv")

        # Load the uploaded data
        if dates_file and sales_file and prices_file:
            sales_df = read_merge(dates_file, sales_file, prices_file)

            # Assuming SKU and Store columns exist in sales and prices
            sku_list = sales_df["SKU"].unique()
            store_list = sales_df["store_id"].unique()

            # SKU and Store selection
            selected_sku = st.sidebar.selectbox("Select SKU", sku_list)
            selected_store = st.sidebar.selectbox("Select Store", store_list)

            # Forecast horizon and granularity on the main page
            st.header("Forecast Settings")
            forecast_horizon = st.selectbox(
                "Select Forecast Horizon", ["1-day", "1-week", "1-month"]
            )
            granularity = st.selectbox(
                "Select Granularity", ["1-day", "1-week", "1-month"]
            )

            # Check that the horizon is greater than granularity
            if not validate_horizon_vs_granularity(forecast_horizon, granularity):
                st.error("Forecast horizon must be greater than granularity.")

            else:
                # Filter the data based on selected SKU and Store
                filtered_sales = sales_df[
                    (sales_df["SKU"] == selected_sku)
                    & (sales_df["store_id"] == selected_store)
                ]

                if len(filtered_sales) == 0:
                    st.warning(
                        f"SKU {selected_sku} has never been sold in store {selected_store}"
                    )

                # Plotting the sales data
                st.subheader(f"Sales for SKU {selected_sku} at Store {selected_store}")

                # [TODO] - zeinovich - add filters for date (periods) based on horizon
                sales_plot = px.line(
                    filtered_sales, x="date", y="cnt", title="Sales Over Time"
                )
                st.plotly_chart(sales_plot)

                # Plotting the price data
                st.subheader(f"Prices for SKU {selected_sku} at Store {selected_store}")
                price_plot = px.line(
                    filtered_sales, x="date", y="sell_price", title="Prices Over Time"
                )
                st.plotly_chart(price_plot)
        else:
            st.sidebar.warning(
                "Please upload all three CSV files (Dates, Sales, Prices)."
            )

    # Placeholder for future pages
    elif page == "Other Pages (Coming Soon)":
        st.title("Coming Soon")
        st.write("More features and pages will be added here in the future.")


if __name__ == "__main__":
    main()
