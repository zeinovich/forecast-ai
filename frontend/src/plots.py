from typing import Dict, Any, List
from datetime import timedelta

import pandas as pd
from plotly import graph_objects as go

_pallette = [
    "#5ba300",
    "#89ce00",
    "#0073e6",
    "#e6308a",
    "#b51963",
]


def sku_plot(
    df: pd.DataFrame, x: str, y: str, segment_name: str, segments: List[str]
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
    plot = go.Figure()

    for segment in segments:
        df_ = df[df[segment_name] == segment]
        plot.add_trace(
            go.Scatter(
                x=df_[x],
                y=df_[y],
                mode="lines",
                name=segment,
                showlegend=False,
            )
        )

    return plot


def add_minmax(
    plot: go.Figure, min_y: float, max_y: float, min_x: float, max_x: float
) -> go.Figure:
    # Add dashed horizontal lines for min and max Y with annotations
    plot.add_shape(
        type="line",
        x0=min_x,
        x1=max_x,
        y0=min_y,
        y1=min_y,
        line={"color": "Green", "width": 2, "dash": "dash"},
        name=f"Min: {min_y}",
        showlegend=True,
    )

    plot.add_shape(
        type="line",
        x0=min_x,
        x1=max_x,
        y0=max_y,
        y1=max_y,
        line={"color": "Red", "width": 2, "dash": "dash"},
        name=f"Max: {max_y}",
        showlegend=True,
    )

    return plot


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

    cmap = dict(zip(event_dates["event_type_1"].unique(), _pallette))

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


def forecast_plot(
    forecast_data: pd.DataFrame,
    segment: str,
    fig: go.Figure = None,
    scatter_args: Dict[str, Any] = None,
) -> go.Figure:
    """
    Plotting function for forecast.

    Args:
        forecast_data (pd.DataFrame): Dataframe consisting of following columns:\
            `["date", "predicted", "upper", "lower"]`, where "upper" and "lower"\
            are confidence intervals (CIs). If it's not possible to compute CIs, leave NaN

        historical (pd.DataFrame, optional): Historical data for particular SKU in \
            a particular store. Should contain columns `["date", "cnt"]`. Defaults to None.

    Returns:
        go.Figure: Forecast plot
    """
    if not fig:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_data["date"],
            y=forecast_data["predicted"],
            mode="lines",
            name=f"Prediction ({segment})",
            showlegend=True,
            **scatter_args,
        )
    )

    # Prediction intervals (Uncertainty bounds) shaded area
    fig.add_trace(
        go.Scatter(
            x=forecast_data["date"],
            y=forecast_data["upper"],
            mode="lines",
            line=dict(width=0),
            name=f"Upper Bound ({segment})",
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
            name=f"Lower Bound ({segment})",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Predicted Demand with Uncertainty Bounds",
        xaxis_title="Date",
        yaxis_title="Demand",
    )

    return fig