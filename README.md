# Automated Demand Forecasting

## Setup

``dev`` Setup up and activate virtual environment (``python 3.10``) ([venv](https://docs.python.org/3/tutorial/venv.html) or [conda](https://docs.anaconda.com/miniconda/miniconda-install/)) and run:

    git clone https://github.com/zeinovich/dream-team.git
    cd dream-team
    python -m pip install poetry
    poetry install

## Data

``shop_sales_dates.csv``

    date
    wm_yr_wk: Encoded week (i.e. '11136' -> 1-11-36 - IDK, year 2011, 36th week)
    weekday
    wday: INT for week day (starts with Sat = 1)
    month
    year
    event_name_1: Describes events, like holidays, religiuos or big sports events (OrthodoxEaster, Ramadan starts, NBAFinalsStart)
    event_type_1: Describes type of event: Sporting, Cultural, National, Religious
    event_name_2: same for 2nd event (optional)
    event_type_2: same
    date_id: ID for date for JOINs
    CASHBACK_STORE_1: If cashback program was active in STORE_1
    CASHBACK_STORE_2: If cashback program was active in STORE_2
    CASHBACK_STORE_3: If cashback program was active in STORE_3

``shop_sales_prices.csv``

    store_id: one of 'STORE_1', 'STORE_2', 'STORE_3' 
    item_id: item ID (in format "<STORE_ID>_<ID>)
    wm_yr_wk: Encoded week
    sell_price: FLOAT for price, price is set for a week

``shop_sales.csv``

    item_id
    store_id
    date_id
    cnt