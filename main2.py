import yfinance
from bs4 import BeautifulSoup
import requests
from io import StringIO
from datetime import datetime
import pytz
import threading
from backtester import save_pickle, load_pickle
import pandas as pd

S_AND_P_500_LINK = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
GRANULARITY_DEFAULT = "1d"
DATASET_OBJ = "dataset.obj"


def _helper_func(i, tickers, period_start, period_end, dataframes, granularity=GRANULARITY_DEFAULT):
    """
    Function to help with getting historical data for a given ticker.

    :param i: Index of the ticker in the list of tickers.
    :param tickers: List of ticker symbols.
    :param period_start: Start date of the period for which historical data is needed.
    :param period_end: End date of the period for which historical data is needed.
    :param dataframes: Dictionary to store the retrieved historical data.
    :param granularity: (optional) Granularity level for the historical data. Defaults to GRANULARITY_DEFAULT.

    :return: None
    """
    print(tickers[i])
    dataframes[i] = get_history_data(tickers[i], period_start, period_end, granularity=granularity)


def get_ticker_history(tickers, start_dates, end_dates, granularity=GRANULARITY_DEFAULT):
    """
    Retrieves historical ticker data for specified tickers within a given date range.

    :param tickers: A list of tickers to retrieve historical data for.
    :param start_dates: A list of start dates for the historical data.
    :param end_dates: A list of end dates for the historical data.
    :param granularity: The granularity of the historical data (default is GRANULARITY_DEFAULT).

    :return: A tuple containing the filtered list of tickers and the corresponding dataframes.
    """
    dataframes = [None] * len(tickers)
    threads = [
        threading.Thread(target=_helper_func, args=(i, tickers, start_dates, end_dates, dataframes))
        for i in range(len(tickers))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    tickers = [tickers[i] for i in range(len(tickers)) if not dataframes[i].empty]
    dataframes = [df for df in dataframes if not df.empty]
    return tickers, dataframes


def get_s_and_p500_tickers():
    """
    Returns a list of ticker symbols for companies listed on the S&P 500.

    :return: A list of ticker symbols.
    """
    response = requests.get(S_AND_P_500_LINK)
    soup = BeautifulSoup(response.content, 'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(StringIO(str(table)))
    tickers = list(df[0]["Symbol"])
    return tickers


def get_history_data(ticker, start, end, granularity=GRANULARITY_DEFAULT, tries=0):
    """
    Retrieve historical stock data for a given ticker within a specified date range.

    :param ticker: The stock ticker symbol.
    :param start: The start date of the data range (inclusive).
    :param end: The end date of the data range (inclusive).
    :param granularity: The interval size of the data (default: GRANULARITY_DEFAULT).
    :param tries: The number of retry attempts in case of an error (default: 0).
    :return: A pandas DataFrame containing the historical stock data.
    """
    try:
        data = yfinance.Ticker(ticker).history(
            start=start,
            end=end,
            auto_adjust=True,
            interval=granularity
        ).reset_index()
        data = data.rename(columns={
            "Date": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        if data.empty:
            return pd.DataFrame()

        data["datetime"] = data["datetime"].dt.tz_convert(pytz.utc)
        data = data.drop(columns=["Dividends", "Stock Splits"])
        data = data.set_index("datetime")
        return data
    except Exception as err:
        if tries < 5:
            return get_history_data(ticker, start, end, granularity, tries + 1)
        return pd.DataFrame()


def get_tickers_and_dfs(start, end):
    """
    Retrieve tickers and dataframes for a given period.

    :param start: Start date of the period in YYYY-MM-DD format.
    :param end: End date of the period in YYYY-MM-DD format.
    :return: A tuple containing tickers (list) and dataframes (list) for the given period.
    """
    try:
        tickers, dfs = load_pickle(DATASET_OBJ)
    except Exception as err:
        tickers = get_s_and_p500_tickers()
        tickers, dfs = get_ticker_history(tickers, start, end)
        save_pickle(DATASET_OBJ, (tickers, dfs))
    return tickers, dfs


period_start = datetime(2010, 1, 1, tzinfo=pytz.utc)
period_end = datetime(2020, 1, 1, tzinfo=pytz.utc)
tickers, dfs = get_tickers_and_dfs(start=period_start, end=period_end)
tickers = tickers[:20]
from backtester import Alpha
alpha = Alpha(insts=tickers, dfs=dfs, start=period_start, end=period_end)
simulation = alpha.run_simulation()

if simulation is not None:
    for portfolio_df in simulation:
        print(portfolio_df)