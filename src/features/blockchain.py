"""
This module provides functions to fetch and process data from the Blockchain.com Charts API.

It includes the following functions:

- get_blockchain_data: Fetches data over time from the Blockchain.com Charts API and returns it as a DataFrame.
- get_hash_rate_over_time: Fetches the hash rate over time from the Blockchain.com Charts API and returns it as a DataFrame.
- get_avg_block_size: Fetches the average block size over time from the Blockchain.com Charts API and returns it as a DataFrame.
- get_network_difficulty: Fetches the network difficulty over time from the Blockchain.com Charts API and returns it as a DataFrame.
- get_miners_revenue: Fetches the miners revenue over time from the Blockchain.com Charts API and returns it as a DataFrame.
- get_mempool_size: Fetches the mempool size over time from the Blockchain.com Charts API and returns it as a DataFrame.

Each function takes optional parameters to specify the timespan and start date for the data, and returns a DataFrame with the fetched data.
"""
from src.api.blockchain_com_api import fetch_blockchain_chart_data
import pandas as pd


def get_blockchain_data(
    chart_name,
    timespan=None,
    rolling_average=None,
    start_date=None,
    format="json",
    sampled="true",
):
    """
    Fetches data over time from the Blockchain.com Charts API and returns it as a DataFrame.

    :param chart_name: The name of the chart to fetch data from.
    :param timespan: The duration of the chart (optional).
    :param rolling_average: The duration over which the data should be averaged (optional).
    :param start_date: The start date for the chart data (optional).
    :param format: The format of the data, either 'json' or 'csv'. Defaults to 'json'.
    :param sampled: Whether to limit the number of datapoints returned for performance reasons. Defaults to 'true'.
    :return: A DataFrame with the fetched data.
    """
    try:
        data = fetch_blockchain_chart_data(
            chart_name, timespan, rolling_average, start_date, format, sampled
        )

        # Convert the list of dictionaries to a DataFrame
        dataframe = pd.DataFrame(data)

        # Extract 'x' and 'y' keys from the 'values' column into separate columns
        dataframe[["Date", chart_name]] = pd.DataFrame(
            dataframe["values"].tolist(), index=dataframe.index
        )

        # Convert the 'Date' column to datetime
        dataframe["Date"] = pd.to_datetime(dataframe["Date"], unit="s")

        # Set 'Date' as the index
        dataframe.set_index("Date", inplace=True)

        # Keep only the data column
        dataframe = dataframe[[chart_name]]

        return dataframe
    except Exception as e:
        print(f"An error occurred while fetching the {chart_name} data: {e}")


def get_hash_rate_over_time(timespan=None, start_date=None):
    """
    Fetches the hash rate over time from the Blockchain.com Charts API and returns it as a DataFrame.

    :param timespan: The duration of the chart (optional).
    :param start_date: The start date for the chart data (optional).
    :return: A DataFrame with the fetched hash rate data.
    """
    return get_blockchain_data("hash-rate", timespan=timespan, start_date=start_date)


def get_avg_block_size(timespan=None, start_date=None):
    """
    Fetches the average block size over time from the Blockchain.com Charts API and returns it as a DataFrame.

    :param timespan: The duration of the chart (optional).
    :param start_date: The start date for the chart data (optional).
    :return: A DataFrame with the fetched average block size data.
    """
    return get_blockchain_data(
        "avg-block-size", timespan=timespan, start_date=start_date
    )


def get_network_difficulty(timespan=None, start_date=None):
    """
    Fetches the network difficulty over time from the Blockchain.com Charts API and returns it as a DataFrame.

    :param timespan: The duration of the chart (optional).
    :param start_date: The start date for the chart data (optional).
    :return: A DataFrame with the fetched network difficulty data.
    """
    return get_blockchain_data("difficulty", timespan=timespan, start_date=start_date)


def get_miners_revenue(timespan=None, start_date=None):
    """
    Fetches the miners revenue over time from the Blockchain.com Charts API and returns it as a DataFrame.

    :param timespan: The duration of the chart (optional).
    :param start_date: The start date for the chart data (optional).
    :return: A DataFrame with the fetched miners revenue data.
    """
    return get_blockchain_data(
        "miners-revenue", timespan=timespan, start_date=start_date
    )


def get_mempool_size(timespan=None, start_date=None):
    """
    Fetches the mempool size over time from the Blockchain.com Charts API and returns it as a DataFrame.

    :param timespan: The duration of the chart (optional).
    :param start_date: The start date for the chart data (optional).
    :return: A DataFrame with the fetched mempool size data.
    """
    return get_blockchain_data("mempool-size", timespan=timespan, start_date=start_date)
