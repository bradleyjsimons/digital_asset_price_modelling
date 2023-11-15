"""
blockchain_com_api.py
---------------------

This module provides a function to fetch chart data from the Blockchain.com Charts API.

The main function is `fetch_blockchain_chart_data`, which takes the name of a chart and optional parameters for the timespan, rolling average, start time, format, and whether to limit the number of datapoints returned. It constructs the URL for the API request, sends the request, and returns the fetched data. If the request fails, it raises an exception.

Example usage:

    from api.blockchain_com_api import fetch_blockchain_chart_data

    data = fetch_blockchain_chart_data('hash-rate')

Functions:
- fetch_blockchain_chart_data(chart_name, timespan=None, rolling_average=None, start=None, format='json', sampled='true'): Fetches chart data from the Blockchain.com Charts API.
"""

import requests


def fetch_blockchain_chart_data(
    chart_name,
    timespan=None,
    rolling_average=None,
    start=None,
    format="json",
    sampled="true",
):
    """
    Fetches chart data from the Blockchain.com Charts API.

    :param chart_name: The name of the chart to fetch.
    :param timespan: The duration of the chart (optional).
    :param rolling_average: The duration over which the data should be averaged (optional).
    :param start: The datetime at which to start the chart (optional).
    :param format: The format of the data, either 'json' or 'csv'. Defaults to 'json'.
    :param sampled: Whether to limit the number of datapoints returned for performance reasons. Defaults to 'true'.
    :return: The fetched chart data.
    """
    base_url = "https://api.blockchain.info/charts/"
    url = f"{base_url}{chart_name}?format={format}&sampled={sampled}"

    if timespan is not None:
        url += f"&timespan={timespan}"

    if rolling_average is not None:
        url += f"&rollingAverage={rolling_average}"

    if start is not None:
        url += f"&start={start}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()  # or response.text if format is 'csv'
    else:
        response.raise_for_status()
