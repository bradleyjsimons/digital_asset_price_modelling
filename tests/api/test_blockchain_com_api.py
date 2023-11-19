"""
This module contains tests for the fetch_blockchain_chart_data function in the blockchain_com_api module.

Tests cover successful requests, failed requests, and different parameter combinations.
"""

import pytest
from src.api.blockchain_com_api import fetch_blockchain_chart_data
from requests.exceptions import HTTPError


@pytest.fixture(autouse=True)
def mock_requests_get(mocker):
    """
    A pytest fixture that mocks requests.get to return a successful response.
    This fixture is automatically used in all tests in this module.
    """
    mock = mocker.patch("requests.get")
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = {"values": []}  # Mock response content
    yield


def test_fetch_blockchain_chart_data_success():
    """
    Test the fetch_blockchain_chart_data function with a known chart name and additional parameters.
    The function should return a dictionary with a 'values' key.
    """
    # Test with a known chart name and additional parameters
    chart_name = "hash-rate"
    timespan = "5weeks"
    rolling_average = "8hours"
    start = "2022-01-01"
    data = fetch_blockchain_chart_data(chart_name, timespan, rolling_average, start)
    assert isinstance(data, dict)  # The function should return a dictionary
    assert "values" in data  # The returned data should have a 'values' key


def test_fetch_blockchain_chart_data_failure(mocker):
    """
    Test the fetch_blockchain_chart_data function with an unknown chart name.
    The function should raise an HTTPError.
    """
    # Mock the requests.get function to return a failure response
    mock_get = mocker.patch("requests.get")
    mock_response = mocker.Mock()
    mock_response.raise_for_status.side_effect = HTTPError
    mock_response.status_code = 404  # Simulate a not found error
    mock_get.return_value = mock_response
    # Test with an unknown chart name
    chart_name = "unknown-chart"
    with pytest.raises(HTTPError):
        data = fetch_blockchain_chart_data(chart_name)
