"""
This module contains tests for the functions in the blockchain module.

Tests cover the fetching and processing of data from the Blockchain.com Charts API.
"""

import pytest
import pandas as pd
from src.features.blockchain import (
    get_blockchain_data,
    get_hash_rate_over_time,
    get_avg_block_size,
    get_network_difficulty,
    get_miners_revenue,
    get_mempool_size,
)


@pytest.fixture(autouse=True)
def mock_fetch_blockchain_chart_data(mocker):
    """
    A pytest fixture that mocks the fetch_blockchain_chart_data function to return a mock response.
    This fixture is automatically used in all tests in this module.
    """
    mock = mocker.patch("src.features.blockchain.fetch_blockchain_chart_data")
    mock.return_value = {"values": [{"x": 1633046400, "y": 133.3795}]}
    return mock


def test_get_blockchain_data_error(mocker):
    """
    Test the get_blockchain_data function with an error in the fetch_blockchain_chart_data function.
    """
    # Mock the fetch_blockchain_chart_data function to raise an exception
    mocker.patch(
        "src.features.blockchain.fetch_blockchain_chart_data", side_effect=Exception
    )
    # Call the function and check that it returns None
    df = get_blockchain_data("hash-rate")
    assert df is None


def test_get_blockchain_data(mock_fetch_blockchain_chart_data):
    """
    Test the get_blockchain_data function.
    """
    df = get_blockchain_data("hash-rate")
    assert not df.empty
    assert "hash-rate" in df.columns


def test_get_hash_rate_over_time(mock_fetch_blockchain_chart_data):
    """
    Test the get_hash_rate_over_time function.
    """
    df = get_hash_rate_over_time()
    assert not df.empty
    assert "hash-rate" in df.columns


def test_get_avg_block_size(mock_fetch_blockchain_chart_data):
    """
    Test the get_avg_block_size function.
    """
    df = get_avg_block_size()
    assert not df.empty
    assert "avg-block-size" in df.columns


def test_get_network_difficulty(mock_fetch_blockchain_chart_data):
    """
    Test the get_network_difficulty function.
    """
    df = get_network_difficulty()
    assert not df.empty
    assert "difficulty" in df.columns


def test_get_miners_revenue(mock_fetch_blockchain_chart_data):
    """
    Test the get_miners_revenue function.
    """
    df = get_miners_revenue()
    assert not df.empty
    assert "miners-revenue" in df.columns


def test_get_mempool_size(mock_fetch_blockchain_chart_data):
    """
    Test the get_mempool_size function.
    """
    df = get_mempool_size()
    assert not df.empty
    assert "mempool-size" in df.columns
