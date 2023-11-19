"""
This module contains tests for the create_model_directory function in the folder_manager module.
"""

import os
import pytest

from src.utils.folder_manager import create_model_directory


def test_create_model_directory_default_path(mocker):
    """
    Test the create_model_directory function with the default base path.
    """
    # Mock the os.path.exists and os.makedirs functions
    mocker.patch("os.path.exists", return_value=False)
    mock_makedirs = mocker.patch("os.makedirs")

    # Call the function
    directory = create_model_directory()

    # Check that the directory is in the default base path
    assert directory.startswith("src/models")

    # Check that os.makedirs was called with the correct arguments
    mock_makedirs.assert_called_once_with(directory, exist_ok=True)


def test_create_model_directory_custom_path(mocker):
    """
    Test the create_model_directory function with a custom base path.
    """
    # Mock the os.path.exists and os.makedirs functions
    mocker.patch("os.path.exists", return_value=False)
    mock_makedirs = mocker.patch("os.makedirs")

    # Call the function with a custom base path
    base_path = "custom/path"
    directory = create_model_directory(base_path)

    # Check that the directory is in the custom base path
    assert directory.startswith(base_path)

    # Check that os.makedirs was called with the correct arguments
    mock_makedirs.assert_called_once_with(directory, exist_ok=True)


def test_create_model_directory_existing_directory(mocker):
    """
    Test the create_model_directory function when the directory already exists.
    """
    # Mock the os.path.exists function to return True
    mocker.patch("os.path.exists", return_value=True)
    # Mock the os.listdir function to return an empty list
    mocker.patch("os.listdir", return_value=[])
    # Mock the os.makedirs function
    mock_makedirs = mocker.patch("os.makedirs")

    # Call the function
    directory = create_model_directory()

    # Check that os.makedirs was called with the correct arguments
    mock_makedirs.assert_called_once_with(directory, exist_ok=True)


def test_create_model_directory_existing_non_empty_directory(mocker):
    """
    Test the create_model_directory function when the directory already exists and is not empty.
    """
    # Mock the os.path.exists function to return True at first, then False
    mocker.patch("os.path.exists", side_effect=[True, False])
    # Mock the os.listdir function to return a non-empty list at first, then an empty list
    mocker.patch("os.listdir", side_effect=[["file"], []])
    # Mock the os.makedirs function
    mock_makedirs = mocker.patch("os.makedirs")

    # Call the function
    directory = create_model_directory()

    # Check that the directory name has a counter appended
    assert directory.endswith("_1")

    # Check that os.makedirs was called with the correct arguments
    mock_makedirs.assert_called_once_with(directory, exist_ok=True)
