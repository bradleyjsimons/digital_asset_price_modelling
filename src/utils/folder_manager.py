"""
This module provides a function to create a directory for storing model files.

Function:
- create_model_directory: Creates a new directory with a unique name based on the current date.

This function uses the os and datetime modules to create a new directory in the specified base path. 
The name of the new directory is based on the current date, and if a directory with the same name 
already exists, a counter is appended to the name to make it unique.
"""
import os
import datetime


def create_model_directory(base_path="src/models"):
    """
    Creates a new directory with a unique name based on the current date.

    This function creates a new directory in the specified base path. The name of the new directory
    is based on the current date, and if a directory with the same name already exists and is not empty,
    a counter is appended to the name to make it unique.

    :param base_path: A string representing the base path where the new directory will be created.
                      Defaults to "src/models".
    :return: A string representing the full path of the newly created directory.
    """
    today = datetime.datetime.now().strftime("%Y%m%d")
    dir_name = today
    counter = 1

    full_path = os.path.join(base_path, dir_name)
    while os.path.exists(full_path) and os.listdir(full_path):
        dir_name = f"{today}_{counter}"
        full_path = os.path.join(base_path, dir_name)
        counter += 1

    os.makedirs(full_path, exist_ok=True)
    return full_path
