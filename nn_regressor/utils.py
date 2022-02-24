# -*- coding: utf-8 -*-
import logging
import os

import yaml

logger = logging.getLogger("__main__")


def dump_yaml(data: dict, file_path: str):
    """
    Dump a dict into a yaml file.

    Parameters
    ----------
    data : dict
        _description_
    file_path : str
        _description_
    """
    with open(file_path, "w") as handler:
        yaml.dump(data, stream=handler)


def list2query(columns: list, table_name=""):

    """
    This function convert a list of columns into a sql-compliant partial query

    :param columns: [description]
    :type columns: list
    :param table_name: name of the table, defaults to ''
    :type table_name: str, optional
    :return: [description]
    :rtype: [type]
    """
    query_part = ""
    if table_name != "":
        table_name = table_name + "."
    for column in columns:
        query_part = query_part + table_name + column + ", "
    return query_part[0:-2]


def load_yaml(file_path: str) -> dict:
    """
    An utility function to load a yaml file from the config_file folder.

    file_path: str
        the name of the yaml (without extension)
    """
    with open(file_path, "r", encoding="utf-8") as handler:
        return yaml.load(handler, Loader=yaml.FullLoader)


def file_folder_exists(path: str):
    """
    Return True if a file or folder exists.

    :param path: the full path to be checked
    :type path: str
    """
    try:
        os.stat(path)
        return True

    except Exception as e:
        logger.error("Error - not folder exists. Returning False..")
        return False


def select_or_create(path: str):
    """
    Check if a folder exists. If it doesn't, it create the folder.

    :param path: path to be selected
    :type path: str
    """
    if not file_folder_exists(path):
        os.makedirs(path)
    return path
