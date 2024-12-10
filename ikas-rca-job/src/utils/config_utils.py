# utils/config_utils.py
import configparser

def read_properties_file(file_path):
    """
    读取 properties 文件
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    return config
