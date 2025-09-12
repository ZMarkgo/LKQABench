import configparser

def load_config(config_file='config.ini', encoding='utf-8') -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_file, encoding=encoding)
    return config

def get_db_config(config: configparser.ConfigParser) -> dict:
    """
    :param config: 配置对象
    :return: 数据库配置字典 {
        'host': 主机地址,
        'user': 用户名,
        'password': 密码,
        'database': 数据库名,
    }
    """
    db_config = {
        'host': config['db_config']['host'],
        'user': config['db_config']['user'],
        'password': config['db_config']['password'],
        'database': config['db_config']['database']
    }
    return db_config

def get_api_key_secret_key(config: configparser.ConfigParser) -> str:
    return config['db_config']['api_key_secret_key']