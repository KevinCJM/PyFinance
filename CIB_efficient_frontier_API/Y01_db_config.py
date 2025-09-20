# -*- encoding: utf-8 -*-
"""
@File: Y01_db_config.py.py
@Modify Time: 2025/9/20 17:31       
@Author: Kevin-Chen
@Descriptions: 数据库配置信息
"""

db_type = 'mysql'
db_ip = '127.0.0.1'
db_port = '3306'
db_name = 'mysql'
db_user = 'root'
db_password = '*********'

if db_type == 'mysql':
    db_driver = 'pymysql'
    db_url = f'mysql+{db_driver}://{db_user}:{db_password}@{db_ip}:{db_port}/{db_name}?charset=utf8mb4'
