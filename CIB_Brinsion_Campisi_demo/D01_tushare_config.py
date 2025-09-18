# -*- encoding: utf-8 -*-
"""
@File: D01_tushare_config.py
@Modify Time: 2025/9/8 08:34       
@Author: Kevin-Chen
@Descriptions: 
"""

import tushare as ts

ts.set_token('YOUR_TUSHARE_TOKEN')
pro = ts.pro_api()
