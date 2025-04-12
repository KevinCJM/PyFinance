# -*- encoding: utf-8 -*-
"""
@File: set_tushare.py
@Modify Time: 2025/4/3 17:27       
@Author: Kevin-Chen
@Descriptions: 在这里输入 Tushare 的 token
"""
import tushare as ts

ts.set_token('TushareToken')  # 替换为你的 Tushare token
pro = ts.pro_api()

