# -*- encoding: utf-8 -*-
"""
@File: product_filter.py
@Modify Time: 2025/8/28 15:01       
@Author: Kevin-Chen
@Descriptions: 
"""
import os
import pandas as pd

# 拼接相对路径
product_priority = pd.read_csv("/Users/chenjunming/Desktop/CIB_wealth_manage/calculate/config/product_priority.csv")
product_priority['三级分类码值'] = product_priority['三级分类码值'].astype(str).apply(
    lambda x: str(int(float(x))).zfill(8)
)


def setting_by_sjdy(df):
    """
    根据三级分类码值设置是否可买入标识

    参数:
        df (pandas.DataFrame): 包含三级分类码值和是否可买入字段的数据框

    返回:
        pandas.DataFrame: 更新了是否可买入标识的数据框
    """
    # 获取不允许增持的三级分类码值集合
    invalid = set(product_priority[product_priority['是否允许增持'] == '否']['三级分类码值'])

    # 将不允许增持的三级分类对应的数据行的是否可买入标识设为False
    df.loc[df['三级分类码值'].apply(lambda x: x in invalid), '是否可买入'] = False

    return df
