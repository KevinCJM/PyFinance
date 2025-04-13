# -*- encoding: utf-8 -*-
"""
@File: metrics_data_prepare.py
@Modify Time: 2025/4/13 10:20       
@Author: Kevin-Chen
@Descriptions: 指标预处理
"""
import os
import duckdb
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option('display.width', 1000)  # 表格不分段显示
pd.set_option('display.max_columns', 1000)  # 显示字段的数量

selected_fund = '510050.SH'  # 上证50ETF
# 指定文件夹路径
folder_path = '../Data/Metrics'

# query = f"""
#     SELECT trade_date, {selected_fund}
#     FROM '../Data/wide_close_df.parquet'
# """
# df_selected = duckdb.query(query).to_df()
# print(df_selected)

df_final = None
for idx, fil in enumerate(os.listdir(folder_path)):
    if not fil.endswith('.parquet'):
        continue

    full_path = os.path.join(folder_path, fil)
    alias = f"t{idx}"  # 给每个表一个别名，防止字段冲突

    query = f"""
    SELECT *
    FROM '{full_path}'
    WHERE ts_code = '{selected_fund}'
    """

    df = duckdb.query(query).to_df()
    # 删除无用字段，比如 __index_level_0__
    df = df.loc[:, ~df.columns.str.startswith("__")]

    # 第一个表作为基础
    if df_final is None:
        df_final = df
    else:
        df_final = pd.merge(df_final, df, on=['ts_code', 'date'], how='outer')

# 按日期排序
df_final = df_final.sort_values('date').reset_index(drop=True)

print(df_final)
