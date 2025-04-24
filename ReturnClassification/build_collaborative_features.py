# -*- encoding: utf-8 -*-
"""
@File: build_collaborative_features.py
@Modify Time: 2025/4/24 16:27       
@Author: Kevin-Chen
@Descriptions: 构建协同特征
"""
import pandas as pd
from autofeat import AutoFeatRegressor
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from ReturnClassification.metrics_data_prepare import get_fund_metrics_data

# 获取指标数据
metrics_data = get_fund_metrics_data(
    selected_fund='510050.SH',  # 指定基金代码，此处为 '510050.SH'
    data_folder_path='../Data',  # 基金价格数据的文件夹路径，默认为 '../Data'
    metrics_folder_path='../Data/Metrics',  # 基金指标数据的文件夹路径，默认为 '../Data/Metrics'
    basic_data_as_metric=True,  # 是否将基本数据（如开盘价、收盘价、交易量等）作为特征数据，默认为 True
    index_folder_path='../Data/Index',  # 指数数据的文件夹路径，默认为 '../Data/Index'
    index_close_as_metric=True,  # 是否使用指数收盘价作为指标数据，默认为 True
)

print(metrics_data)

# Step 1: 使用 AutoFeat 生成静态组合特征（不含时间列）
X_static = metrics_data.drop(columns=["date", "ts_code"])
autofeat_model = AutoFeatRegressor(feateng_steps=2)
y_temp = metrics_data["close"]  # 临时目标变量，只为跑 AutoFeat 不报错
X_autofeat = autofeat_model.fit_transform(X_static, y=y_temp)
print(X_autofeat)

# # Step 2: 使用 tsfresh 生成动态组合特征（含时间列）
# df_tsfresh_ready = metrics_data.copy()
# df_tsfresh_ready["id"] = df_tsfresh_ready["ts_code"]
# df_tsfresh_ready["time"] = df_tsfresh_ready["date"]
#
# X_tsfresh = extract_features(
#     df_tsfresh_ready.drop(columns=["ts_code", "date"]),
#     column_id="id", column_sort="time",
#     disable_progressbar=True
# )
#
# # 清洗缺失
# impute(X_tsfresh)
#
# # Step 3: 合并 AutoFeat + TSFresh 特征
# X_combined = pd.concat([X_autofeat.reset_index(drop=True), X_tsfresh.reset_index(drop=True)], axis=1)

