# -*- encoding: utf-8 -*-
"""
@File: build_collaborative_features.py
@Modify Time: 2025/4/24 16:27       
@Author: Kevin-Chen
@Descriptions: 构建协同特征
"""
import os.path
import time
import joblib
import numpy as np
import pandas as pd
from autofeat import AutoFeatRegressor
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


# from ReturnClassification.metrics_data_prepare import main_data_prepare


# 使用 AutoFeat 生成静态组合特征
def get_cross_metrics(raw_x, raw_y, selected_metrics_list, steps=2,
                      model_folder_path='../Data/Models', joblib_file_name='autofeat_model_2.joblib'):
    """
    使用AutoFeat进行交叉特征指标构建。

    参数:
    raw_x: DataFrame, 原始特征数据。
    raw_y: Series, 目标变量。
    selected_metrics_list: List[str], 选定的特征指标列表。
    steps: int, 特征构建的步骤数，默认为2。
    model_folder_path: str, 模型保存的文件夹。
    joblib_file_name: str, 模型文件名。

    返回:
    DataFrame, 增加交叉项后的特征数据。
    """
    # 开始特征工程的打印信息
    print(f"[AutoFeat开始] 进行交叉特征指标的构建,构建步骤为{steps}步 ... ")
    print(f"\t原始数据结构: {raw_x.shape}")
    s_t = time.time()

    # 挑选用于交叉特征工程的指标
    raw_x = raw_x[selected_metrics_list]
    print(f"\t挑选指标后的数据结构: {raw_x.shape}")

    # 重命名列名，添加前缀 "AF_", 避免字段名与关键字重复
    raw_x.columns = [f"AF_{col}" for col in raw_x.columns]

    # 构建模型
    autofeat_model = AutoFeatRegressor(feateng_steps=steps)

    # 进行特征工程
    x_autofeat = autofeat_model.fit_transform(raw_x, y=raw_y)
    print(f"\t增加交叉项后的数据结构为: {x_autofeat.shape}")

    # 保存模型到本地
    model_file = os.path.join(model_folder_path, joblib_file_name)
    joblib.dump(autofeat_model, model_file)
    print(f"\t模型已保存到: {model_file}")

    print(f"[AutoFeat结束] 完成构建, 耗时{((time.time() - s_t) / 60):.2f}分钟 ... ")
    # 返回增加交叉项后的特征数据
    return x_autofeat


# 使用 AutoFeat 生成选好的特征
def get_cross_metrics_transform(raw_x, selected_metrics_list,
                                model_folder_path='../Data/Models',
                                joblib_file_name='autofeat_model_2.joblib'):
    # 开始特征工程的打印信息
    print(f"[AutoFeat开始] 使用已经训练好的 AutoFeat 模型进行交叉特征指标的构建 ... ")
    print(f"\t原始数据结构: {raw_x.shape}")
    s_t = time.time()

    # 挑选用于交叉特征工程的指标
    raw_x = raw_x[selected_metrics_list]
    print(f"\t挑选指标后的数据结构: {raw_x.shape}")

    # 重命名列名，添加前缀 "AF_", 避免字段名与关键字重复
    new_column_name = [f"AF_{col}" for col in raw_x.columns]
    raw_x.columns = new_column_name

    # 读取模型
    model_file_path = os.path.join(model_folder_path, joblib_file_name)
    loaded_autofeat_model = joblib.load(model_file_path)
    # 构建新增指标
    transformed_df = loaded_autofeat_model.transform(raw_x)
    transformed_df = transformed_df[[col for col in transformed_df.columns if col not in new_column_name]]

    print(f"\t增加交叉项后的数据结构为: {transformed_df.shape}")
    print(f"[AutoFeat结束] 完成构建, 耗时{(time.time() - s_t):.2f}秒 ... ")
    return transformed_df


if __name__ == '__main__':
    pass
