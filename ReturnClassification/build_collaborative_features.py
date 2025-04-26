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

from ReturnClassification.metrics_data_prepare import main_data_prepare


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
    # 数据读取，拆分训练集和测试集，以及 原始指标数据
    x_train, y_train, x_test, y_test, metrics_data = main_data_prepare(
        the_fund_code='510050.SH',  # 指定基金代码，此处为 '510050.SH'
        n_days=5,  # 预测未来收益的天数，变量 d 在循环中定义，表示不同的预测周期
        folder_path='../Data',  # 基金价格数据的文件夹路径，默认为 '../Data'
        metrics_folder='../Data/Metrics',  # 基金指标数据的文件夹路径，默认为 '../Data/Metrics'
        train_start=None,  # 训练集开始日期，如果为 None，则从数据的最早日期开始
        train_end='2024-03-31',  # 训练集结束日期，指定为 '2024-11-30'
        test_start='2024-04-01',  # 测试集开始日期，指定为 '2024-12-01'
        test_end='2025-04-30',  # 测试集结束日期，指定为 '2025-04-30'
        nan_method='drop',  # 处理缺失值的方法，默认为 'drop'（删除缺失值），可选 'median' 或 'mean'
        period_metrics=True,  # 是否使用区间指标，默认为 True
        rolling_metrics=True,  # 是否使用滚动指标，默认为 True
        standardize_method='zscore',  # 指标标准化的方法,可选: 'minmax', 'zscore', 'both', 'none'。
        basic_data_as_metric=True,  # 是否将基本数据（如开盘价、收盘价、交易量等）作为特征数据，默认为 True
        return_threshold=0.0,  # 标签生成方法，未来收益率大于 0.01 的样本标记为 1，否则为 0
        dim_reduction=False,  # 是否PCA
        index_folder_path='../Data/Index',  # 指数数据的文件夹路径，默认为 '../Data/Index'
        index_close_as_metric=False,  # 是否使用指数收盘价作为指标数据，默认为 True
    )

    selected_metrics = [
        'low', 'high', 'amount', 'open', 'close', 'vol',
        'pct', 'change', 'log',
        'TotalReturn:5d', 'TotalReturn:10d', 'TotalReturn:15d', 'TotalReturn:25d',
        'Volatility:5d', 'Volatility:10d', 'Volatility:15d', 'Volatility:25d',
        'AvgLow:5d', 'AvgLow:10d', 'AvgLow:15d', 'AvgLow:25d',
        'AvgHigh:5d', 'AvgHigh:10d', 'AvgHigh:15d', 'AvgHigh:25d',
        'VolAvg:5d', 'VolAvg:10d', 'VolAvg:15d', 'VolAvg:25d',
        'PriceSigma:3', 'PriceSigma:5', 'PriceSigma:10', 'PriceSigma:15',
        'CloseMA:3', 'CloseMA:5', 'CloseMA:10', 'CloseMA:15',
        'VolMA:3', 'VolMA:5', 'VolMA:10', 'VolMA:15',
        'TRIX:3', 'TRIX:5', 'TRIX:10', 'TRIX:15',
        'PVT:0', 'OBV:0',
        'RSI:3', 'RSI:5', 'RSI:10', 'RSI:15',
        'EMA:3', 'EMA:5', 'EMA:10', 'EMA:15',
        'PSY:3', 'PSY:5', 'PSY:10', 'PSY:15',
        'CCI:3', 'CCI:5', 'CCI:10', 'CCI:15',
        'CR:3', 'CR:5', 'CR:10', 'CR:15',
        'VR:3', 'VR:5', 'VR:10', 'VR:15',
        'AR:3', 'AR:5', 'AR:10', 'AR:15',
        'BR:3', 'BR:5', 'BR:10', 'BR:15',
        'PDI:3', 'PDI:5', 'PDI:10', 'PDI:15',
        'MDI:3', 'MDI:5', 'MDI:10', 'MDI:15',
        'DKX:3', 'DKX:5', 'DKX:10', 'DKX:15',
        'BIAS:3', 'BIAS:5', 'BIAS:10', 'BIAS:15',
        'KDJ-K-3:3', 'KDJ-K-3:5', 'KDJ-K-3:10', 'KDJ-K-3:15',
        'KDJ-D-3:3', 'KDJ-D-3:5', 'KDJ-D-3:10', 'KDJ-D-3:15',
        'KDJ-J-3:3', 'KDJ-J-3:5', 'KDJ-J-3:10', 'KDJ-J-3:15',
        'BollDo-2:3', 'BollDo-2:5', 'BollDo-2:10', 'BollDo-2:15',
        'BollUp-2:3', 'BollUp-2:5', 'BollUp-2:10', 'BollUp-2:15',
    ]

    # # 训练模型
    # get_cross_metrics(x_train, y_train, selected_metrics, steps=2,
    #                   model_folder_path='../Data/Models', joblib_file_name='autofeat_model_2_10.joblib')

    x_train = get_cross_metrics_transform(x_train, selected_metrics,
                                          model_folder_path='../Data/Models',
                                          joblib_file_name='autofeat_model_2_10.joblib')
    print(x_train)