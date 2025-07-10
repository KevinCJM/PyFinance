# -*- encoding: utf-8 -*-
"""
@File: A01_DataPrepare.py
@Modify Time: 2025/7/10 09:52       
@Author: Kevin-Chen
@Descriptions: 
"""
import numpy as np
import pandas as pd
import os
from A02_OperatorLibrary import winsorize  # 导入winsorize函数

# 定义原始数据文件路径
basic_info_files = {
    "log": "../Data/wide_log_return_df.parquet",
    "high": "../Data/wide_high_df.parquet",
    "low": "../Data/wide_low_df.parquet",
    "vol": "../Data/wide_vol_df.parquet",
    "amount": "../Data/wide_amount_df.parquet",
    "close": "../Data/wide_close_df.parquet",
    "open": "../Data/wide_open_df.parquet",
}

# 定义预处理后数据保存路径
PROCESSED_DATA_DIR = "../Data/Processed_ETF_Data/"


def prepare_and_save_etf_data(output_dir=PROCESSED_DATA_DIR):
    """
    加载ETF数据，进行缺失值和异常值预处理，并将处理后的数据保存为Parquet文件。
    
    Args:
        output_dir (str): 预处理后数据保存的目录。
    
    Returns:
        None
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始数据
    log_df = pd.read_parquet(basic_info_files["log"])
    high_df = pd.read_parquet(basic_info_files["high"])
    low_df = pd.read_parquet(basic_info_files["low"])
    vol_df = pd.read_parquet(basic_info_files["vol"])
    amount_df = pd.read_parquet(basic_info_files["amount"])
    close_df = pd.read_parquet(basic_info_files["close"])
    open_df = pd.read_parquet(basic_info_files["open"])

    print("原始数据加载完成！")
    print(f"log_df shape: {log_df.shape}")
    print(f"high_df shape: {high_df.shape}")
    print(f"low_df shape: {low_df.shape}")
    print(f"vol_df shape: {vol_df.shape}")
    print(f"amount_df shape: {amount_df.shape}")
    print(f"close_df shape: {close_df.shape}")
    print(f"open_df shape: {open_df.shape}")

    # --- 缺失值处理 (按DesignDoc.md规定) ---
    # 高、低、开、收: 前向填充 (ffill)
    high_df = high_df.ffill()
    low_df = low_df.ffill()
    open_df = open_df.ffill()
    close_df = close_df.ffill()

    # 成交量、成交额、日对数收益率: 零值填充 (fillna(0))
    vol_df = vol_df.fillna(0)
    amount_df = amount_df.fillna(0)
    log_df = log_df.fillna(0)  # 如果原始价格已处理，此项应极少缺失。若有，填充0。

    # --- 异常值处理 (按DesignDoc.md规定) ---
    # 高、低、开、收: 不对原始价格进行异常值处理

    # 成交量、成交额、日对数收益率: 双侧缩尾 (Winsorization, 1%和99%分位数)
    # 注意：winsorize函数默认对DataFrame的列（axis=0）进行操作，
    # 但这里需要对每个日期（横截面，axis=1）进行缩尾。
    vol_df = winsorize(vol_df, lower_percentile=0.01, upper_percentile=0.99, axis=1)
    amount_df = winsorize(amount_df, lower_percentile=0.01, upper_percentile=0.99, axis=1)
    log_df = winsorize(log_df, lower_percentile=0.01, upper_percentile=0.99, axis=1)

    # --- 标准化 (按DesignDoc.md规定) ---
    # 原始数据不进行标准化。标准化处理主要针对因子值，在因子计算完成后进行。

    print("\n数据预处理完成！")
    print(f"log_df shape after preprocessing: {log_df.shape}")

    # 保存预处理后的数据为Parquet文件
    print(f"正在将预处理后的数据保存到: {output_dir}")
    log_df.to_parquet(os.path.join(output_dir, "processed_log_df.parquet"))
    high_df.to_parquet(os.path.join(output_dir, "processed_high_df.parquet"))
    low_df.to_parquet(os.path.join(output_dir, "processed_low_df.parquet"))
    vol_df.to_parquet(os.path.join(output_dir, "processed_vol_df.parquet"))
    amount_df.to_parquet(os.path.join(output_dir, "processed_amount_df.parquet"))
    close_df.to_parquet(os.path.join(output_dir, "processed_close_df.parquet"))
    open_df.to_parquet(os.path.join(output_dir, "processed_open_df.parquet"))
    print("数据保存完成！")


def load_processed_etf_data(input_dir=PROCESSED_DATA_DIR):
    """
    从Parquet文件加载预处理后的ETF数据，并转换为NumPy数组。
    
    Args:
        input_dir (str): 预处理后数据保存的目录。
        
    Returns:
        tuple: 包含预处理后的log_np, high_np, low_np, vol_np, amount_np, close_np, open_np。
               所有数据均为NumPy数组。
    """
    print(f"正在从 {input_dir} 加载预处理后的数据...")
    log_df = pd.read_parquet(os.path.join(input_dir, "processed_log_df.parquet"))
    high_df = pd.read_parquet(os.path.join(input_dir, "processed_high_df.parquet"))
    low_df = pd.read_parquet(os.path.join(input_dir, "processed_low_df.parquet"))
    vol_df = pd.read_parquet(os.path.join(input_dir, "processed_vol_df.parquet"))
    amount_df = pd.read_parquet(os.path.join(input_dir, "processed_amount_df.parquet"))
    close_df = pd.read_parquet(os.path.join(input_dir, "processed_close_df.parquet"))
    open_df = pd.read_parquet(os.path.join(input_dir, "processed_open_df.parquet"))
    print("数据加载完成！")

    # 转换为NumPy数组
    log_np = log_df.values
    high_np = high_df.values
    low_np = low_df.values
    vol_np = vol_df.values
    amount_np = amount_df.values
    close_np = close_df.values
    open_np = open_df.values

    return log_np, high_np, low_np, vol_np, amount_np, close_np, open_np


# 示例用法
if __name__ == "__main__":
    # 1. 准备并保存数据
    prepare_and_save_etf_data()

    # 2. 加载已保存的数据并转换为NumPy数组
    log_np, high_np, low_np, vol_np, amount_np, close_np, open_np = load_processed_etf_data()

    print("\n加载并转换为NumPy数组后的数据形状:")
    print(f"log_np shape: {log_np.shape}")
    print(f"high_np shape: {high_np.shape}")
    # 可以在这里添加一些断言或打印部分数据来验证处理结果
    # print("\nProcessed log_np head (first 5 rows, first 5 columns):")
    # print(log_np[:5, :5])
