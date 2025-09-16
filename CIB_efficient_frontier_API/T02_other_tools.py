# -*- encoding: utf-8 -*-
"""
@File: T02_other_tools.py
@Modify Time: 2025/9/16 10:17       
@Author: Kevin-Chen
@Descriptions: 
"""
from __future__ import annotations  # 启用 Python 3.10+ 的类型注解特性

import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from numba import njit, prange

    HAS_NUMBA = True
    print("Numba 可用，启用 JIT 加速。")
except Exception:
    HAS_NUMBA = False
    print("Numba 不可用，使用纯 Python 版本。")

import numpy as np
import pandas as pd


# 返回当前时间戳字符串
def _ts() -> str:
    """返回当前时间戳字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 标准化日志输出，带时间戳
def log(msg: str) -> None:
    """标准化日志输出，带时间戳。"""
    print(f"[{_ts()}] {msg}")


# 读取excel的数据
def load_returns_from_excel(
        excel_path: str,
        sheet_name: str,
        assets_list: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """从 Excel 读取净值数据，生成日收益二维数组 (T,N)。"""
    log(f"加载数据: {excel_path} | sheet={sheet_name}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index(ascending=True)
    df = df.rename(
        {
            "货基指数": "货币现金类",
            "固收类": "固定收益类",
            "混合类": "混合策略类",
            "权益类": "权益投资类",
            "另类": "另类投资类",
        },
        axis=1,
    )

    # 若部分资产列不存在，报错以避免静默问题
    missing = [c for c in assets_list if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    hist_ret_df = df[assets_list].pct_change().dropna()
    # 全局采用 float32，以获得更好的吞吐（矩阵乘法/缓存）
    arr = hist_ret_df.values.astype(np.float32, copy=False)
    log(f"数据加载完成，样本天数={arr.shape[0]}，资产数={arr.shape[1]}")
    return arr, assets_list
