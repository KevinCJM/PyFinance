import pandas as pd
import numpy as np

# 读取万得原始数据
index_df = pd.read_excel("万得指数数据.xlsx", sheet_name="万得原始数据")

# 规范日期列
index_df["date"] = pd.to_datetime(index_df["date"])
index_df = index_df[index_df["date"] < pd.to_datetime("2025-09-01")]

# 识别数值列（除 date 外）
value_cols = [c for c in index_df.columns if c != "date"]
print(value_cols)

# 将包含千分位逗号的字符串转为数值（避免使用 applymap 已被弃用）
# 1) 统一为字符串后用正则去除逗号与空白字符
tmp_vals = index_df[value_cols].astype(str).replace({r"[,\s]": ""}, regex=True)
# 2) 列级别安全转数值（无法转换的置为 NaN）
index_df[value_cols] = tmp_vals.apply(pd.to_numeric, errors="coerce")

# 设置索引
index_df = index_df.set_index("date")
print(index_df)

# 计算日收益率
ret_df = index_df.pct_change()

# 对首行（由 pct_change 产生）NaN 用 0 填充；其余 NaN 保留以避免引入伪收益
if len(ret_df) > 0:
    first_idx = ret_df.index[0]
    ret_df.loc[first_idx] = ret_df.loc[first_idx].fillna(0.0)

print(ret_df.head())
