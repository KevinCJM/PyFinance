# @title 基础引入和函数
import os
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

plt.rcParams['font.sans-serif'] = ['STHeiti']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.options.display.float_format = '{:.4%}'.format


def annualize_ret(ret: float, n_per_year: int = 252) -> float:
    return np.power(1 + ret, n_per_year) - 1


def annualize_vol(vol: float, n_per_year: int = 252) -> float:
    return np.sqrt(vol * n_per_year)


def annualize_vol_std(vol: float, n_per_year: int = 252) -> float:
    return vol * np.sqrt(n_per_year)


print(os.getcwd())

# @title 加载历史净值数据(hist_value)
# with open('历史净值数据.csv') as file:
#     hist_value = pd.read_csv(file)

hist_value = pd.read_excel('历史净值数据.xlsx')
hist_value = hist_value.set_index('date')
hist_value.index = pd.to_datetime(hist_value.index)
hist_value = hist_value.dropna()
hist_value = hist_value.rename({
    "货基指数": "货币现金类",
    '固收类': '固定收益类',
    '混合类': '混合策略类',
    '权益类': '权益投资类',
    '另类': '另类投资类',
    '安逸型': 'C1',
    '谨慎型': 'C2',
    '稳健型': 'C3',
    '增长型': 'C4',
    '进取型': 'C5',
    '激进型': 'C6'}, axis=1)

# @title 预设的风险等级配比(proposed_alloc_df)
proposed_alloc = {
    'C1': {'货币现金类': 0.9, '同业存单': 0.1},
    'C2': {'货币现金类': 0.2, '固定收益类': 0.8},
    'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35},
    'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
    'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
    'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1},
}
proposed_alloc_df = pd.DataFrame(proposed_alloc).T
proposed_alloc_df = proposed_alloc_df.fillna(0)

# @title 生成资产大类列表(assets_list)
assets_list = proposed_alloc_df.columns.tolist()
assets_list.remove('同业存单')

# @title 每日收益率数据(hist_value_r)
hist_value_r = hist_value.pct_change().dropna()
currency_cash_nav = hist_value['货币现金类']

# 假设 currency_cash_nav 已经是原始净值序列，索引为日期
# 1. 计算单日收益率序列
nav = currency_cash_nav.copy()
nav_1d_ret = nav.pct_change(1)

# 2. 以最后一个日期的最后7日收益率均值作为基准
benchmark_7d_ret = nav_1d_ret.iloc[-7:-1].mean()

# 3. 用基准除以前面所有的7日收益率序列，得到权重序列
weights = benchmark_7d_ret / nav_1d_ret

# 4. 补齐权重序列的空值（只用历史数据，不用未来数据）
weights = weights.ffill().fillna(1)  # 前向填充，首位用1

# 5. 计算日度收益率序列
nav_daily_ret = nav.pct_change(1)

# 6. 用权重数据乘以日度收益率序列，得到调整后的日度收益率序列
adj_daily_ret = nav_daily_ret * weights

# 7. 计算调整后的净值序列
adj_nav = (1 + adj_daily_ret).cumprod() * nav.iloc[0]
adj_nav = adj_nav.ffill().fillna(1)

# 8. 按照区间年化收益率算法，计算调整前和调整后年化收益率序列          # 原始净值序列
nav_start = nav.iloc[0]
annual_ret_raw = np.log(nav / nav_start) / np.arange(1, len(nav) + 1) * 252

# 调整后净值序列
adj_nav_start = adj_nav.iloc[0]
annual_ret_adj = np.log(adj_nav / adj_nav_start) / np.arange(1, len(adj_nav) + 1) * 252

# 9. 作图比较
plt.figure(figsize=(12, 6))
plt.plot(annual_ret_raw, label='原始年化收益率')
plt.plot(annual_ret_adj, label='调整后年化收益率')
plt.title('年化收益率对比（原始 vs 调整后）')
plt.xlabel('日期')
plt.ylabel('年化收益率')
plt.legend()
plt.grid(True)
plt.show()

# 打印最后一个日期的年化收益率
print('原始年化收益率（最后一日）:', annual_ret_raw.dropna().iloc[-1])
print('调整后年化收益率（最后一日）:', annual_ret_adj.dropna().iloc[-1])

hist_value['货币现金类'] = adj_nav
hist_value.iloc[0] = 1
hist_value.to_excel('历史净值数据-货币调整.xlsx', sheet_name='历史净值数据')
print('已保存调整后的历史净值数据至 "历史净值数据-货币调整.xlsx"')
