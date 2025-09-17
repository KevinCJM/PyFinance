import pandas as pd
import numpy as np
import datetime
from copy import deepcopy
from typing import Dict
from scipy.spatial import ConvexHull
import time
import warnings

warnings.filterwarnings('ignore')
from itertools import combinations
import matplotlib

matplotlib.use('Qt5Agg')  # 必须在 plt 之前设置
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 加权求和计算指数组合每日收益数据
def generate_alloc_perf(portfolio_alloc: Dict) -> Dict:
    port_daily = hist_value_r.copy()
    port_daily = port_daily[portfolio_alloc.keys()]

    port_daily.loc[:, 'r'] = port_daily[portfolio_alloc.keys()].values @ np.array(list(portfolio_alloc.values()))

    port_daily['value'] = (1 + port_daily['r']).cumprod()
    port_daily.loc[port_daily.index[0] - datetime.timedelta(days=1), 'value'] = 1
    port_daily = port_daily.sort_index()

    port_vol = np.std(port_daily['r'].apply(lambda r: np.log(r + 1)).dropna(), ddof=1) * np.sqrt(252)
    port_ret = np.log(port_daily['value'].iloc[-1] / port_daily['value'].iloc[0]) / (len(port_daily) - 1) * 252

    return {"ret_annual": float(port_ret), "vol_annual": float(port_vol)}


# 计算不同权重下资产组合的收益波动信息
def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray, p95=1.65) -> np.ndarray:
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    # Step 1: 所有组合收益率（向量化）
    port_return_daily = port_daily @ portfolio_allocs.T

    # Step 2: 累计收益率（log 累乘）
    port_cum_returns = np.cumprod(1 + port_return_daily, axis=0)

    # Step 3: 年化收益率
    log_total_ret = np.log(port_cum_returns[-1, :])
    port_ret_annual = log_total_ret / (port_return_daily.shape[0]) * 252

    # Step 4: 年化波动率（用 log return）
    log_returns = np.log(1 + port_return_daily)
    port_vol_annual = np.std(log_returns, axis=0, ddof=1) * np.sqrt(252)

    sp_annual = port_ret_annual / port_vol_annual
    VaR95_annual = port_ret_annual - port_vol_annual * p95

    # Step 5: 打包结果为 DataFrame
    ret_df = pd.DataFrame({
        "ret_annual": port_ret_annual,
        "vol_annual": port_vol_annual,
        "sp_annual": sp_annual,
        "VaR95_annual": VaR95_annual
    })

    # Step 6: 合并权重数据
    weight_df = pd.DataFrame(portfolio_allocs)

    return pd.concat([weight_df, ret_df], axis=1)


def generate_simplex_grid(n_assets: int, resolution: int):
    """
    n_assets: 几个资产类别（维度）
    resolution: 把 1 划分为 resolution 份，即 Δ=1/resolution
    返回所有加总为1，且每个资产为 Δ 的倍数的组合
    """
    total_slots = resolution + n_assets - 1
    grid = []
    for bars in combinations(range(total_slots), n_assets - 1):
        bars = (-1,) + bars + (total_slots,)
        vec = [bars[i + 1] - bars[i] - 1 for i in range(n_assets)]
        grid.append(np.array(vec) / resolution)
    return np.array(grid)


# 通过近似法求解
def generate_simplex_grid_constraint(n_assets: int, resolution: int, cons: np.array, threshold: float):
    """
    n_assets: 几个资产类别（维度）
    resolution: 把 1 划分为 resolution 份，即 Δ=1/resolution
    返回所有加总为1，且每个资产为 Δ 的倍数的组合
    """
    total_slots = resolution + n_assets - 1
    grid = []
    for bars in combinations(range(total_slots), n_assets - 1):
        bars = (-1,) + bars + (total_slots,)
        vec = [bars[i + 1] - bars[i] - 1 for i in range(n_assets)]
        if abs(np.array(vec) / resolution - cons).max(axis=1).min() <= threshold:
            grid.append(np.array(vec) / resolution)
    return np.array(grid)


def cal_ef2_v4_ultra_fast(data):
    """
    版本4: 超高速版本 - 适用于大数据集
    优点：
    - Pure NumPy操作，避免所有pandas开销
    - 最小内存分配
    - 最高性能
    缺点：
    - 代码相对复杂
    - 失去pandas的易读性下`
    时间复杂度: O(n log n)
    适用场景: 大数据集 (> 100K点)
    """
    # 获取NumPy数组
    ret_values = data['ret_annual'].values
    vol_values = data['vol_annual'].values

    # 排序索引
    sorted_idx = np.argsort(ret_values)[::-1]

    # 向量化计算累计最小值
    sorted_vol = vol_values[sorted_idx]
    cummin_vol = np.minimum.accumulate(sorted_vol)

    # 识别有效前沿点
    on_ef_sorted = (sorted_vol == cummin_vol)

    # 重新映射到原始顺序
    on_ef = np.zeros(len(data), dtype=bool)
    on_ef[sorted_idx] = on_ef_sorted

    data['on_ef'] = on_ef
    return data


def find_best(weights):
    random_port_perf = []
    for w in weights:
        asset_alloc = dict(zip(assets_list, list(w)))
        alloc_perf = generate_alloc_perf(asset_alloc)
        random_port_perf.append(dict(**asset_alloc, **alloc_perf))
    random_port_perf = pd.DataFrame(random_port_perf)
    random_port_perf['on_ef'] = False
    random_port_perf = cal_ef2_v4_ultra_fast(random_port_perf)
    return random_port_perf, random_port_perf[random_port_perf['on_ef'] == True].iloc[:, :5].values


# 绘制有效前沿图
def scatter_plot(data, risk_label='var_annual'):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    for i, row in data.iterrows():
        color_style = 'bo' if row['on_ef'] else 'ro'
        ax.plot(row[risk_label], row['ret_annual'], color_style, markersize=1)
    plt.gca().invert_xaxis()  # 这里反转x轴
    plt.title('资产组合有效前沿', fontsize=14)
    plt.xlabel(risk_label, fontsize=12)
    plt.ylabel('ret_annual', fontsize=12)
    # 优化布局
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 读取数据
    hist_value = pd.read_excel('历史净值数据.xlsx', sheet_name='历史净值数据')
    # 数据输入处理
    hist_value = hist_value.set_index('date')
    hist_value.index = pd.to_datetime(hist_value.index)
    hist_value = hist_value.dropna().sort_index(ascending=True)
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
        '激进型': 'C6'
    }, axis=1)

    # 预设的风险等级配比(proposed_alloc_df)，前端输入
    proposed_alloc = {
        'C1': {'货币现金类': 1.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }
    proposed_alloc_df = pd.DataFrame(proposed_alloc).T
    proposed_alloc_df = proposed_alloc_df.fillna(0)

    # 生成资产大类列表(assets_list)
    assets_list = proposed_alloc_df.columns.tolist()

    # 大类资产净值数据转为每日收益率数据
    hist_value_r = hist_value.pct_change().dropna()

    # 计算大类资产的年化收益率、年化波动率、夏普比率
    VaR95 = 1.96
    hist_perf = pd.concat(
        [
            np.log(hist_value.iloc[-1] / hist_value.iloc[0]) / (hist_value.count() - 1) * 252,
            np.std(np.log(hist_value / hist_value.shift(-1))[1:], axis=0) * np.sqrt(252)
        ],
        axis=1
    )
    print(f"数据形状: {hist_value_r.shape}")
    print(f"时间范围: {hist_value_r.index[0]} 到 {hist_value_r.index[-1]}")
    print("\n各资产年化统计指标:")
    hist_perf.columns = ['Annual Return', 'Annual Volatility']
    hist_perf['Sharpe Ratio'] = hist_perf['Annual Return'] / hist_perf['Annual Volatility']
    hist_perf['VaR95'] = hist_perf['Annual Return'] - hist_perf['Annual Volatility'] * VaR95

    # =========================== 分层网格搜索：100 -> 20 -> 50 -> 100 ===========================

    # # 第一层：1% 精度
    print("\nLayer 1: 10% precision (resolution=10)")
    start_time = time.time()
    w_r10 = generate_simplex_grid(n_assets=5, resolution=100)
    print(f"Generated {len(w_r10):,} weight combinations")
    r10, best_r10 = find_best(w_r10)
    time_r10 = time.time() - start_time
    print(f"Found {len(best_r10)} efficient frontier points in {time_r10:.2f} seconds")

    # # 第二层：5% 精度
    print("\nLayer 2: 5% precision (resolution=20)")
    start_time = time.time()
    w_r20 = generate_simplex_grid_constraint(n_assets=5, resolution=20, cons=best_r10, threshold=0.1)
    print(f"Generated {len(w_r20):,} constrained weight combinations")
    if len(w_r20) > 0:
        r20, best_r20 = find_best(w_r20)
        time_r20 = time.time() - start_time
        print(f"Found {len(best_r20)} efficient frontier points in {time_r20:.2f} seconds")
    else:
        print("No valid combinations found")
        best_r20 = best_r10
        time_r20 = 0

    # # 第三层：2% 精度
    print("\nLayer 3: 2% precision (resolution=50)")
    start_time = time.time()
    w_r50 = generate_simplex_grid_constraint(n_assets=5, resolution=50, cons=best_r20, threshold=0.05)
    print(f"Generated {len(w_r50):,} constrained weight combinations")
    if len(w_r50) > 0:
        r50, best_r50 = find_best(w_r50)
        time_r50 = time.time() - start_time
        print(f"Found {len(best_r50)} efficient frontier points in {time_r50:.2f} seconds")
    else:
        print("No valid combinations found")
        best_r50 = best_r20
        time_r50 = 0

    # # 第四层：1% 精度
    print("\nLayer 4: 1% precision (resolution=100)")
    start_time = time.time()
    w_r100 = generate_simplex_grid_constraint(n_assets=5, resolution=100, cons=best_r50, threshold=0.02)
    print(f"Generated {len(w_r100):,} constrained weight combinations")
    if len(w_r100) > 0:
        r100, best_r100 = find_best(w_r100)
        time_r100 = time.time() - start_time
        print(f"Found {len(best_r100)} efficient frontier points in {time_r100:.2f} seconds")
    else:
        print("No valid combinations found")
        best_r100 = best_r50
        time_r100 = 0

    print("\n" + "=" * 50)
    print("Hierarchical search completed!")
    total_time = time_r10 + time_r20 + time_r50 + time_r100
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Final efficient frontier points: {len(best_r100)}")

    # # 第五层：0.5% 精度
    print("\nLayer 5: 1% precision (resolution=200)")
    start_time = time.time()
    w_r200 = generate_simplex_grid_constraint(n_assets=5, resolution=200, cons=best_r100, threshold=0.01)
    print(f"Generated {len(w_r200):,} constrained weight combinations")
    if len(w_r200) > 0:
        r200, best_r200 = find_best(w_r200)
        time_r200 = time.time() - start_time
        print(f"Found {len(best_r200)} efficient frontier points in {time_r200:.2f} seconds")
    else:
        print("No valid combinations found")
        best_r200 = best_r100
        time_r200 = 0
    '''
    r100_df = pd.read_csv("./data/新有效前沿点.csv")
    grid_data = pd.read_csv("./data/网格搜索0.5.csv")
    filtered_df = pd.read_csv("./data/网格搜索1.csv")
    '''
    # 网格搜索1
    filtered_df = pd.concat([r10, r20, r50, r100]).drop_duplicates().sort_values(by='ret_annual').reset_index(
        drop=True)  # 75692 个点
    filtered_df['var_annual'] = filtered_df['ret_annual'] - filtered_df['vol_annual'] * VaR95
    # 网格搜索0.5
    grid_data = pd.concat([r10, r20, r50, r100, r200]).drop_duplicates().sort_values(by='ret_annual').reset_index(
        drop=True)  # 75692 个点
    grid_data['var_annual'] = grid_data['ret_annual'] - grid_data['vol_annual'] * VaR95
    # 新有效前沿点
    r100_df = filtered_df[filtered_df['on_ef'] == True]
    # r100_df_ = cal_ef2_v4_ultra_fast(r100_df)

    scatter_plot(grid_data)

    # 可视化有效前沿
    大类资产 = ["C1", "C2", "C3", "C4", "C5", "C6"]
    # if len(best_r100) > 0:
    plt.figure(figsize=(12, 8))
    # 绘制有效前沿
    plt.scatter(
        x=filtered_df['vol_annual'],  # 使用波动率作为x轴
        y=filtered_df['ret_annual'],  # 收益率
        s=10,
        c='blue',
        alpha=0.2,
        label='Other Choice'
    )

    plt.scatter(
        x=r100_df['vol_annual'],  # 使用波动率作为x轴
        y=r100_df['ret_annual'],  # 收益率
        s=10,
        c='red',
        alpha=0.8,
        label='Efficient Frontier'
    )

    # 添加单个资产点
    for i, asset in enumerate(大类资产):
        if asset in hist_perf.index:
            if i == 0:
                plt.scatter(
                    hist_perf.loc[asset, 'Annual Volatility'],  # 使用年化波动率
                    hist_perf.loc[asset, 'Annual Return'],
                    s=100,
                    alpha=1,
                    c='orange',
                    label='C1~C6'
                )
            else:
                plt.scatter(
                    hist_perf.loc[asset, 'Annual Volatility'],  # 使用年化波动率
                    hist_perf.loc[asset, 'Annual Return'],
                    s=100,
                    alpha=1,
                    c='orange'
                )

    plt.title('Efficient Frontier', fontsize=14)
    plt.xlabel('Annual Volatility', fontsize=12)  # 更新x轴标签
    plt.ylabel('Annual Return', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
