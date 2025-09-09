# @title 基础引入和函数
import pandas as pd
import numpy as np
import cvxpy as cp
import datetime
from copy import deepcopy
from typing import Dict
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
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
    for asset in portfolio_alloc.keys():
        port_daily[asset] = port_daily[asset] * portfolio_alloc[asset]

    port_daily['r'] = port_daily.sum(axis=1)

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
    # port_return_daily: shape (n_days, n_portfolios)
    port_return_daily = port_daily @ portfolio_allocs.T

    # Step 2: 累计收益率（log 累乘）
    port_cum_returns = np.cumprod(1 + port_return_daily, axis=0)

    # Step 3: 年化收益率
    log_total_ret = np.log(port_cum_returns[-1, :])
    port_ret_annual = log_total_ret / (port_return_daily.shape[0]) * 252  # shape (n_portfolios,)

    # Step 4: 年化波动率（用 log return）
    log_returns = np.log(1 + port_return_daily)
    port_vol_annual = np.std(log_returns, axis=0, ddof=1) * np.sqrt(252)  # shape (n_portfolios,)

    sp_annual = port_ret_annual / port_vol_annual

    var95_annual = port_ret_annual - port_vol_annual * p95

    # Step 5: 打包结果为 DataFrame
    ret_df = pd.DataFrame({
        "ret_annual": port_ret_annual,
        "vol_annual": port_vol_annual,
        "sp_annual": sp_annual,
        "var95_annual": var95_annual
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


# @ title 通过近似法求解
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


def find_best(weights):
    random_port_perf = []
    for w in weights:
        asset_alloc = dict(zip(assets_list, list(w)))
        alloc_perf = generate_alloc_perf(asset_alloc)
        random_port_perf.append(dict(**asset_alloc, **alloc_perf))
    random_port_perf = pd.DataFrame(random_port_perf)
    random_port_perf['on_ef'] = False
    points = random_port_perf[['vol_annual', 'ret_annual']].values
    # 计算凸包
    hull = ConvexHull(points)
    # 获取凸包上的点
    hull_points = points[hull.vertices]
    # 按波动率排序
    hull_points = hull_points[np.argsort(hull_points[:, 0])]
    # 筛选“有效前沿”那一段（凸包的右上边界）
    efficient_frontier = []
    max_ret = -np.inf
    for v, r in hull_points:
        if r >= max_ret:
            efficient_frontier.append((v, r))
            max_ret = r
    for r in efficient_frontier:
        random_port_perf.loc[
            (random_port_perf.vol_annual == r[0]) & (random_port_perf.ret_annual == r[1]), 'on_ef'] = True
    return random_port_perf, random_port_perf[random_port_perf['on_ef'] == True].iloc[:, :5].values


# 根据给定的波动率计算资产组合最优收益率
def best_vol_point(alloc_value_r: pd.DataFrame, vol_min: float, vol_max: float, n=1000, var95=1.96):
    '''
    :param vol_min: 最小风险(以现金管理为标准)
    :param vol_max: 最大风险(以权益类为标准)
    :param n: 区分度
    :return:
    '''
    results = []
    # 在[vol_min,vol_max]之间生成1000个波动率。计算每个波动率下面收益最高的点
    for vol in np.linspace(vol_min, vol_max, n):
        # 日对数收益
        log_return_daily = np.log(1 + alloc_value_r.values)
        T, N = log_return_daily.shape
        # 整数变量，表示配置“百分比”乘100后的整数（如40% -> 40）
        w = cp.Variable(N, integer=True)
        # 换算为“比例”权重
        w_scaled = w / 100
        # 每日组合收益
        w_return_daily = log_return_daily @ w_scaled
        # 平均日收益 & 年化
        avg_return = cp.sum(w_return_daily) / T
        avg_return_annual = avg_return * 252
        # 日波动 & 年化波动
        std_return = cp.norm(w_return_daily - avg_return, 2) / cp.sqrt(T)
        std_return_annual = std_return * np.sqrt(252)
        # 约束：非负 + 总比例为 1（即 100）
        constraints = [
            w >= 0,
            cp.sum(w) == 100,
            std_return_annual <= vol  # 年化波动率上限
        ]
        # 最大化年化对数收益
        obj = cp.Maximize(avg_return_annual)
        # 求解
        prob = cp.Problem(obj, constraints)
        prob.solve(cp.ECOS_BB)
        if prob.status == 'optimal':
            results.append([*np.round(w.value).tolist(), std_return_annual.value, avg_return_annual.value])
    # 整理结果
    best_vol_point = pd.DataFrame(results)
    best_vol_point.columns = ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类', 'vol_annual',
                              'ret_annual']
    best_vol_point[['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']] = best_vol_point[
                                                                                                 ['货币现金类',
                                                                                                  '固定收益类',
                                                                                                  '混合策略类',
                                                                                                  '权益投资类',
                                                                                                  '另类投资类']].astype(
        int) / 100
    best_vol_point[['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']] = best_vol_point[
        ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']]
    best_vol_point = best_vol_point.drop_duplicates(
        ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']).reset_index(drop=True)
    # 计算var95
    best_vol_point['var95'] = best_vol_point['ret_annual'] - var95 * best_vol_point['vol_annual']
    best_vol_point.sort_values(by='ret_annual', inplace=True)

    return best_vol_point


# 根据给定的波动率计算资产组合最优收益率
def best_vol_point2(alloc_value_r: pd.DataFrame, vol_min: float, vol_max: float, n=1000, var95=1.96):
    '''
    :param vol_min: 最小风险(以现金管理为标准)
    :param vol_max: 最大风险(以权益类为标准)
    :param n: 区分度
    :return:
    '''
    results = []
    # 在[vol_min,vol_max]之间生成1000个波动率。计算每个波动率下面收益最高的点
    for vol in np.linspace(vol_min, vol_max, n):
        # 日对数收益
        log_return_daily = np.log(1 + alloc_value_r.values)
        T, N = log_return_daily.shape
        # 整数变量，表示配置“百分比”乘100后的整数（如40% -> 40）
        w = cp.Variable(N, integer=True)
        # 换算为“比例”权重
        w_scaled = w / 200
        # 每日组合收益
        w_return_daily = log_return_daily @ w_scaled
        # 平均日收益 & 年化
        avg_return = cp.sum(w_return_daily) / T
        avg_return_annual = avg_return * 252
        # 日波动 & 年化波动
        std_return = cp.norm(w_return_daily - avg_return, 2) / cp.sqrt(T)
        std_return_annual = std_return * np.sqrt(252)
        # 约束：非负 + 总比例为 1（即 100）
        constraints = [
            w >= 0,
            cp.sum(w) == 200,
            std_return_annual <= vol  # 年化波动率上限
        ]
        # 最大化年化对数收益
        obj = cp.Maximize(avg_return_annual)
        # 求解
        prob = cp.Problem(obj, constraints)
        prob.solve(cp.ECOS_BB)
        if prob.status == 'optimal':
            results.append([*np.round(w.value).tolist(), std_return_annual.value, avg_return_annual.value])
    # 整理结果
    best_vol_point = pd.DataFrame(results)
    best_vol_point.columns = ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类', 'vol_annual',
                              'ret_annual']
    best_vol_point[['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']] = best_vol_point[
                                                                                                 ['货币现金类',
                                                                                                  '固定收益类',
                                                                                                  '混合策略类',
                                                                                                  '权益投资类',
                                                                                                  '另类投资类']].astype(
        int) / 200
    best_vol_point[['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']] = best_vol_point[
        ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']]
    best_vol_point = best_vol_point.drop_duplicates(
        ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']).reset_index(drop=True)
    # 计算var95
    best_vol_point['var95'] = best_vol_point['ret_annual'] - var95 * best_vol_point['vol_annual']
    best_vol_point.sort_values(by='ret_annual', inplace=True)

    return best_vol_point


# 找到与输入组合最接近的组合
def find_optimized_portfolio(raw_portfolio, optimized_portfolio_df, var95=1.96):
    raw_portfolio_ = deepcopy(raw_portfolio)
    raw_perf = generate_alloc_perf(raw_portfolio)

    raw_ret = raw_perf['ret_annual']
    raw_vol = raw_perf['vol_annual']
    raw_var95 = raw_ret - var95 * raw_vol
    raw_portfolio_['vol_annual'] = raw_vol
    raw_portfolio_['ret_annual'] = raw_ret
    raw_portfolio_['var95'] = raw_var95

    # max_ret_index = np.argmin(np.abs(optimized_portfolio_df['ret_annual'].values - raw_ret)) #353
    # min_vol_index = np.argmin(np.abs(optimized_portfolio_df['vol_annual'].values - raw_vol)) #360
    # # 从2个最优点的波动率的中点出发寻找最优点，跟翟老师确认过
    # medial_vol_annual = (optimized_portfolio_df.iloc[max_ret_index]['vol_annual']+optimized_portfolio_df.iloc[min_vol_index]['vol_annual'])/2
    # medial_vol_index = np.argmin(np.abs(optimized_portfolio_df['vol_annual'].values - medial_vol_annual))
    # 收益最大点
    max_ret = optimized_portfolio_df[optimized_portfolio_df['vol_annual'] <= raw_vol].sort_values(by='vol_annual',
                                                                                                  ascending=False).iloc[
              0, :]
    # 波动最小点
    min_vol = optimized_portfolio_df[optimized_portfolio_df['ret_annual'] >= raw_ret].sort_values(by='ret_annual',
                                                                                                  ascending=True).iloc[
              0, :]
    # 从2个最优点的波动率的中点出发寻找最优点，跟翟老师确认过
    avg_vol = (max_ret['vol_annual'] + min_vol['vol_annual']) / 2
    medial_vol = np.argmin(np.abs(optimized_portfolio_df['vol_annual'].values - avg_vol))
    # 找到最小距离的索引
    nearest_dist_index = (
            (optimized_portfolio_df[['vol_annual', 'ret_annual']] - np.array([raw_vol, raw_ret])) ** 2).sum(
        axis=1).idxmin()
    # nearest_alloc = np.argmin(np.sum(np.abs(optimized_portfolio_df[['货币现金类','固定收益类','混合策略类','权益投资类','另类投资类']].values - np.array(list(raw_portfolio.values()))), axis=1))

    return {
        'raw_portfolio': raw_portfolio_,  # 持仓组合
        # 'nearest_alloc': optimized_portfolio_df.iloc[nearest_alloc].to_dict(),      #换手率最低
        'max_ret': max_ret.to_dict(),  # 收益最大点
        'min_vol': min_vol.to_dict(),  # 波动最小点
        'medial_vol': optimized_portfolio_df.iloc[medial_vol].to_dict(),
        'nearest_dist': optimized_portfolio_df.iloc[nearest_dist_index].to_dict()
    }


if __name__ == '__main__':
    var95 = 1.96
    # 数据输入
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
        '激进型': 'C6'}, axis=1)

    # 预设的风险等级配比(proposed_alloc_df)，前端输入
    proposed_alloc = {
        'C1': {'货币现金类': 0.9, '同业存单': 0.1},
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
    assets_list.remove('同业存单')

    # 大类资产净值数据转为每日收益率数据
    hist_value_r = hist_value.pct_change().dropna()

    # 计算大类资产的年化收益率、年化波动率、夏普比率
    hist_perf = pd.concat(
        [
            np.log(hist_value.iloc[-1] / hist_value.iloc[0]) / (hist_value.count() - 1) * 252,
            np.std(np.log(hist_value / hist_value.shift(-1))[1:], axis=0) * np.sqrt(252)
        ],
        axis=1
    )
    hist_perf.columns = ['年化收益率', '年化波动率']
    hist_perf['夏普比率'] = hist_perf['年化收益率'] / hist_perf['年化波动率']
    hist_perf['var95'] = hist_perf['年化收益率'] - hist_perf['年化波动率'] * var95

    # 模拟权重
    w_r10 = generate_simplex_grid(n_assets=5, resolution=10)
    r10, best_r10 = find_best(w_r10)
    w_r20 = generate_simplex_grid_constraint(n_assets=5, resolution=20, cons=best_r10, threshold=0.1)
    r20, best_r20 = find_best(w_r20)
    w_r50 = generate_simplex_grid_constraint(n_assets=5, resolution=50, cons=best_r20, threshold=0.05)
    r50, best_r50 = find_best(w_r50)
    w_r100 = generate_simplex_grid_constraint(n_assets=5, resolution=100, cons=best_r50, threshold=0.02)
    r100, best_r100 = find_best(w_r100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, row in pd.concat([r100, r50, r20, r10]).drop_duplicates().iterrows():
        color_style = 'ro' if not row['on_ef'] else 'bo'
        ax.plot(row['vol_annual'], row['ret_annual'], color_style, markersize=1)
    plt.show()

    # 输出有效前沿点
    r100_df = generate_alloc_perf_batch(
        hist_value_r[['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']].values, best_r100)
    r100_df.columns = ['csh_mgt_typ_pos', 'fx_yld_pos', 'mix_strg_typ_pos', 'eqty_invst_typ_pos', 'altnt_invst_pos',
                       'rate', 'liquid', 'shrp_prprtn', 'var95']
    r100_df.sort_values(by='rate', axis=0, ascending=True, inplace=True)
    r100_df.index = range(11, 11 + len(r100_df))  # 11, 12, 13...

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=r100_df['liquid'],  # 波动率 (第二列)
        y=r100_df['rate'],  # 收益率 (第一列)
        s=10,
        c='red'
    )
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('收益率', rotation=270, labelpad=15)
    # 添加标题和标签
    plt.title('资产组合有效前沿', fontsize=14)
    plt.xlabel('波动率 (风险)', fontsize=12)
    plt.ylabel('收益率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    # 优化布局
    plt.tight_layout()
    plt.show()

    # 有效前沿点与6个点合并并输出
    hist_perf_ = hist_perf.loc[['C1', 'C2', 'C3', 'C4', 'C5', 'C6'], :]
    # 同业存单持仓占比合并到现金类
    proposed_alloc_df['货币现金类'] = proposed_alloc_df['货币现金类'] + proposed_alloc_df['同业存单']
    alloc_df = proposed_alloc_df.drop('同业存单', axis=1).join(hist_perf_, how='left')
    alloc_df.sort_index(inplace=True, ascending=True)
    alloc_df.reset_index(drop=True, inplace=True)
    alloc_df.index = alloc_df.index + 1
    alloc_df = alloc_df[
        ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类', '年化收益率', '年化波动率', '夏普比率',
         'var95']]
    alloc_df.columns = ['csh_mgt_typ_pos', 'fx_yld_pos', 'mix_strg_typ_pos', 'eqty_invst_typ_pos', 'altnt_invst_pos',
                        'rate', 'liquid', 'shrp_prprtn', 'var95']

    result = pd.concat([r100_df, alloc_df], axis=0)
    result = result.reset_index().rename(columns={'index': 'rsk_lvl'})
    result.to_csv("有效前沿点.csv", index=False, encoding='utf8')

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=r100_df['var95'],  # 波动率 (第二列)
        y=r100_df['rate'],  # 收益率 (第一列)
        s=10,
        c='red'
    )
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('收益率', rotation=270, labelpad=15)
    # 添加标题和标签
    plt.title('资产组合有效前沿', fontsize=14)
    plt.xlabel('var95', fontsize=12)
    plt.ylabel('rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    # 优化布局
    plt.tight_layout()
    plt.show()
    print('图1', '-' * 20)

    n = 1000
    vol_min = hist_perf.loc['货币现金类', '年化波动率']
    vol_max = hist_perf.loc['权益投资类', '年化波动率']
    alloc_value_r = hist_value_r[['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']]
    # 距离为1%
    best_vol_point = best_vol_point(alloc_value_r, vol_min, vol_max, n=1000)
    best_vol_point.reset_index(drop=True, inplace=True)
    raw_portfolio = {'货币现金类': 0.2, '固定收益类': 0.2, '混合策略类': 0.2, '权益投资类': 0.2, '另类投资类': 0.2}
    optimized_portfolio_df = deepcopy(best_vol_point)
    best_point = find_optimized_portfolio(raw_portfolio, optimized_portfolio_df)

    plt.figure(figsize=(20, 12))
    plt.scatter(optimized_portfolio_df['vol_annual'], optimized_portfolio_df['ret_annual'], c='blue', s=10)
    # plt.scatter(optimized_portfolio_df['vol_annual'], optimized_portfolio_df['ret_annual'], 'b-', linewidth=1.5, label='有效前沿', alpha=0.5)
    # 添加颜色条
    special_points = [
        (
            best_point['raw_portfolio']['vol_annual'], best_point['raw_portfolio']['ret_annual'], '持仓组合', 'black',
            'o'),
        (best_point['max_ret']['vol_annual'], best_point['max_ret']['ret_annual'], '高收益', 'black', 'o'),
        (best_point['min_vol']['vol_annual'], best_point['min_vol']['ret_annual'], '高收益', 'red', 'o'),
        (best_point['medial_vol']['vol_annual'], best_point['medial_vol']['ret_annual'], '中点', 'green', 'o'),
        # (best_point['nearest_dist']['vol_annual'], best_point['nearest_dist']['ret_annual'], '距离最近', 'black', 'o')
    ]
    # 当前持仓：        (0.05601378830467611, 0.05354807757307646)
    # 收益一定波动最小   (0.055909521993949655, 0.053771448254689914)
    # 中点             (0.05559642897659834, 0.053622155570070905)
    # 波动一定收益最大   (0.05529416938977679, 0.05356648565618989)
    # 距离最近         (0.055909521993949655, 0.053771448254689914)
    for var, ret, label, color, marker in special_points:
        plt.scatter(var, ret, c=color, marker=marker, s=10,
                    label=label)
    # 添加标题和标签
    plt.title('资产组合有效前沿', fontsize=14)
    plt.xlabel('vol_annual', fontsize=12)
    plt.ylabel('ret_annual', fontsize=12)
    # 优化布局
    plt.tight_layout()
    plt.show()

    # 距离为0.5%
    best_vol_point2 = best_vol_point2(alloc_value_r, vol_min, vol_max, n=1000)
    best_vol_point2.reset_index(drop=True, inplace=True)
    optimized_portfolio_df2 = deepcopy(best_vol_point2)
    best_point2 = find_optimized_portfolio(raw_portfolio, optimized_portfolio_df2)

    plt.figure(figsize=(20, 12))
    plt.scatter(optimized_portfolio_df['vol_annual'], optimized_portfolio_df['ret_annual'], c='blue', s=10)
    # plt.scatter(optimized_portfolio_df['vol_annual'], optimized_portfolio_df['ret_annual'], 'b-', linewidth=1.5, label='有效前沿', alpha=0.5)
    # 添加颜色条
    special_points = [
        (
            best_point2['raw_portfolio']['vol_annual'], best_point2['raw_portfolio']['ret_annual'], '持仓组合', 'black',
            'o'),
        (best_point2['max_ret']['vol_annual'], best_point2['max_ret']['ret_annual'], '高收益', 'black', 'o'),
        (best_point2['min_vol']['vol_annual'], best_point2['min_vol']['ret_annual'], '高收益', 'red', 'o'),
        (best_point2['medial_vol']['vol_annual'], best_point2['medial_vol']['ret_annual'], '中点', 'green', 'o'),
        (best_point['nearest_dist']['vol_annual'], best_point['nearest_dist']['ret_annual'], '距离最近', 'orange', 'o')
    ]
    # 当前持仓：        (0.05601378830467611, 0.05354807757307646)
    # 收益一定波动最小   (0.055909521993949655, 0.053771448254689914)
    # 中点             (0.05559642897659834, 0.053622155570070905)
    # 波动一定收益最大   (0.05529416938977679, 0.05356648565618989)
    # 距离最近         (0.055909521993949655, 0.053771448254689914)
    for var, ret, label, color, marker in special_points:
        plt.scatter(var, ret, c=color, marker=marker, s=10,
                    label=label)
    # 添加标题和标签
    plt.title('资产组合有效前沿', fontsize=14)
    plt.xlabel('vol_annual', fontsize=12)
    plt.ylabel('ret_annual', fontsize=12)
    # 优化布局
    plt.tight_layout()
    plt.show()

    optimized_portfolio_df2.to_csv('optimized_portfolio_df2.csv', encoding='utf8')
