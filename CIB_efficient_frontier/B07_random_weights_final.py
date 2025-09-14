# -*- encoding: utf-8 -*-
"""
@File: B07_random_weights_final.py
@Author: Kevin-Chen
@Descriptions:
  1) 用 QCQP 逐风险扫描，刻准有效前沿（带线性约束与多资产联合约束）
  2) 以前沿锚点为种子，小步随机游走 + POCS 投影，填充前沿之下的可行空间
  3) 支持生成样本的“权重精度”可选（0.1% / 0.2% / 0.5%），并去重
  4) 批量计算绩效，并作图
"""
import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any

''' 一、通用工具 & 画图 '''


# 绘制投资组合有效前沿图
def plot_efficient_frontier(
        scatter_points_data: List[Dict[str, Any]],
        title: str = '投资组合与有效前沿',
        x_axis_title: str = '年化波动率 (Annual Volatility)',
        y_axis_title: str = '年化收益率 (Annual Return)',
        x_col: str = 'vol_annual',
        y_col: str = 'ret_annual',
        hover_text_col: str = 'hover_text',
        output_filename: str = None
):
    """
    绘制投资组合有效前沿图

    该函数使用plotly绘制散点图，用于展示投资组合的有效前沿曲线，支持多组数据的可视化展示

    参数:
        scatter_points_data (List[Dict[str, Any]]): 散点图数据列表，每个元素为包含数据和样式信息的字典
            字典应包含以下键值:
            - "data": pandas DataFrame，包含实际的x、y坐标和悬停文本数据
            - "color": str，散点颜色
            - "size": int，散点大小
            - "opacity": float，散点透明度
            - "name": str，图例名称
            - "marker_line": dict，可选，散点边框样式
        title (str): 图表标题，默认为'投资组合与有效前沿'
        x_axis_title (str): x轴标题，默认为'年化波动率 (Annual Volatility)'
        y_axis_title (str): y轴标题，默认为'年化收益率 (Annual Return)'
        x_col (str): DataFrame中用作x轴数据的列名，默认为'vol_annual'
        y_col (str): DataFrame中用作y轴数据的列名，默认为'ret_annual'
        hover_text_col (str): DataFrame中用作悬停文本的列名，默认为'hover_text'
        output_filename (str, optional): 输出文件名。如果提供，图表将保存为HTML文件。默认为None，直接显示图表。

    返回值:
        无返回值，直接显示或保存图表
    """
    # 创建图表对象
    fig = go.Figure()

    # 遍历所有散点数据集，添加到图表中
    for point_set in scatter_points_data:
        df = point_set["data"]
        # 添加散点图轨迹
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            hovertext=df[hover_text_col],
            hoverinfo='text',
            mode='markers',
            marker=dict(
                color=point_set["color"],
                size=point_set["size"],
                opacity=point_set["opacity"],
                line=point_set.get("marker_line")
            ),
            name=point_set["name"]
        ))

    # 更新图表布局设置
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        legend_title="图例",
        hovermode='closest'
    )

    # 显示或保存图表
    if output_filename:
        fig.write_html(output_filename)
        print(f"图表已保存到: {output_filename}")
    else:
        fig.show()


# 分块计算年化收益和年化波动率
def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray,
                              chunk_size: int = 20000) -> pd.DataFrame:
    """
    分块计算避免 T×N 爆内存；对 log 计算做 clip 并清理 ±inf。

    参数:
        port_daily (np.ndarray): 形状为 [T, n]，表示 T 个交易日、n 个资产的日收益率。
        portfolio_allocs (np.ndarray): 形状为 [N, n]，表示 N 个投资组合在 n 个资产上的权重分配。
        chunk_size (int): 每次处理的投资组合数量，用于分块计算，默认为 20000。

    返回:
        pd.DataFrame: 包含每个投资组合的年化收益(ret_annual)和年化波动率(vol_annual)，以及对应的权重列 w_0, w_1, ..., w_{n-1}。
                      同时会移除包含无穷大值的行。
    """
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    T, n = port_daily.shape
    N = portfolio_allocs.shape[0]

    res_list = []
    # 分块处理 portfolio_allocs，避免一次性计算造成内存溢出
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        W = portfolio_allocs[s:e]  # [m, n]
        R = port_daily @ W.T  # [T, m]

        # 计算累积收益并限制下界防止 log(0)
        one_plus_R = np.clip(1.0 + R, 1e-12, None)
        port_cum = np.cumprod(one_plus_R, axis=0)  # [T, m]
        final_ret = port_cum[-1, :]

        # 年化对数总收益
        log_total = np.log(np.clip(final_ret, 1e-12, None))
        ret_annual = (log_total / T) * 252.0

        # 日对数收益及年化波动率
        log_daily = np.log(one_plus_R)
        vol_annual = np.std(log_daily, axis=0, ddof=1) * np.sqrt(252.0)

        # 构造当前批次结果 DataFrame
        df = pd.DataFrame({
            "ret_annual": ret_annual,
            "vol_annual": vol_annual,
        })
        wdf = pd.DataFrame(W, columns=[f"w_{i}" for i in range(n)])
        res_list.append(pd.concat([wdf, df], axis=1))

    # 合并所有批次结果，并清理无穷大值
    out = pd.concat(res_list, axis=0, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


# 从散点中识别有效前沿（按收益降序，保留波动率的前缀最小值）
def cal_ef2_v4_ultra_fast(data: pd.DataFrame) -> pd.DataFrame:
    """
    从散点中识别有效前沿（按收益降序，保留波动率的前缀最小值）

    参数:
        data (pd.DataFrame): 包含投资组合收益和波动率数据的DataFrame，必须包含'ret_annual'和'vol_annual'列

    返回:
        pd.DataFrame: 原始数据添加'on_ef'列后的DataFrame，'on_ef'列为布尔值，标识对应数据点是否在有效前沿上
    """
    data = data.copy()
    ret_values = data['ret_annual'].values
    vol_values = data['vol_annual'].values

    # 按收益降序排列，获取排序索引
    sorted_idx = np.argsort(ret_values)[::-1]
    sorted_vol = vol_values[sorted_idx]

    # 计算排序后波动率的累积最小值，用于识别有效前沿
    cummin_vol = np.minimum.accumulate(sorted_vol)
    on_ef_sorted = (sorted_vol <= cummin_vol + 1e-6)

    # 将有效前沿标识映射回原始数据顺序
    on_ef = np.zeros(len(data), dtype=bool)
    on_ef[sorted_idx] = on_ef_sorted
    data['on_ef'] = on_ef
    return data


''' 二、POCS 投影 (盒约束 ∩ 和=1 ∩ 多资产组约束) '''


# 使用 POCS 方法将向量投影到多个约束集合的交集上。
def project_to_constraints_pocs(v: np.ndarray,
                                single_limits,  # list[(low, high)]
                                multi_limits: dict,  # {(tuple_idx): (low, high)}
                                max_iter=200, tol=1e-9, damping=0.9):
    """
    使用 POCS（Projection Onto Convex Sets）方法将向量投影到多个约束集合的交集上。

    该函数处理以下三类约束：
    1. 每个变量的上下界约束（盒约束）；
    2. 所有变量之和等于 1；
    3. 某些指定索引组的和在给定范围内（组约束）。

    参数:
        v (np.ndarray): 待投影的输入向量。
        single_limits (list of tuple): 每个元素为 (low, high)，表示对应位置变量的上下界。
        multi_limits (dict): 键为索引元组，值为 (low, high)，表示对应索引组的和的范围。
        max_iter (int): 最大迭代次数，默认为 200。
        tol (float): 收敛判断的容差，默认为 1e-9。
        damping (float): 组约束投影时的阻尼系数，用于提高稳定性，默认为 0.9。

    返回:
        np.ndarray or None: 若成功投影则返回满足所有约束的向量；否则返回 None。
    """
    x = v.astype(np.float64).copy()
    n = x.size

    # 提取每个维度的上下界
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    # 预处理 multi_limits，计算每组的长度平方（用于投影公式）
    groups = []
    for idx_tuple, (low, up) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=np.int64)
        a2 = float(len(idx))  # ||a||^2
        groups.append((idx, float(low), float(up), a2))

    # 初始粗投影：先满足盒约束和和=1约束
    x = np.clip(x, lows, highs)
    x += (1.0 - x.sum()) / n

    # 迭代进行交替投影
    for _ in range(max_iter):
        x_prev = x

        # 投影到盒约束集合
        x = np.clip(x, lows, highs)
        # 投影到和=1的集合
        x += (1.0 - x.sum()) / n
        # 投影到各组的上下半空间约束
        for idx, low, up, a2 in groups:
            if a2 == 0:
                continue
            s = x[idx].sum()
            if s > up + 1e-12:
                x[idx] -= damping * (s - up) / a2
            elif s < low - 1e-12:
                x[idx] += damping * (low - s) / a2
        # 再次修正和=1约束（组投影可能破坏总和）
        x += (1.0 - x.sum()) / n

        # 判断是否收敛
        if np.max(np.abs(x - x_prev)) < tol:
            break

    # 末端校验：确保所有约束都满足
    if (x < lows - 1e-6).any() or (x > highs + 1e-6).any():
        return None
    for idx, low, up, _ in groups:
        if len(idx) == 0:
            continue
        s = x[idx].sum()
        if s < low - 1e-6 or s > up + 1e-6:
            return None
    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        return None
    return x


''' 三、统计量与前沿刻画 (QCQP 逐风险扫描) '''


# 计算年化收益率均值和年化协方差矩阵
def ann_mu_sigma(log_returns: np.ndarray):
    """
    计算年化收益率均值和年化协方差矩阵

    参数:
        log_returns (np.ndarray): 对数收益率矩阵，形状为[T, n]，其中T为时间步数，n为资产数量

    返回:
        tuple: 包含两个元素的元组
            - mu (np.ndarray): 年化收益率均值向量，形状为[n,]
            - Sigma (np.ndarray): 年化收益率协方差矩阵，形状为[n, n]

    注: 假设一年有252个交易日
    """
    # 计算年化收益率均值（乘以年交易日数252）
    mu = log_returns.mean(axis=0) * 252.0

    # 计算年化协方差矩阵（使用样本协方差，自由度修正为1，乘以年交易日数252）
    Sigma = np.cov(log_returns, rowvar=False, ddof=1) * 252.0

    return mu, Sigma


# 计算投资组合的收益率和波动率统计量
def port_stats(W: np.ndarray, mu: np.ndarray, Sigma: np.ndarray):
    """
    计算投资组合的收益率和波动率统计量

    参数:
        W: np.ndarray, 形状为 [m, n] 或 [n,]，投资组合权重矩阵或向量
           m为投资组合数量，n为资产数量
        mu: np.ndarray, 形状为 [n,]，资产预期收益率向量
        Sigma: np.ndarray, 形状为 [n, n]，资产收益率协方差矩阵

    返回:
        tuple: (rets, vols)
            rets: np.ndarray, 形状为 [m,] 或 [1,]，投资组合收益率
            vols: np.ndarray, 形状为 [m,] 或 [1,]，投资组合波动率（标准差）
    """
    # 处理单个投资组合权重向量的情况
    if W.ndim == 1:
        ret = float(W @ mu)
        vol = float(np.sqrt(W @ Sigma @ W))
        return np.array([ret]), np.array([vol])

    # 处理多个投资组合权重矩阵的情况
    # 计算所有投资组合的收益率
    rets = W @ mu

    # 使用爱因斯坦求和约定计算所有投资组合的波动率
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, Sigma, W))
    return rets, vols


# 求解最小方差投资组合优化问题
def solve_min_variance(Sigma, single_limits, multi_limits):
    """
    求解最小方差投资组合优化问题

    该函数通过凸优化方法求解带约束条件的最小方差投资组合问题。
    目标是最小化投资组合的方差，约束条件包括权重和为1、单个资产权重限制和多个资产组合权重限制。

    参数:
        Sigma: numpy.ndarray, shape (n, n)
            资产收益率的协方差矩阵
        single_limits: list of tuples
            单个资产权重的上下限约束，每个元素为 (lower_bound, upper_bound) 形式的元组
        multi_limits: dict
            多个资产组合的权重约束，键为资产索引的集合，值为 (lower_bound, upper_bound) 形式的元组

    返回值:
        numpy.ndarray
            最优投资组合权重向量
    """
    # 获取资产数量
    n = Sigma.shape[0]

    # 定义优化变量：投资组合权重
    w = cp.Variable(n)

    # 构建约束条件列表; 添加权重和为1的约束
    cons = [cp.sum(w) == 1]

    # 添加单个资产权重的上下限约束
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]

    # 添加多个资产组合的权重约束
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]

    # 构建并求解优化问题; 目标函数：最小化二次型 w^T * Sigma * w（即投资组合方差）
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), cons)
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=1e-8, reltol=1e-8, feastol=1e-8)

    # 返回最优权重
    return w.value


# 求解最大收益投资组合优化问题
def solve_max_return(mu, single_limits, multi_limits):
    """
    求解最大收益投资组合优化问题

    该函数通过凸优化方法求解在给定约束条件下的最大收益投资组合分配问题。
    目标是在满足单个资产权重限制和多个资产组合权重限制的条件下，
    最大化投资组合的预期收益。

    参数:
        mu: numpy数组，长度为n，表示n个资产的预期收益率
        single_limits: 列表，长度为n，每个元素为(下限, 上限)元组，
                      表示对应资产权重的约束范围
        multi_limits: 字典，键为资产索引的可迭代对象，值为(下限, 上限)元组，
                     表示指定资产组合的权重约束范围

    返回值:
        numpy数组，长度为n，表示最优解下的各资产权重分配
    """
    n = mu.size
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]

    # 添加单个资产的权重约束
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]

    # 添加多个资产组合的权重约束
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]

    # 构建并求解优化问题
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=1e-8, reltol=1e-8, feastol=1e-8)
    return w.value


# 求解在给定风险约束下的最大收益投资组合优化问题
def solve_max_return_at_risk(mu, Sigma, s_target, single_limits, multi_limits, w0=None):
    """
    求解在给定风险约束下的最大收益投资组合优化问题

    该函数通过凸优化求解马科维茨均值-方差优化问题，在满足风险约束和权重限制条件下，
    寻找能够最大化期望收益的投资组合权重。

    参数:
        mu: numpy数组，长度为n，表示n个资产的期望收益率
        Sigma: numpy数组，形状为(n,n)，表示n个资产的协方差矩阵
        s_target: float，目标风险水平(标准差)的上限值
        single_limits: list of tuples，长度为n，每个元素(lo, hi)表示对应资产权重的上下界约束
        multi_limits: dict，键为资产索引的可迭代对象，值为(lo, up)元组，表示一组资产权重和的约束
        w0: numpy数组，可选，长度为n，优化变量的初始值，用于热启动加速求解

    返回值:
        numpy数组，长度为n，表示优化得到的各资产最优权重
    """
    n = mu.size
    w = cp.Variable(n)

    # 构建约束条件：权重和为1，投资组合方差不超过目标风险的平方
    cons = [cp.sum(w) == 1, cp.quad_form(w, Sigma) <= float(s_target ** 2)]

    # 添加单个资产的权重边界约束
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]

    # 添加多个资产组合的权重和约束
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]

    # 构建并求解优化问题：最大化期望收益
    prob = cp.Problem(cp.Maximize(mu @ w), cons)

    # 如果提供了初始值，则设置变量的初始值用于热启动
    if w0 is not None:
        try:
            w.value = w0
        except Exception:
            pass

    # 求解优化问题
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=5e-8, reltol=5e-8, feastol=5e-8, max_iters=1000)

    return w.value


# 扫描风险网格 [σ_min, σ_max]，每个 σ 上最大化收益
def sweep_frontier_by_risk(mu, Sigma, single_limits, multi_limits, n_grid=1200):
    """
    扫描风险网格 [σ_min, σ_max]，每个 σ 上最大化收益

    参数:
        mu: 收益率向量
        Sigma: 协方差矩阵
        single_limits: 单一资产限制条件
        multi_limits: 多资产限制条件
        n_grid: 风险网格点数，默认为1200

    返回:
        grid: 风险网格点数组
        W: 每个风险水平下的最优权重矩阵
        R: 对应的收益率数组
        S: 对应的风险(标准差)数组
        w_minv: 最小方差组合权重
        w_maxr: 最大收益组合权重
    """
    # 计算边界组合：最小方差组合和最大收益组合
    w_minv = solve_min_variance(Sigma, single_limits, multi_limits)
    w_maxr = solve_max_return(mu, single_limits, multi_limits)
    _, s_min = port_stats(w_minv, mu, Sigma);
    s_min = s_min[0]
    _, s_max = port_stats(w_maxr, mu, Sigma);
    s_max = float(max(s_min, s_max))

    # 构建风险扫描网格
    grid = np.linspace(s_min, s_max, n_grid)

    # 在每个风险水平上求解最大收益组合
    W = []
    w0 = w_minv
    for s in grid:
        w = solve_max_return_at_risk(mu, Sigma, s, single_limits, multi_limits, w0=w0)
        W.append(w)
        w0 = w
    W = np.asarray(W)
    R, S = port_stats(W, mu, Sigma)
    return grid, W, R, S, w_minv, w_maxr


# 构造一个上包络函数，用于计算给定σ值对应的 R_upper 值
def make_upper_envelope_fn(R: np.ndarray, S: np.ndarray):
    """
    构造一个上包络函数，用于计算给定σ值对应的 R_upper 值

    该函数通过对(σ, μ)数据点进行分段线性插值来构建上包络函数R_upper(σ)。
    首先将输入数据按照S值进行排序，然后使用线性插值方法构建插值函数。

    参数:
        R (np.ndarray): μ值数组，表示函数值
        S (np.ndarray): σ值数组，表示自变量值

    返回:
        function: 插值函数f(sig)，接受σ值作为输入，返回对应的R_upper值
    """
    # 按照S值对数据进行排序，确保插值的正确性
    order = np.argsort(S)
    S_sorted = S[order]
    R_sorted = R[order]

    def f(sig):
        """
        上包络插值函数

        参数:
            sig: σ值，可以是标量或数组

        返回:
            对应的R_upper值，数据类型与输入sig保持一致
        """
        sig = np.atleast_1d(sig)
        # 使用线性插值计算对应σ值的R_upper值
        # 对于超出范围的值，使用边界值进行外推
        return np.interp(sig, S_sorted, R_sorted, left=R_sorted[0], right=R_sorted[-1])

    return f


''' 四、量化到指定精度 & 去重 '''


# 解析精度值字符串，将其转换为浮点数
def _parse_precision(choice: str) -> float:
    """
    解析精度值字符串，将其转换为浮点数

    参数:
        choice (str): 精度值字符串，支持百分比格式(如'0.1%')或直接的浮点数格式(如'0.001')

    返回:
        float: 转换后的精度值浮点数

    示例:
        '0.1%' -> 0.001
        '0.2%' -> 0.002
        '0.5%' -> 0.005
        '0.001' -> 0.001
    """
    choice = str(choice).strip()
    # 处理百分比格式的精度值
    if choice.endswith('%'):
        val = float(choice[:-1]) / 100.0
    else:
        val = float(choice)
    # 规范到合理小数
    return float(val)


# 使用最大余数法将权重向量 w 投影到指定步长的网格上，同时满足: 权重和为1的约束以及单资产盒约束
def _snap_to_grid_simplex(w: np.ndarray, step: float, single_limits) -> np.ndarray | None:
    """
    使用最大余数法将权重向量 w 投影到指定步长的网格上，同时满足以下约束：
    - 权重和为 1；
    - 每个资产的权重满足对应的上下界（单资产盒约束）。

    注意：该函数不处理组约束，组约束应由外层的 POCS 量化循环进行修复。

    参数：
        w (np.ndarray): 原始权重向量，形状为 (n,)。
        step (float): 网格步长，用于量化权重。
        single_limits (list of tuple): 每个资产的权重上下界，格式为 [(low_1, high_1), ..., (low_n, high_n)]。

    返回：
        np.ndarray | None: 若成功投影，则返回量化后的权重向量；否则返回 None。
    """
    R = int(round(1.0 / step))  # 分辨率（总单位数）
    w = np.clip(w, 0.0, 1.0)  # 安全裁剪，防止数值越界

    # 将权重映射到格子空间
    k_float = w / step
    k_floor = np.floor(k_float).astype(np.int64)
    frac = k_float - k_floor  # 小数部分，用于最大余数法分配

    # 提取并转换单资产上下界为格子单位
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)
    lo_units = np.ceil(lows / step - 1e-12).astype(np.int64)
    hi_units = np.floor(highs / step + 1e-12).astype(np.int64)

    # 初始整数解，裁剪到合法范围
    k = np.clip(k_floor, lo_units, hi_units)
    diff = R - int(k.sum())  # 当前总和与目标总单位数的差值

    if diff > 0:
        # 需要增加 diff 个单位，优先分配给小数部分大的元素
        cap = hi_units - k  # 每个位置还能增加的单位数
        idx = np.argsort(-frac)  # 按小数部分从大到小排序
        for i in idx:
            if diff == 0:
                break
            add = min(cap[i], diff)
            if add > 0:
                k[i] += add
                diff -= add
        if diff != 0:
            return None  # 容量不足，无法满足和为1
    elif diff < 0:
        # 需要减少 -diff 个单位，优先从 frac 小的元素中扣除
        cap = k - lo_units  # 每个位置还能减少的单位数
        idx = np.argsort(frac)  # 按小数部分从小到大排序
        for i in idx:
            if diff == 0:
                break
            sub = min(cap[i], -diff)
            if sub > 0:
                k[i] -= sub
                diff += sub
        if diff != 0:
            return None  # 无法满足和为1

    # 将整数解转换回权重空间
    wq = k.astype(np.float64) / R

    # 再次检查是否满足单资产边界和总和约束（数值安全）
    if (wq < lows - 1e-12).any() or (wq > highs + 1e-12).any():
        return None
    if not np.isclose(wq.sum(), 1.0, atol=1e-12):
        return None

    return wq


# 使用投影方法对权重进行量化，结合POCS约束优化和网格量化两个步骤进行迭代优化
def quantize_with_projection(w: np.ndarray, step: float,
                             single_limits, multi_limits,
                             rounds: int = 5) -> np.ndarray | None:
    """
    使用投影方法对权重进行量化，结合POCS约束优化和网格量化两个步骤进行迭代优化。

    该函数通过循环执行POCS（凸集投影）约束优化和网格量化操作，最多执行指定轮数，
    直到找到满足约束条件的量化结果或达到最大迭代次数。

    参数:
        w: 输入的权重数组，需要进行量化的原始数据
        step: 量化的步长，决定量化精度
        single_limits: 单个元素的约束范围限制
        multi_limits: 多个元素间的联合约束限制
        rounds: 最大迭代轮数，默认为5轮

    返回值:
        np.ndarray | None: 量化后的权重数组，如果量化失败则返回None
    """
    x = w.copy()
    for _ in range(rounds):
        # 先在连续域修组约束
        x = project_to_constraints_pocs(x, single_limits, multi_limits,
                                        max_iter=300, tol=1e-10, damping=0.9)
        if x is None:
            return None
        # 再吸附到网格
        xq = _snap_to_grid_simplex(x, step, single_limits)
        if xq is None:
            return None
        # 如果量化后几乎不变，则结束
        if np.max(np.abs(xq - x)) < step * 0.5:
            return xq
        x = xq
    # 最后一次校验
    return x


# 对数据点进行去重处理
def dedup_by_grid(W: np.ndarray, step: float) -> np.ndarray:
    """
    通过网格化方法对数据点进行去重处理

    该函数将输入的点集映射到规则网格上，对落在同一网格单元内的点只保留一个代表点，
    从而实现数据点的去重和简化。

    参数:
        W (np.ndarray): 输入的点集数组，形状为(n, d)，其中n为点的数量，d为维度
        step (float): 网格步长，用于确定网格单元的大小

    返回:
        np.ndarray: 去重后的点集数组，保持原有的数据类型和形状特性
    """
    # 处理空数组的边界情况
    if W.size == 0:
        return W

    # 将点坐标映射到网格索引空间
    K = np.rint(W / step).astype(np.int64)

    # 找到唯一的网格索引并获取对应原始点的索引
    _, idx = np.unique(K, axis=0, return_index=True)

    # 按照索引顺序返回去重后的点集
    return W[np.sort(idx)]


''' 五、以前沿锚点为种子：小步随机游走 + POCS（可选精度）填厚前沿之下 '''


# 对前沿锚点进行随机游走采样，生成位于前沿下方的随机组合权重
def random_walk_below_frontier(W_anchor: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                               single_limits, multi_limits,
                               per_anchor: int = 30, step: float = 0.01,
                               sigma_tol: float = 1e-4, seed: int = 123,
                               precision: str | float | None = None):
    """
    对前沿锚点进行随机游走采样，生成位于前沿下方的 portfolio 权重。

    该函数对每个前沿锚点 w0 进行多次零和小步扰动，并将扰动后的权重投影到约束空间中。
    若扰动后的新权重仍位于前沿曲线下方（考虑容差），则保留该样本。
    如果指定了精度参数，则会对样本进行量化并去重。

    参数:
        W_anchor (np.ndarray): 形状为 (n, m) 的前沿锚点权重矩阵，每一行是一个 portfolio 权重向量。
        mu (np.ndarray): 长度为 m 的资产预期收益向量。
        Sigma (np.ndarray): 形状为 (m, m) 的资产协方差矩阵。
        single_limits: 单一资产约束条件，用于投影操作。
        multi_limits: 多资产联合约束条件，用于投影操作。
        per_anchor (int): 对每个锚点进行扰动的次数，默认为 30。
        step (float): 扰动的标准差，默认为 0.01。
        sigma_tol (float): 判断是否在前沿下方时使用的风险容差，默认为 1e-4。
        seed (int): 随机种子，默认为 123。
        precision (str | float | None): 网格精度参数，若不为 None，则对结果进行量化与去重。

    返回:
        np.ndarray: 形状为 (k, m) 的采样权重矩阵，其中 k 是满足条件的样本数。
    """
    rng = np.random.default_rng(seed)
    R_anchor, S_anchor = port_stats(W_anchor, mu, Sigma)
    f_upper = make_upper_envelope_fn(R_anchor, S_anchor)

    step_grid = None
    if precision is not None:
        step_grid = _parse_precision(precision)

    collected = []

    # 对每个锚点执行随机游走采样
    for w0 in W_anchor:
        _, s0 = port_stats(w0, mu, Sigma)
        s0 = s0[0]
        s_bar = s0 + sigma_tol
        for _ in range(per_anchor):
            # 生成零和扰动
            eps = rng.normal(0.0, step, size=w0.size)
            eps -= eps.mean()  # 零和扰动
            w_try = project_to_constraints_pocs(w0 + eps, single_limits, multi_limits,
                                                max_iter=200, tol=1e-9, damping=0.9)
            if w_try is None:
                continue

            # 如果需要“精度”，先量化（包含一次 POCS-量化循环保证组约束）
            if step_grid is not None:
                w_try = quantize_with_projection(w_try, step_grid, single_limits, multi_limits, rounds=5)
                if w_try is None:
                    continue

            # 检查扰动后点是否在前沿下方
            r, s = port_stats(w_try, mu, Sigma)
            if (s[0] <= s_bar + 1e-12) and (r[0] <= f_upper(s)[0] + 1e-8):
                collected.append(w_try)

    W = np.array(collected) if collected else np.empty((0, W_anchor.shape[1]))
    # 去重（仅当有精度时按网格整数去重）
    if step_grid is not None and W.size:
        W = dedup_by_grid(W, step_grid)
    return W


# 生成服从指定约束的随机权重
def generate_constrained_portfolios(num_points: int, single_limits, multi_limits):
    """
    在给定约束下，生成指定数量的投资组合权重。

    参数:
        num_points (int): 需要生成的投资组合数量
        single_limits: 单个资产的权重限制，格式为[(low1, high1), (low2, high2), ...]
        multi_limits: 多资产联合约束条件

    返回:
        numpy.ndarray: 生成的投资组合权重矩阵，每行代表一个投资组合
    """
    generated_weights = []
    # 提取单个资产权重的上下限
    lows = np.array([l for l, h in single_limits])
    highs = np.array([h for l, h in single_limits])

    # 循环生成满足约束条件的投资组合权重
    while len(generated_weights) < num_points:
        # 在单资产限制范围内随机生成权重提案
        proposal = np.random.uniform(lows, highs)
        # 将提案权重投影到所有约束条件下
        adjusted_weights = project_to_constraints_pocs(proposal, single_limits, multi_limits)
        # 如果投影后的权重有效，则添加到结果中
        if adjusted_weights is not None:
            generated_weights.append(adjusted_weights)

    return np.array(generated_weights)


if __name__ == '__main__':
    ''' --- 1) 数据加载与预处理 --- '''
    hist_value = pd.read_excel('历史净值数据.xlsx', sheet_name='历史净值数据')
    # hist_value = pd.read_excel('benchmark_index.xlsx')
    hist_value = hist_value.set_index('date')
    hist_value.index = pd.to_datetime(hist_value.index)
    hist_value = hist_value.dropna().sort_index(ascending=True)
    hist_value = hist_value.rename({
        "货基指数": "货币现金类", '固收类': '固定收益类', '混合类': '混合策略类',
        '权益类': '权益投资类', '另类': '另类投资类', '安逸型': 'C1',
        '谨慎型': 'C2', '稳健型': 'C3', '增长型': 'C4',
        '进取型': 'C5', '激进型': 'C6'
    }, axis=1)

    assets_list = ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']
    hist_value_r = hist_value[assets_list].pct_change().dropna()
    port_daily_returns = hist_value_r.values

    # 用 log 收益率估计 μ, Σ（和风险最优化保持一致）
    log_r = np.log1p(port_daily_returns)
    mu, Sigma = ann_mu_sigma(log_r)

    ''' --- 2) 约束定义 (可按需修改/添加组约束) --- '''
    n_assets = len(assets_list)
    single_limits = [(0.0, 1.0)] * n_assets
    multi_limits = {
        # (0,1,2): (0.3, 1.0),   # 示例：前三类合计至少 30%
        # (3,4):   (0.0, 0.7),   # 示例：权益+另类合计不超过 70%
    }

    ''' --- 3) 扫描风险网格，刻准有效前沿 (锚点) --- '''
    print("开始刻画有效前沿（QCQP 逐风险扫描）...")
    risk_grid, W_frontier, R_frontier, S_frontier, w_minv, w_maxr = sweep_frontier_by_risk(
        mu, Sigma, single_limits, multi_limits, n_grid=300
    )
    # 去支配，得到真正的有效锚点（按 σ 升序保持收益前缀上包络）
    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    cummax_R = np.maximum.accumulate(R_sorted)
    keep = np.isclose(R_sorted, cummax_R, atol=1e-10)
    W_anchors = W_sorted[keep]
    R_anchors, S_anchors = R_sorted[keep], S_sorted[keep]
    print(f"有效前沿锚点数量: {len(W_anchors)}")

    ''' --- 4) 以前沿锚点为种子：小步随机游走 + POCS (可选精度) 填厚前沿之下区域 --- '''
    # 选择精度：'0.1%' / '0.2%' / '0.5%' / None
    precision_choice = None  # <- 你可以改成 '0.2%'、'0.5%' 或 None（表示不量化）
    print(f"开始填充前沿之下的可行空间（precision={precision_choice}) ...")

    W_below = random_walk_below_frontier(
        W_anchor=W_anchors, mu=mu, Sigma=Sigma,
        single_limits=single_limits, multi_limits=multi_limits,
        per_anchor=100, step=0.12, sigma_tol=1e-4, seed=123,
        precision=precision_choice
    )
    print(f"填充样本数量（量化&去重后）: {len(W_below)}")

    # 合并（把锚点也纳入样本，便于一起评估和可视化）
    W_all = np.vstack([W_anchors, W_below]) if len(W_below) else W_anchors

    ''' --- 5) 批量计算收益与风险 (用于画图与再次识别前沿) --- '''
    print("批量计算绩效指标...")
    perf_df = generate_alloc_perf_batch(port_daily_returns, W_all)

    # 标记“是否为锚点”
    anchor_perf = generate_alloc_perf_batch(port_daily_returns, W_anchors)
    anchor_perf['is_anchor'] = True

    perf_df['is_anchor'] = False
    # 合并后再以 is_anchor 标色
    full_df = pd.concat([perf_df, anchor_perf], ignore_index=True).drop_duplicates()

    # 标记有效前沿（可选：对全样本做一次快速识别）
    full_df = cal_ef2_v4_ultra_fast(full_df)

    ''' --- 6) 准备悬停文本并作图 --- '''
    weight_cols = {f"w_{i}": assets_list[i] for i in range(n_assets)}
    full_df = full_df.rename(columns=weight_cols)


    def create_hover_text(row):
        s = f"年化收益率: {row['ret_annual']:.2%}<br>年化波动率: {row['vol_annual']:.2%}<br><br><b>权重</b>:<br>"
        for asset in assets_list:
            if asset in row and row[asset] > 1e-4:
                s += f"  {asset}: {row[asset]:.1%}<br>"
        s += f"<br>锚点: {'是' if row.get('is_anchor', False) else '否'}"
        return s


    full_df['hover_text'] = full_df.apply(create_hover_text, axis=1)

    # 分层：前沿锚点、前沿点（自动识别）、填充样本
    df_anchor = full_df[full_df['is_anchor'] == True]
    df_ef = full_df[(full_df['on_ef'] == True) & (full_df['is_anchor'] == False)]
    df_fill = full_df[(full_df['on_ef'] == False) & (full_df['is_anchor'] == False)]

    scatter_data = [
        {"data": df_fill, "name": "前沿之下填充样本", "color": "lightblue", "size": 3, "opacity": 0.8},
        {"data": df_ef, "name": "识别出的有效前沿", "color": "deepskyblue", "size": 3, "opacity": 0.8},
        {"data": df_anchor, "name": "前沿锚点", "color": "crimson", "size": 5, "opacity": 0.8,
         "marker_line": dict(width=1, color='black')},
    ]

    ''' --- 7)  C1~C6 刻画 --- '''
    proposed_alloc = {
        'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }
    weight_cols_map = {f'w_{j}': assets_list[j] for j in range(len(assets_list))}

    print("\n--- 正在为每个风险等级生成可配置空间 ---")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    deviation = 0.2
    num_points_per_level = 10000
    base_points_to_plot = []

    for i, (risk_level, base_alloc_map) in enumerate(proposed_alloc.items()):
        print(f"--- 正在处理: {risk_level} ---")
        base_weights = np.array([base_alloc_map.get(asset, 0.0) for asset in assets_list])
        specific_single_limits = [(max(0.0, w * (1 - deviation)), min(1.0, w * (1 + deviation))) for w in base_weights]

        risk_level_weights = generate_constrained_portfolios(num_points_per_level, specific_single_limits, {})

        if len(risk_level_weights) > 0:
            perf_df = generate_alloc_perf_batch(port_daily_returns, risk_level_weights)
            perf_df = perf_df.rename(columns=weight_cols_map)
            perf_df['hover_text'] = perf_df.apply(lambda row: create_hover_text(row), axis=1)
            scatter_data.append({
                "data": perf_df, "name": f"{risk_level} 可配置空间", "color": colors[i % len(colors)],
                "size": 2, "opacity": 0.5
            })

        # 准备中心基准点
        base_perf_df = generate_alloc_perf_batch(port_daily_returns, base_weights.reshape(1, -1))
        base_perf_df = base_perf_df.rename(columns=weight_cols_map)
        base_perf_df['hover_text'] = base_perf_df.apply(lambda row: create_hover_text(row), axis=1)
        scatter_data.append({
            "data": base_perf_df, "name": f"{risk_level} 基准点", "color": colors[i % len(colors)],
            "size": 3, "opacity": 1.0, "symbol": "star", "marker_line": dict(width=1.5, color='black')
        })

    ''' --- 8) 图像画点 --- '''
    plot_efficient_frontier(
        scatter_points_data=scatter_data,
        title=f"QCQP 标准有效前沿 + 小步随机游走填充（precision={precision_choice}）",
        # output_filename="定死上下限20%区间.html"
    )
