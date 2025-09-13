# -*- encoding: utf-8 -*-
"""
@File: B01_random_weights.py
@Modify Time: 2025/9/9 20:23
@Author: Kevin-Chen
@Descriptions: 结合了（1）通用随机组合与有效前沿 和（2）基于建议配置生成各风险等级可配置空间的功能
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Any, Dict, Tuple, Iterable


# 绘图函数 (模块化)
def plot_efficient_frontier(
        scatter_points_data: List[Dict[str, Any]],
        title: str = '投资组合与有效前沿',
        x_axis_title: str = '年化波动率 (Annual Volatility)',
        y_axis_title: str = '年化收益率 (Annual Return)',
        x_col: str = 'vol_annual',
        y_col: str = 'ret_annual',
        hover_text_col: str = 'hover_text'
):
    """
    使用 Plotly 绘制可定制的有效前沿和投资组合散点图。
    """
    fig = go.Figure()

    for point_set in scatter_points_data:
        df = point_set["data"]
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
                line=point_set.get("marker_line")  # 安全地获取可选的边界线样式
            ),
            name=point_set["name"]
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        legend_title="图例",
        hovermode='closest'
    )
    fig.show()


# 指标计算函数
def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray) -> pd.DataFrame:
    """
    批量计算多个资产组合的性能指标，使用向量化操作以提高效率。
    """
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    # 步骤1: 所有组合的日收益率 (矩阵乘法)
    port_return_daily = port_daily @ portfolio_allocs.T

    # 步骤2: 年化收益率 (基于对数总收益)
    # 添加一个极小值防止log(0)
    port_cum_returns = np.cumprod(1 + port_return_daily, axis=0)
    final_returns = port_cum_returns[-1, :]
    log_total_ret = np.log(final_returns, where=(final_returns > 0), out=np.full_like(final_returns, -np.inf))
    port_ret_annual = log_total_ret / (port_return_daily.shape[0]) * 252

    # 步骤3: 年化波动率 (基于对数日收益率)
    log_returns = np.log(1 + port_return_daily)
    port_vol_annual = np.std(log_returns, axis=0, ddof=1) * np.sqrt(252)

    # 步骤4: 打包成DataFrame
    ret_df = pd.DataFrame({
        "ret_annual": port_ret_annual,
        "vol_annual": port_vol_annual,
    })

    # 步骤5: 合并权重数据
    weight_df = pd.DataFrame(portfolio_allocs, columns=[f'w_{i}' for i in range(portfolio_allocs.shape[1])])

    return pd.concat([weight_df, ret_df], axis=1).dropna()


def generate_alloc_perf_batch_fast(
        port_daily: np.ndarray,  # shape: (T, N)  资产日收益率
        portfolio_allocs: np.ndarray,  # shape: (M, N)  组合权重
        trading_days: float = 252.0,
        ddof: int = 1,
) -> pd.DataFrame:
    """
    批量计算多个资产组合的性能指标（年化对数收益、年化波动），纯向量化 & 省内存版本。
    - 组合日收益: R = port_daily @ portfolio_allocs.T   (一次BLAS)
    - 年化收益:   ret_annual = (sum_t log1p(R_t))/T * 252
    - 年化波动:   vol_annual = std(log1p(R_t)) * sqrt(252)

    说明：
    1) 与“先累乘再取log”的做法等价，但直接按列 sum(log1p) 更快更稳。
    2) 为了避免额外大矩阵，log1p 在 R 上原地执行（内存复用）。
    3) 若某组合存在 -100% 日收益（1+r=0），对应列会出现 -inf，这里用 isfinite 过滤掉。
    """
    # --- 形状与 dtype 检查 ---
    assert port_daily.ndim == 2 and portfolio_allocs.ndim == 2, "输入必须是二维数组"
    T, N = port_daily.shape
    M, Nw = portfolio_allocs.shape
    assert N == Nw, "资产数不一致"
    # 强制为 float64（BLAS/归约在 f64 上通常更稳更快）
    if port_daily.dtype != np.float64:
        port_daily = port_daily.astype(np.float64, copy=False)
    if portfolio_allocs.dtype != np.float64:
        portfolio_allocs = portfolio_allocs.astype(np.float64, copy=False)

    # --- 1) 组合日收益：一次 GEMM（最重的一步）---
    # R shape: (T, M)
    R = port_daily @ portfolio_allocs.T

    # --- 2) 在 R 上“原地”转成对数收益：R ← log1p(R) ---
    # 避免额外分配巨大中间矩阵
    # 若存在 r = -1 导致 1+r=0 → -inf，这里保留 -inf，稍后统一过滤
    np.log1p(R, out=R)

    # --- 3) 年化收益（对数收益的年化）：(sum_t log(1+r_t))/T * 252 ---
    sum_log = R.sum(axis=0)  # (M,)
    ret_annual = (sum_log / float(T)) * float(trading_days)

    # --- 4) 年化波动（对数收益的标准差年化）---
    vol_annual = R.std(axis=0, ddof=ddof) * np.sqrt(float(trading_days))

    # --- 5) 组装 DataFrame（并过滤掉无效列）---
    valid = np.isfinite(ret_annual) & np.isfinite(vol_annual)
    if not np.all(valid):
        # 仅保留数值有效的组合（例如出现 -inf 的列会被去掉）
        ret_annual = ret_annual[valid]
        vol_annual = vol_annual[valid]
        portfolio_allocs = portfolio_allocs[valid, :]

    weight_df = pd.DataFrame(
        portfolio_allocs,
        columns=[f"w_{i}" for i in range(portfolio_allocs.shape[1])]
    )
    perf_df = pd.DataFrame(
        {
            "ret_annual": ret_annual,
            "vol_annual": vol_annual,
        }
    )
    return pd.concat([weight_df, perf_df], axis=1)


# 识别出位于有效前沿上的点
def cal_ef2_v4_ultra_fast(data: pd.DataFrame) -> pd.DataFrame:
    """
    从给定的投资组合点中，高效识别出位于有效前沿上的点。
    """
    data = data.copy()
    ret_values = data['ret_annual'].values
    vol_values = data['vol_annual'].values

    sorted_idx = np.argsort(ret_values)[::-1]

    sorted_vol = vol_values[sorted_idx]
    cummin_vol = np.minimum.accumulate(sorted_vol)

    on_ef_sorted = (sorted_vol <= cummin_vol + 1e-6)  # 加上一个小的容差

    on_ef = np.zeros(len(data), dtype=bool)
    on_ef[sorted_idx] = on_ef_sorted

    data['on_ef'] = on_ef
    return data


# 约束满足函数 (老)
def primal_dual_interior_point(proposal, the_single_limits, the_multi_limits, max_iter=100):
    """
    使用迭代投影法将一个可能无效的权重修正为满足所有约束的有效权重。
    """
    num_assets = len(proposal)
    num_constraints = 2 * num_assets + 2 * len(the_multi_limits) + 2

    A = np.zeros((num_constraints, num_assets))
    b = np.zeros(num_constraints)

    idx = 0
    for i_ in range(num_assets):
        A[idx, i_] = 1
        b[idx] = the_single_limits[i_][1]
        idx += 1
        A[idx, i_] = -1
        b[idx] = -the_single_limits[i_][0]
        idx += 1

    for indices, (lower, upper) in the_multi_limits.items():
        A[idx, list(indices)] = 1
        b[idx] = upper
        idx += 1
        A[idx, list(indices)] = -1
        b[idx] = -lower
        idx += 1

    A[idx, :] = 1
    b[idx] = 1
    A[idx + 1, :] = -1
    b[idx + 1] = -1

    x = np.copy(proposal)

    for _ in range(max_iter):
        Ax_b = A.dot(x) - b
        violating = Ax_b > 1e-6  # 添加容差

        if not np.any(violating):
            return x / np.sum(x)  # 返回前最后归一化

        if not np.any(A[violating]):  # 如果违反约束的行全为0，则无法修正
            return None

        correction = np.linalg.lstsq(A[violating], Ax_b[violating], rcond=None)[0]
        x -= correction

        x = np.clip(x, a_min=[lim[0] for lim in the_single_limits], a_max=[lim[1] for lim in the_single_limits])
        x /= np.sum(x)

    return None


# 约束满足函数 (POCS/Dykstra 投影)
def project_to_constraints_pocs(v: np.ndarray,
                                single_limits,  # list[(low, high)]
                                multi_limits: dict,  # {(tuple_idx): (low, high)}
                                max_iter=200, tol=1e-9, damping=1.0):
    """
    用 POCS 交替投影到：盒约束 ∩ 总和=1 ∩ 各组半空间 之交集。
    支持多资产联合约束。damping∈(0,1] 可缓解过冲。
    """
    x = v.copy()
    n = x.size

    # 预取单资产上下界向量
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    # 预编译组约束（a 向量的稀疏结构：组里为 1，其他为 0）
    groups = []
    for idx_tuple, (low, up) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=np.int64)
        a_norm2 = float(len(idx))  # ||a||^2 = 组大小
        groups.append((idx, float(low), float(up), a_norm2))

    # 初始：投影到盒约束+总和=1，减少后续振荡
    x = np.clip(x, lows, highs)
    s = x.sum()
    x = x + (1.0 - s) / n

    for _ in range(max_iter):
        x_prev = x

        # 1) 盒约束
        x = np.clip(x, lows, highs)

        # 2) 总和=1 的超平面投影
        s = x.sum()
        x = x + (1.0 - s) / n

        # 3) 每个组的上下半空间投影
        # 上界：a^T x ≤ up；若超出则投影：x ← x - ((a^T x - up)/||a||^2) a
        # 下界：a^T x ≥ low；若低于则投影：x ← x + ((low - a^T x)/||a||^2) a
        for idx, low, up, a_norm2 in groups:
            if a_norm2 == 0:
                continue
            t = x[idx].sum()
            if t > up + 1e-12:
                delta = (t - up) / a_norm2
                x[idx] -= damping * delta  # 等价于沿 -a 方向走
            elif t < low - 1e-12:
                delta = (low - t) / a_norm2
                x[idx] += damping * delta  # 等价于沿 +a 方向走

        # 再次总和=1（组投影会打破总和），保证可行
        s = x.sum()
        x = x + (1.0 - s) / n

        # 收敛判据 + 约束校验（可选更严格）
        if np.linalg.norm(x - x_prev, ord=np.inf) < tol:
            break

    # 最终一次严格校验（允许极小容差）
    if np.any(x < lows - 1e-6) or np.any(x > highs + 1e-6):
        return None
    for idx, low, up, _ in groups:
        if len(idx) == 0:
            continue
        t = x[idx].sum()
        if t < low - 1e-6 or t > up + 1e-6:
            return None
    # 总和
    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        return None
    return x


def _build_group_struct(multi_limits: Dict[Tuple[int, ...], Tuple[float, float]], n: int):
    """
    将多资产联合约束编译为向量化所需的稀疏结构：
    - members:  所有组成员索引拼接的一维数组（长度等于所有组成员总数）
    - gid:      与 members 对应的组id（同长度），用于 segment 归并/散射
    - gsize:    每个组的大小（m,）且转为 float64，作为 ||a||^2
    - low, up:  组下界与上界（m,）
    """
    if not multi_limits:
        # 空约束：返回最小结构以避免分支判断
        members = np.empty(0, dtype=np.int64)
        gid = np.empty(0, dtype=np.int64)
        gsize = np.empty(0, dtype=np.float64)
        low = np.empty(0, dtype=np.float64)
        up = np.empty(0, dtype=np.float64)
        return members, gid, gsize, low, up

    # 展平所有组
    members_list = []
    gid_list = []
    gsize_list = []
    low_list = []
    up_list = []

    g = 0
    for idx_tuple, (lo, hi) in multi_limits.items():
        idx = np.asarray(idx_tuple, dtype=np.int64)
        if idx.size == 0:
            continue
        # 基本健壮性（可按需放宽）
        # 这里不强制 idx 在 [0, n) 内或去重，默认上游保证
        members_list.append(idx)
        gid_list.append(np.full(idx.size, g, dtype=np.int64))
        gsize_list.append(float(idx.size))
        low_list.append(float(lo))
        up_list.append(float(hi))
        g += 1

    if g == 0:
        members = np.empty(0, dtype=np.int64)
        gid = np.empty(0, dtype=np.int64)
        gsize = np.empty(0, dtype=np.float64)
        low = np.empty(0, dtype=np.float64)
        up = np.empty(0, dtype=np.float64)
    else:
        members = np.concatenate(members_list, axis=0)
        gid = np.concatenate(gid_list, axis=0)
        gsize = np.asarray(gsize_list, dtype=np.float64)
        low = np.asarray(low_list, dtype=np.float64)
        up = np.asarray(up_list, dtype=np.float64)

    return members, gid, gsize, low, up


def project_to_constraints_pocs_fast(v: np.ndarray,
                                     single_limits: Iterable[Tuple[float, float]],
                                     multi_limits: Dict[Tuple[int, ...], Tuple[float, float]],
                                     max_iter: int = 200,
                                     tol: float = 1e-9,
                                     damping: float = 1.0):
    """
    高性能 POCS：在每次迭代中，
      1) 盒约束投影（向量化 clip）
      2) 总和=1 超平面投影（x += (1-sum)/n）
      3) 所有“组和”半空间并行投影（一次性算组和 -> 计算超/欠 -> segment 散射更新）

    与原版的差异：
      - 第3步不再逐组 for-loop，而是“并行组校正”（block-Jacobi 式），数值上更稳定；
      - 保留 damping∈(0,1] 抑制过冲；
      - 收敛停止使用 L_inf 范数；尾部做严格校验，若有违约束返回 None。

    参数与返回：
      v: 初始向量（不会被原地改动）
      single_limits: [(low_i, high_i)]
      multi_limits: {(i,j,k,...): (low, high)}
      返回：满足所有约束的解向量；若未满足（数值容差1e-6）则返回 None
    """
    x = np.array(v, dtype=np.float64, copy=True)
    n = x.size

    # 单资产上下界（一次性取出成向量，零拷贝广播）
    single_limits = tuple(single_limits)
    lows = np.fromiter((a for a, _ in single_limits), count=n, dtype=np.float64)
    highs = np.fromiter((b for _, b in single_limits), count=n, dtype=np.float64)

    # 预编译组结构
    members, gid, gsize, low_g, up_g = _build_group_struct(multi_limits, n)
    m = low_g.size  # 组数

    inv_n = 1.0 / n

    # 预处理：先盒约束，再对齐到 sum=1，减少初始振荡
    np.clip(x, lows, highs, out=x)
    x += (1.0 - x.sum()) * inv_n

    # 为收敛判据准备缓冲，避免每轮分配新内存
    x_prev = x.copy()

    for _ in range(max_iter):
        # 1) 盒约束
        np.clip(x, lows, highs, out=x)

        # 2) 总和=1
        x += (1.0 - x.sum()) * inv_n

        # 3) 并行组投影（若 m=0 则跳过）
        if m:
            # 组和：sum_{i in group g} x_i
            # 用 bincount 做 segment reduce：O(#members)
            t = np.bincount(gid, weights=x[members], minlength=m).astype(np.float64, copy=False)

            # 超出/低于量（>=0）
            over = t - up_g
            over = np.where(over > 0.0, over, 0.0)  # max(t - up, 0)
            under = low_g - t
            under = np.where(under > 0.0, under, 0.0)  # max(low - t, 0)

            # 每组需要在组内均匀分摊的校正量：(+表示上调, -表示下调)
            #   t>up  ⇒  每个成员 -= (t-up)/|G|
            #   t<low ⇒  每个成员 += (low-t)/|G|
            # 合并为：net = (-over + under) / |G|
            net = (-over + under) / gsize
            if damping != 1.0:
                net *= damping

            # 将 net[g] 散射到资产维度：Δx_i = sum_{g: i∈G} net[g]
            # 用 bincount 做“组到资产”的一次性聚合，避免 np.add.at 的原子慢路径
            delta_x = np.bincount(members, weights=net[gid], minlength=n).astype(np.float64, copy=False)
            x += delta_x

            # 再次总和=1（组投影会改变总和）
            x += (1.0 - x.sum()) * inv_n

        # 收敛判据（L_inf）
        if np.max(np.abs(x - x_prev)) < tol:
            break
        # 覆盖旧缓冲，避免重复分配
        x_prev[:] = x

    # -------- 严格校验（带容差） --------
    # 单资产
    if (x < lows - 1e-6).any() or (x > highs + 1e-6).any():
        return None

    # 组和
    if m:
        t = np.bincount(gid, weights=x[members], minlength=m).astype(np.float64, copy=False)
        if (t < low_g - 1e-6).any() or (t > up_g + 1e-6).any():
            return None

    # 总和
    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        return None

    return x


if __name__ == '__main__':
    s_t = time.time()
    # --- 1. 数据加载和预处理 ---
    hist_value = pd.read_excel('历史净值数据.xlsx', sheet_name='历史净值数据')
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
    proposed_alloc = {
        'C1': {'货币现金类': 1.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }

    # --- 2. 约束随机游走 ---
    print("开始通过约束随机游走生成投资组合...")
    num_of_asset = len(assets_list)
    current_weights = np.array([1 / num_of_asset] * num_of_asset)
    step_size = 0.99
    single_limits = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # 示例：每个资产的权重限制在0%到100%
    multi_limits = {}  # 多资产联合约束, 写法: {(tuple_idx): (low, high)}

    final_weights = []
    s_t_2 = time.time()
    for i in range(100000):
        new_proposal = current_weights + np.random.normal(0, step_size, len(current_weights))
        adjusted_weights = project_to_constraints_pocs_fast(new_proposal, single_limits, multi_limits)

        if adjusted_weights is not None:
            final_weights.append(adjusted_weights)
            current_weights = adjusted_weights

    print(f"成功生成 {len(final_weights)} 个候选投资组合。耗时 {time.time() - s_t_2:.2f} 秒。")

    # --- 3. 显式校验和过滤权重 (新功能) ---
    print("正在对所有生成的权重进行最终校验...")

    validated_weights = []
    for w in final_weights:
        # 校验1: 权重和为1
        if not np.isclose(np.sum(w), 1.0, atol=1e-6):
            continue

        # 校验2: 单资产上下限
        single_valid = all(
            single_limits[i][0] - 1e-6 <= w[i] <= single_limits[i][1] + 1e-6 for i in range(num_of_asset))
        if not single_valid:
            continue

        # 校验3: 多资产组合上下限
        multi_valid = all(lower - 1e-6 <= np.sum(w[list(indices)]) <= upper + 1e-6 for indices, (lower, upper) in
                          multi_limits.items())
        if not multi_valid:
            continue

        validated_weights.append(w)

    print(f"校验完成。有效权重数量: {len(validated_weights)} / {len(final_weights)}")

    # 后续计算使用校验过的权重
    final_weights = validated_weights

    # --- 4. 批量计算收益和风险 ---
    if final_weights:
        print("正在批量计算所有组合的收益与风险...")
        weights_array = np.array(final_weights)
        port_daily_returns = hist_value_r[assets_list].values

        # 批量计算
        s_t_3 = time.time()
        # results_df = generate_alloc_perf_batch(port_daily_returns, weights_array)
        results_df = generate_alloc_perf_batch_fast(port_daily_returns, weights_array)

        # 找出有效前沿
        results_df = cal_ef2_v4_ultra_fast(results_df)
        print(f"计算完成, 耗时: {time.time() - s_t_3:.2f} 秒")

        # --- 5. 使用模块化函数进行交互式可视化 ---
        print("正在生成交互式图表...")

        # 为权重列重命名以用于悬停文本
        weight_cols = {f'w_{i}': assets_list[i] for i in range(len(assets_list))}
        results_df = results_df.rename(columns=weight_cols)


        def create_hover_text(df_row):
            text = f"年化收益率: {df_row['ret_annual']:.2%}<br>年化波动率: {df_row['vol_annual']:.2%}<br><br><b>资产权重</b>:<br>"
            for asset in assets_list:
                if asset in df_row and df_row[asset] > 1e-4:
                    text += f"  {asset}: {df_row[asset]:.1%}<br>"
            return text


        results_df['hover_text'] = results_df.apply(create_hover_text, axis=1)

        # 准备绘图数据
        ef_df = results_df[results_df['on_ef'] == True].copy()
        random_df = results_df.copy()

        # 定义要绘制的数据点集合
        scatter_data_to_plot = [
            {
                "data": random_df,
                "name": "随机权重数据点",
                "color": "grey",
                "size": 2,
                "opacity": 0.5
            },
            {
                "data": ef_df,
                "name": "有效前沿数据点",
                "color": "blue",
                "size": 3,
                "opacity": 0.8,
                # "marker_line": dict(width=1, color='darkslategrey')
            }
        ]

        # 调用新的绘图函数
        plot_efficient_frontier(
            scatter_points_data=scatter_data_to_plot,
            title='约束随机游走生成的投资组合与有效前沿'
        )

    else:
        print("未能生成任何有效的投资组合，无法进行后续计算和绘图。")

    print(f"总耗时: {time.time() - s_t:.2f} 秒")
