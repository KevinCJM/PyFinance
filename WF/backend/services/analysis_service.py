import numpy as np
import pandas as pd
from typing import Iterable, Optional, Tuple
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Bar, Grid, Scatter

pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示


def find_a_points(
        df: pd.DataFrame,
        *,
        code_col: str = "code",
        date_col: str = "date",
        close_col: str = "close",
        volume_col: str = "volume",

        # ==== 0) 条件总开关 ====
        enable_cond1: bool = True,  # 条件1：长均线下跌
        enable_cond2: bool = True,  # 条件2：短均线上穿（“A点仅在当日上穿触发”）
        enable_cond3: bool = True,  # 条件3：A点前N天 价格上穿确认

        # ==== 1) 可调均线 ====
        short_windows: Tuple[int, ...] = (5, 10),  # 多个短均线窗口
        long_window: int = 60,  # 长均线窗口

        # ==== 2) 上穿确认要求（A点当天触发）====
        cross_window: int = 3,  # 允许多条短均线在最近 cross_window 天内先后完成上穿（仅在 require_all=True 时使用）
        require_all: bool = True,  # True=集合内全部满足；False=只要求“当日至少一条发生上穿”

        # ==== 3) 可选：A点前N天，价格上穿 N 日均线 ====
        confirm_lookback_days: int = 0,  # 0=关闭
        confirm_ma_window: Optional[int] = None,  # 需要上穿的均线窗口；None=关闭
        confirm_price_col: str = "high",  # "high" 或 "close"

        # ==== 长均线下跌确认 ====
        down_lookback: int = 30,  # 要求 ma_long(t-1) < ma_long(t-1-down_lookback)

        # ==== 文本说明 ====
        with_explain_strings: bool = True,  # 是否生成 A_point_desc

        # ==== 其他可参数化选项（保持最小化，遵循原始逻辑）====
        vr1_lookback: int = 10,  # VR1 的回看窗口长度（取 max 之前）
        eps: float = 0.0,  # 比较时的容忍度，例如 > 变为 > + eps
        cross_requires_prev_below: bool = True,  # 上穿是否要求前一日不在上方（原始逻辑：要求）
        # ==== 条件组参数（独立控制，可选，提供则覆盖上面的同类项）====
        cond1: Optional[dict] = None,
        cond2: Optional[dict] = None,
        cond3: Optional[dict] = None,
) -> pd.DataFrame:
    """
    返回：在原 df 基础上新增条件解释列与信号列（仅返回 df，不单独返回 a_points）。

    条件定义（默认）：
      条件1（ma_long_down）：  MA_long(t-1) < MA_long(t-1-down_lookback)
      条件2（cross 当日触发）：
        - required_shorts = 必须关注的短均线集合；
        - 当日至少有一条 required_shorts 完成“由下向上上穿 MA_long”的事件（A点触发日必须发生上穿）；
        - 若 require_all=True，再额外要求：
            · 最近 cross_window 日内，required_shorts 每条均已出现过上穿；
            · 且当日这些短均线均在 MA_long 上方。
      条件3（confirm，可选）：
        - A 点前 confirm_lookback_days 日内，confirm_price_col 至少有一天“由下向上上穿 MA_{confirm_ma_window}”。

    额外输出：
      - VR1(t) = V_t / max(V_{t-10..t-1})
      - 每条短均线的 above_k / crossup_k / recent_cross_k（便于复核）
      - condX_enabled/ok/expect/actual 与 A_point
    """
    # ---- 独立条件参数合并 —— 默认值与传入 condX 合并 ----
    c1_enabled = bool(enable_cond1 if cond1 is None else cond1.get("enabled", enable_cond1))
    c1_long = int(long_window if cond1 is None else cond1.get("long_window", long_window))
    c1_down_lookback = int(down_lookback if cond1 is None else cond1.get("down_lookback", down_lookback))

    # 条件2：可独立的短/长窗口与集合
    c2_enabled = bool(enable_cond2 if cond2 is None else cond2.get("enabled", enable_cond2))
    c2_shorts = tuple(short_windows if (cond2 is None or cond2.get("short_windows") is None)
                      else tuple(cond2.get("short_windows")))
    c2_long = int(long_window if cond2 is None else cond2.get("long_window", long_window))
    c2_cross_window = int(cross_window if cond2 is None else cond2.get("cross_window", cross_window))
    c2_require_all = bool(require_all if cond2 is None else cond2.get("require_all", require_all))
    # required_shorts 合并
    if cond2 is not None and cond2.get("required_shorts") is not None:
        req_s = tuple(sorted(set(cond2.get("required_shorts"))))
    else:
        req_s = tuple(sorted(set(c2_shorts)))
    required_shorts = req_s

    c2_cross_prev_below = bool(cross_requires_prev_below if cond2 is None
                               else cond2.get("cross_requires_prev_below", cross_requires_prev_below))

    # 条件3：独立确认参数
    c3_enabled = bool(enable_cond3 if cond3 is None else cond3.get("enabled", enable_cond3))
    c3_lookback = int(
        confirm_lookback_days if cond3 is None else cond3.get("confirm_lookback_days", confirm_lookback_days))
    c3_ma = (confirm_ma_window if cond3 is None else cond3.get("confirm_ma_window", confirm_ma_window))
    c3_price_col = str(confirm_price_col if cond3 is None else cond3.get("confirm_price_col", confirm_price_col))
    # 原始逻辑固定需要≥1次价格上穿；这里保留读取但默认1
    c3_min_crosses = int(cond3.get("confirm_min_crosses", 1)) if cond3 is not None else 1

    # ---------- 预处理 ----------
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d.sort_values([code_col, date_col], inplace=True)
    g = d.groupby(code_col, sort=False)

    # ---------- 均线 ----------
    needed_ma_windows = set(c2_shorts) | {c1_long, c2_long}
    if c3_ma:
        needed_ma_windows.add(int(c3_ma))
    for k in sorted(needed_ma_windows):
        # 原始逻辑：均线 min_periods = 窗口长度
        d[f"ma_{k}"] = g[close_col].transform(lambda s: s.rolling(k, min_periods=k).mean())

    ma_long1 = f"ma_{c1_long}"
    ma_long2 = f"ma_{c2_long}"

    # ---------- 条件1：长均线下跌 ----------
    d["cond1_enabled"] = bool(c1_enabled)

    maL_t1 = g[ma_long1].shift(1)
    maL_t1_L = g[ma_long1].shift(1 + c1_down_lookback)
    cond1_ok_series = (maL_t1.notna() & maL_t1_L.notna()) & ((maL_t1 + eps) < maL_t1_L)
    d["ma_long_down"] = cond1_ok_series
    d["cond1_ok"] = np.where(c1_enabled, cond1_ok_series, True)

    # 期望/实际文本（逐元素格式化）
    d["cond1_expect"] = pd.Series(
        f"MA{c1_long}(t-1) < MA{c1_long}(t-1-{c1_down_lookback})", index=d.index
    )
    ma_t1_str = maL_t1.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    ma_t1L_str = maL_t1_L.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")

    # ✅ 用向量化字符串拼接，不再使用 .format 放 Series
    prefix1 = f"MA{c1_long}(t-1)="
    prefix2 = f", MA{c1_long}(t-1-{c1_down_lookback})="
    d["cond1_actual"] = prefix1 + ma_t1_str + prefix2 + ma_t1L_str

    # ---------- 条件2：短均线上穿（仅当日触发） ----------
    d["cond2_enabled"] = bool(c2_enabled)

    # 为每条短均线生成 above_k / crossup_k / recent_cross_k
    for k in c2_shorts:
        ma_s = f"ma_{k}"
        above_col = f"above_{k}"
        cross_col = f"crossup_{k}"

        d[above_col] = (d[ma_s] + eps) > d[ma_long2]
        # 原始逻辑：要求上一日不在上方（由下向上上穿）
        prev_above = (g[ma_s].shift(1) + eps) > g[ma_long2].shift(1)
        d[cross_col] = d[above_col] & (~prev_above.fillna(False))

    # 当日至少一条 required_shorts 发生上穿
    today_any_cross = None
    for k in required_shorts:
        c = d[f"crossup_{k}"]
        today_any_cross = c if today_any_cross is None else (today_any_cross | c)
    if today_any_cross is None:
        today_any_cross = pd.Series(False, index=d.index)

    # （仅在 require_all=True 时使用）最近 cross_window 内 required_shorts 是否“每条均已发生过上穿”
    if c2_require_all:
        recent_all = None
        for k in required_shorts:
            rc = g[f"crossup_{k}"].transform(lambda s: s.rolling(c2_cross_window, min_periods=1).max()) > 0
            recent_all = rc if recent_all is None else (recent_all & rc)

        # 当日 required_shorts 是否全部在 MA_long 上方
        all_above_today = None
        for k in required_shorts:
            a = d[f"above_{k}"]
            all_above_today = a if all_above_today is None else (all_above_today & a)

        cond2_core = today_any_cross & recent_all & all_above_today
    else:
        cond2_core = today_any_cross
        recent_all = pd.Series(True, index=d.index)
        all_above_today = pd.Series(True, index=d.index)

    d["today_any_cross"] = today_any_cross
    d["recent_all_cross"] = recent_all
    d["today_all_above"] = all_above_today

    d["cond2_ok"] = np.where(c2_enabled, cond2_core, True)

    # 期望/实际（条件2）
    req_list_str = "、".join([f"MA{k}" for k in required_shorts])
    if c2_require_all:
        cond2_expect_str = (
            f"当日至少一条({req_list_str})上穿MA{c2_long}；"
            f"最近{c2_cross_window}日内上述每条均出现过上穿；"
            f"且当日这些均线均在MA{c2_long}上方"
        )
    else:
        cond2_expect_str = f"当日至少一条({req_list_str})上穿MA{c2_long}"

    d["cond2_expect"] = pd.Series(cond2_expect_str, index=d.index)

    # 实际：列出当日完成上穿的短均线名单 + 三个布尔子条件
    crossed_names = []
    for k in required_shorts:
        crossed_names.append("MA{}:".format(k) + d[f"crossup_{k}"].astype(bool).astype(str))
    crossed_combined = crossed_names[0]
    for i in range(1, len(crossed_names)):
        crossed_combined = crossed_combined + ", " + crossed_names[i]

    d["cond2_actual"] = (
            "today_cross[" + crossed_combined + "]"
            + ", recent_all=" + recent_all.astype(bool).astype(str)
            + ", today_all_above=" + all_above_today.astype(bool).astype(str)
    )

    # ---------- 条件3：A点前 N 天 价格上穿 MA_confirm（由下向上） ----------
    d["cond3_enabled"] = bool(c3_enabled)

    if (c3_lookback > 0) and (c3_ma is not None) and c3_enabled:
        price_col = c3_price_col
        if price_col not in d.columns:
            raise ValueError(f"confirm_price_col='{price_col}' 不在 df 列中。")
        ma_c = f"ma_{int(c3_ma)}"

        prev_below = g[price_col].shift(1) < g[ma_c].shift(1)
        cross_price_up = (d[price_col] >= d[ma_c]) & prev_below.fillna(False)
        d["_cross_price_up"] = cross_price_up.astype("int8")

        # 统计 [t-N, t-1] 内的上穿次数
        confirm_cnt = g["_cross_price_up"].transform(
            lambda s: s.shift(1).rolling(int(c3_lookback), min_periods=int(c3_lookback)).sum()
        )
        d["confirm_cross_cnt"] = confirm_cnt
        # 原始逻辑：至少 1 次
        cond3_ok_series = (confirm_cnt >= 1)
        cond3_expect_str = f"A点前{c3_lookback}日内，{price_col}由下上穿MA{int(c3_ma)}≥1次（区间[t-N,t-1]）"
        # 实际：给出次数
        cnt_str = confirm_cnt.fillna(0).astype(int).astype(str)
        d["cond3_actual"] = "cross_cnt=" + cnt_str
    else:
        cond3_ok_series = pd.Series(True, index=d.index)  # 未启用则视为通过
        cond3_expect_str = "（未启用）"
        d["confirm_cross_cnt"] = np.nan
        d["_cross_price_up"] = np.nan
        d["cond3_actual"] = "（未启用）"

    d["cond3_ok"] = np.where(c3_enabled, cond3_ok_series, True)
    d["cond3_expect"] = pd.Series(cond3_expect_str, index=d.index)

    # ---------- VR1 ----------
    prev10max = g[volume_col].transform(lambda s: s.shift(1).rolling(int(vr1_lookback), min_periods=1).max())
    d["vr1"] = d[volume_col] / prev10max

    # ---------- 综合判定：仅把“启用的条件”计入 ----------
    overall_ok = (
            (~d["cond1_enabled"] | d["cond1_ok"]) &
            (~d["cond2_enabled"] | d["cond2_ok"]) &
            (~d["cond3_enabled"] | d["cond3_ok"])
    )
    d["A_point"] = overall_ok.astype(bool)

    # ---------- 可选：中文说明（仅 A_point=True 时） ----------
    if with_explain_strings:
        def _desc(row: pd.Series) -> Optional[str]:
            if not row["A_point"]:
                return None
            code = str(row[code_col])
            dt = pd.to_datetime(row[date_col]).strftime("%Y年%m月%d日")
            parts = []
            parts.append(f"条件1: {row['cond1_expect']}（实际：{row['cond1_actual']}）")
            parts.append(f"条件2: {row['cond2_expect']}（实际：{row['cond2_actual']}）")
            parts.append(f"条件3: {row['cond3_expect']}（实际：{row['cond3_actual']}）")
            return f"{code}股票在{dt}, 为A点, 符合 " + "；".join(parts)

        d["A_point_desc"] = d.apply(_desc, axis=1)

    return d


def find_b_points(
        df: pd.DataFrame,
        *,
        code_col: str = "code",
        date_col: str = "date",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        ma60_col: str = "ma_60",

        # —— 条件总开关（每条可单独关闭）——
        enable_cond1: bool = True,  # 条件1：时间要求
        enable_cond2: bool = True,  # 条件2：短期线在长期线之上
        enable_cond3: bool = True,  # 条件3：接近长期线 + 阴线/收≤昨收（无VR1）
        enable_cond4: bool = True,  # 条件4：量能上限（VR1）
        enable_cond5: bool = True,  # 条件5：干缩
        enable_cond6: bool = False,  # 条件6：价稳

        # ① 时间要求（仅时间/配对控制）
        min_days_from_a: int = 60,  # 从A点开始的最小天数（cond1 回退默认）
        max_days_from_a: Optional[int] = None,  # 最大天数（可空）
        allow_multi_b_per_a: bool = True,  # 是否允许一个A点对应多个B点（默认允许）

        # ② 短期线在长期线之上（MA_N 在 MA_long 上）
        above_maN_window: int = 5,  # 短期均线窗口
        long_ma_window: Optional[int] = None,  # 长期均线窗口（为空则使用 ma60_col）
        above_maN_days: int = 15,  # 在上方的天数（累计）
        above_maN_consecutive: bool = False,  # 是否要求连续在上方
        max_maN_below_days: int = 5,  # 下方允许的最多天数（非连续模式）

        # ③ 触及/击破 MA_long + 阴线/收≤昨收（无VR1）
        touch_price: str = "low",  # "low" 或 "close"（cond3 回退默认）
        touch_relation: str = "le",  # "le" 或 "lt"（cond3 回退默认）
        require_bearish: bool = False,  # 是否要求阴线（cond3 回退默认）
        require_close_le_prev: bool = False,  # 是否要求收盘价 ≤ 昨收（cond3 回退默认）

        # ④ 量能上限（VR1）
        vr1_max: Optional[float] = None,  # cond4 回退默认
        recent_max_vol_window: int = 10,

        # ④ 干缩
        dryness_ratio_max: float = 0.8,  # VMA5/VMA10 ≤ 阈值（cond4 回退默认）
        require_vol_le_vma10: bool = True,
        dryness_recent_window: int = 0,  # 0 = 不启用（cond4 回退默认）
        dryness_recent_min_days: int = 0,

        # ⑤ 价稳
        price_stable_mode: str = "no_new_low",  # "no_new_low" | "ratio" | "atr"（cond5 回退默认）
        max_drop_ratio: float = 0.0,
        use_atr_window: int = 14,
        atr_buffer: float = 0.5,

        # 文本说明
        with_explain_strings: bool = True,

        # —— 条件参数组（优先级高于回退默认）——
        cond1: Optional[dict] = None,
        cond2: Optional[dict] = None,
        cond3: Optional[dict] = None,
        cond4: Optional[dict] = None,
        cond5: Optional[dict] = None,
        cond6: Optional[dict] = None,
) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d.sort_values([code_col, date_col], inplace=True)
    g_code = d.groupby(code_col, sort=False)

    # —— 基础派生：分段/天数/段内前低 ——
    if "seg_id" not in d.columns:
        if "A_point" not in d.columns:
            raise ValueError("缺少 seg_id 且没有 A_point，无法分段。请提供 seg_id 或 A_point。")
        d["seg_id"] = g_code["A_point"].cumsum()

    if "days_since_A" not in d.columns:
        d["_seg_cum"] = d.groupby([code_col, "seg_id"]).cumcount()
        d["days_since_A"] = d["_seg_cum"]

    if "low_prev_min" not in d.columns:
        low_cummin = d.groupby([code_col, "seg_id"])[low_col].cummin()
        d["low_prev_min"] = d.groupby([code_col, "seg_id"])[low_col] \
            .shift(1).combine_first(low_cummin.shift(1))

    # —— 成交量均线（默认可供其他地方使用；条件3会根据自身参数计算专属短/长窗口）——
    if "vma_5" not in d.columns:
        d["vma_5"] = g_code[volume_col].transform(lambda s: s.rolling(5, min_periods=5).mean())
    if "vma_10" not in d.columns:
        d["vma_10"] = g_code[volume_col].transform(lambda s: s.rolling(10, min_periods=10).mean())

    # ============================================================
    # 条件1：时间要求（仅使用 cond1 参数）
    # ============================================================
    _c1 = cond1 or {}
    c1_enabled = bool(_c1.get("enabled", enable_cond1))
    c1_min_days_from_a = int(_c1.get("min_days_from_a", min_days_from_a))
    _max_days_val = _c1.get("max_days_from_a", max_days_from_a)
    try:
        c1_max_days_from_a = int(_max_days_val) if _max_days_val not in (None, "") else None
    except Exception:
        c1_max_days_from_a = None
    c1_allow_multi = bool(_c1.get("allow_multi_b_per_a", allow_multi_b_per_a))

    d["cond1_enabled"] = bool(c1_enabled)
    if c1_max_days_from_a is not None and c1_max_days_from_a >= 0:
        cond1_time_ok = (d["seg_id"] > 0) & (d["days_since_A"] >= c1_min_days_from_a) & (
                d["days_since_A"] <= c1_max_days_from_a)
    else:
        cond1_time_ok = (d["seg_id"] > 0) & (d["days_since_A"] >= c1_min_days_from_a)
    d["cond1_time_ok"] = cond1_time_ok
    d["cond1_ok"] = np.where(c1_enabled, cond1_time_ok, True)
    # 期望/实际（向量化）
    if c1_max_days_from_a is not None:
        time_expect = f"A→B在[{c1_min_days_from_a},{c1_max_days_from_a}]日内"
    else:
        time_expect = f"A→B≥{c1_min_days_from_a}日"
    d["cond1_expect"] = pd.Series(time_expect, index=d.index)
    d["cond1_actual"] = ("seg_id=" + d["seg_id"].astype("Int64").astype(str)
                         + ", days_since_A=" + d["days_since_A"].astype("Int64").astype(str))

    # ============================================================
    # 条件2：短期线在长期线之上（仅使用 cond2 参数）
    # ============================================================
    _c2 = cond2 or {}
    c2_enabled = bool(_c2.get("enabled", enable_cond2))
    c2_above_maN_window = int(_c2.get("above_maN_window", above_maN_window))
    c2_above_maN_days = int(_c2.get("above_maN_days", above_maN_days))
    c2_above_maN_consecutive = bool(_c2.get("above_maN_consecutive", above_maN_consecutive))
    c2_max_maN_below_days = int(_c2.get("max_maN_below_days", max_maN_below_days))
    c2_above_maN_ratio = _c2.get("above_maN_ratio", None)
    if c2_above_maN_ratio is not None:
        try:
            c2_above_maN_ratio = float(c2_above_maN_ratio)
        except Exception:
            c2_above_maN_ratio = None
    _c2_long_win = _c2.get("long_ma_window", _c2.get("long_ma_days", long_ma_window))
    try:
        c2_long_ma_window = int(_c2_long_win) if _c2_long_win not in (None, "") else None
    except Exception:
        c2_long_ma_window = None

    d["cond2_enabled"] = bool(c2_enabled)

    maN_col = f"ma_{int(c2_above_maN_window)}"
    if maN_col not in d.columns:
        d[maN_col] = g_code[close_col].transform(
            lambda s: s.rolling(int(c2_above_maN_window), min_periods=int(c2_above_maN_window)).mean()
        )
    # 条件2的长期参考线：优先使用 cond2.long_ma_window，否则用传入列 ma60_col
    if c2_long_ma_window and c2_long_ma_window > 0:
        c2_ma_col = f"__c2_ma_{int(c2_long_ma_window)}__"
        if c2_ma_col not in d.columns:
            d[c2_ma_col] = g_code[close_col].transform(
                lambda s: s.rolling(int(c2_long_ma_window), min_periods=int(c2_long_ma_window)).mean()
            )
        long_ref = d[c2_ma_col]
    else:
        long_ref = d[ma60_col]
    maN_above = d[maN_col] >= long_ref
    g_seg = d.groupby([code_col, "seg_id"], sort=False)
    # 统一计算 A..t-1 期间“在上方”的累计与比例，供所有判定与诊断使用
    d["_maN_above_int"] = maN_above.astype("int8")
    above_cum = g_seg["_maN_above_int"].transform(lambda s: s.cumsum())
    above_at_A = g_seg["_maN_above_int"].transform("first")
    # A..t-1 累计（不含今天）
    d["maN_above_since_A_tminus1"] = (above_cum.shift(1).fillna(0) - above_at_A.fillna(0))
    span_len_excl = (d["days_since_A"] - 1).clip(lower=0)
    d["maN_below_since_A_tminus1"] = (span_len_excl - d["maN_above_since_A_tminus1"]).clip(lower=0)
    # A..t 累计（含今天）：用于“在上方的比例%”显示与按比例判定
    d["maN_above_since_A_incl_today"] = (above_cum - above_at_A)
    span_len_incl = d["days_since_A"].clip(lower=0)
    denom_incl = span_len_incl.replace(0, np.nan)
    d["maN_above_ratio"] = d["maN_above_since_A_incl_today"] / denom_incl

    if c2_above_maN_consecutive:
        d["_maN_not_above"] = (~maN_above).astype("int8")
        grp_break = g_seg["_maN_not_above"].transform(lambda s: s.cumsum())
        consec = maN_above.groupby([d[code_col], d["seg_id"], grp_break]).cumcount() + 1
        consec = pd.Series(np.where(maN_above.values, consec, 0), index=d.index)
        d["maN_consec_len_prev"] = consec.shift(1).fillna(0).astype(int)
        cond2_maN_ok = d["maN_consec_len_prev"] >= int(c2_above_maN_days)
        maN_expect_str = f"MA{c2_above_maN_window}在长期线上连续≥{c2_above_maN_days}日（A..t-1）"
        maN_actual_text = "MA{}在长期线上(连续)=".format(c2_above_maN_window) + \
                          d["maN_consec_len_prev"].astype(int).astype(str) + "天"
    else:
        if c2_above_maN_ratio is not None:
            ratio = d["maN_above_ratio"]
            cond2_maN_ok = ratio >= float(c2_above_maN_ratio)
            ratio_pct = (ratio * 100.0).map(lambda x: f"{x:.0f}%" if pd.notna(x) else "nan")
            exp_pct = f"{float(c2_above_maN_ratio) * 100:.0f}%"
            maN_expect_str = f"A..t 期间 MA{c2_above_maN_window}在长期线上比例≥{exp_pct}（含今天）"
            maN_actual_text = ("MA{}在长期线上比例=".format(c2_above_maN_window) + ratio_pct +
                               ", 上方天数(A..t)=" + d["maN_above_since_A_incl_today"].fillna(0).astype(int).astype(
                        str) +
                               ", 窗口天数(A..t)=" + span_len_incl.fillna(0).astype(int).astype(str))
        else:
            cond2_maN_ok = (d["maN_above_since_A_tminus1"] >= int(c2_above_maN_days)) & \
                           (d["maN_below_since_A_tminus1"] <= int(c2_max_maN_below_days))
            maN_expect_str = f"MA{c2_above_maN_window}在长期线上累计≥{c2_above_maN_days}日且下方≤{c2_max_maN_below_days}日（A..t-1）"
            maN_actual_text = ("MA{}在长期线上(累计)=".format(c2_above_maN_window)
                               + d["maN_above_since_A_tminus1"].fillna(0).astype(int).astype(str)
                               + "天, 下方=" + d["maN_below_since_A_tminus1"].fillna(0).astype(int).astype(str) + "天")

    d["cond2_maN_rule_ok"] = cond2_maN_ok
    d["cond2_ok"] = np.where(c2_enabled, cond2_maN_ok, True)
    d["cond2_expect"] = pd.Series(maN_expect_str, index=d.index)
    d["cond2_actual"] = (maN_actual_text if isinstance(maN_actual_text, pd.Series)
                         else pd.Series(maN_actual_text, index=d.index))

    # ============================================================
    # 条件3：阴线/收≤昨收 + 触及/击破 MA_long（仅使用 cond3 参数）
    # ============================================================
    _c3 = cond3 or {}
    c3_enabled = bool(_c3.get("enabled", enable_cond3))
    c3_touch_price = str(_c3.get("touch_price", touch_price))
    c3_touch_relation = str(_c3.get("touch_relation", touch_relation))
    c3_require_bearish = bool(_c3.get("require_bearish", require_bearish))
    c3_require_close_le_prev = bool(_c3.get("require_close_le_prev", require_close_le_prev))
    d["cond3_enabled"] = bool(c3_enabled)
    prev_close = g_code[close_col].shift(1)

    cond3_bearish_ok = (d[close_col] < d[open_col]) if c3_require_bearish else pd.Series(True, index=d.index)
    cond3_close_le_prev_ok = (d[close_col] <= prev_close) if c3_require_close_le_prev else pd.Series(True,
                                                                                                     index=d.index)

    cmp_left = d[low_col] if c3_touch_price == "low" else d[close_col]
    cond3_touch_ok = (cmp_left <= d[ma60_col]) if c3_touch_relation == "le" else (cmp_left < d[ma60_col])

    d["cond3_bearish_ok"] = cond3_bearish_ok
    d["cond3_close_le_prev_ok"] = cond3_close_le_prev_ok
    d["cond3_touch_ok"] = cond3_touch_ok

    cond3_all = cond3_bearish_ok & cond3_close_le_prev_ok & cond3_touch_ok
    d["cond3_ok"] = np.where(c3_enabled, cond3_all, True)

    _bear_txt = "阴线" if c3_require_bearish else "不强制阴线"
    _prev_txt = "收≤昨收" if c3_require_close_le_prev else "不强制收≤昨收"
    _touch_txt = f"{c3_touch_price}{'≤' if c3_touch_relation == 'le' else '<'}长期均线"
    d["cond3_expect"] = pd.Series(f"{_bear_txt}；{_prev_txt}；{_touch_txt}", index=d.index)

    d["cond3_actual"] = ("bearish=" + cond3_bearish_ok.astype(bool).astype(str)
                         + ", close≤prev=" + cond3_close_le_prev_ok.astype(bool).astype(str)
                         + ", " + _touch_txt + "=" + cond3_touch_ok.astype(bool).astype(str))

    # ============================================================
    # 条件4：量能上限（VR1）（仅使用 cond4 参数）
    # ============================================================
    _c4 = cond4 or {}
    c4_enabled = bool(_c4.get("enabled", enable_cond4))
    c4_vr1_max = _c4.get("vr1_max", vr1_max)
    c4_recent_max_win = int(_c4.get("recent_max_vol_window", recent_max_vol_window))

    d["cond4_enabled"] = bool(c4_enabled)

    # 如果外部未提供 vr1，则基于 recent_max_vol_window 计算
    if "vr1" not in d.columns:
        prev_max = g_code[volume_col].shift(1).rolling(c4_recent_max_win, min_periods=1).max()
        d["vr1"] = d[volume_col] / prev_max

    cond4_vr1_ok = ((d["vr1"] <= float(c4_vr1_max)) if (c4_vr1_max is not None) else pd.Series(True, index=d.index))
    d["cond4_vr1_ok"] = cond4_vr1_ok
    d["cond4_ok"] = np.where(c4_enabled, cond4_vr1_ok, True)
    d["cond4_expect"] = pd.Series(
        (f"VR1≤{c4_vr1_max}（近{c4_recent_max_win}日最大量为参照）" if c4_vr1_max is not None else "（未设置VR1阈值）"),
        index=d.index
    )
    d["cond4_actual"] = ("vr1=" + d["vr1"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
                         + ", ok=" + cond4_vr1_ok.astype(bool).astype(str))

    # ============================================================
    # 条件5：干缩（仅使用 cond5 参数）
    # ============================================================
    _c5_dry = cond5 or {}
    c5_enabled_dry = bool(_c5_dry.get("enabled", enable_cond5))
    # 新式模块化开关（如未提供，则回退到旧式“全部同时要求”逻辑）
    has_new_flags = any(k in _c5_dry for k in (
        "ratio_enabled", "vol_cmp_enabled", "vol_down_enabled", "vol_decreasing_days",
        "vr1_enabled", "recent_n", "vr1_max", "vma_rel_enabled"
    ))
    # 在启用“新式缩量子模块”时，默认不强制旧式子模块（比例/量≤长均），需显式打开
    c5_ratio_enabled = bool(_c5_dry.get("ratio_enabled", (False if has_new_flags else True)))
    c5_vol_cmp_enabled = bool(_c5_dry.get("vol_cmp_enabled", (
        _c5_dry.get("require_vol_le_vma10", require_vol_le_vma10) if not has_new_flags else False)))
    c5_vol_down_enabled = bool(_c5_dry.get("vol_down_enabled", False))
    c5_vol_decreasing_days = int(_c5_dry.get("vol_decreasing_days", 3))
    # 新增：非放量（VR1上限）与 量均比较（短<=长）
    c5_vr1_enabled = bool(_c5_dry.get("vr1_enabled", False))
    c5_recent_n = int(_c5_dry.get("recent_n", 10))
    c5_vr1_max = _c5_dry.get("vr1_max", None)
    c5_vr1_max = float(c5_vr1_max) if c5_vr1_max not in (None, "") else None
    c5_vma_rel_enabled = bool(_c5_dry.get("vma_rel_enabled", False))

    c5_dryness_ratio_max = float(_c5_dry.get("dryness_ratio_max", dryness_ratio_max))
    c5_vma_short = int(_c5_dry.get("vma_short_window", _c5_dry.get("short_days", 5))) if _c5_dry.get("vma_short_window",
                                                                                                     None) not in (
                                                                                             None, "") else 5
    c5_vma_long = int(_c5_dry.get("vma_long_window", _c5_dry.get("long_days", 10))) if _c5_dry.get("vma_long_window",
                                                                                                   None) not in (
                                                                                           None, "") else 10
    # 均量比较的“长天数”可单独配置；未提供时回退为比例上限的长天数
    c5_vol_cmp_long = int(_c5_dry.get("vol_compare_long_window", c5_vma_long)) if _c5_dry.get("vol_compare_long_window",
                                                                                              None) not in (
                                                                                      None, "") else c5_vma_long
    # 旧式“近窗干缩次数”参数（仅在未使用新式模块化开关时生效）
    c5_dryness_recent_window = int(_c5_dry.get("dryness_recent_window", dryness_recent_window))
    c5_dryness_recent_min_days = int(_c5_dry.get("dryness_recent_min_days", dryness_recent_min_days))

    d["cond5_enabled"] = bool(c5_enabled_dry)

    # 仅在需要时计算量均
    need_vma = (c5_ratio_enabled or c5_vol_cmp_enabled or c5_vma_rel_enabled) or (not has_new_flags)
    if need_vma:
        vma_short = g_code[volume_col].transform(
            lambda s: s.rolling(int(max(1, c5_vma_short)), min_periods=int(max(1, c5_vma_short))).mean())
        vma_long = g_code[volume_col].transform(
            lambda s: s.rolling(int(max(1, c5_vma_long)), min_periods=int(max(1, c5_vma_long))).mean())
        vma_long_cmp = g_code[volume_col].transform(
            lambda s: s.rolling(int(max(1, c5_vol_cmp_long)), min_periods=int(max(1, c5_vol_cmp_long))).mean())
    else:
        vma_short = pd.Series(np.nan, index=d.index)
        vma_long = pd.Series(np.nan, index=d.index)
        vma_long_cmp = pd.Series(np.nan, index=d.index)
    d["vma_short"] = vma_short
    d["vma_long"] = vma_long
    d["vma_long_cmp"] = vma_long_cmp

    # 子模块1：均量比例上限（仅启用时有意义；未启用则诊断列返回 NaN）
    dryness_ratio = vma_short / vma_long
    d["dryness_ratio"] = dryness_ratio if c5_ratio_enabled else pd.Series(np.nan, index=d.index)
    c5_ratio_ok = (dryness_ratio <= float(c5_dryness_ratio_max)) if c5_ratio_enabled else pd.Series(True, index=d.index)
    # 子模块2：当日量不高于“长天数”均量（未启用则诊断列 NaN）
    c5_vol_cmp_ok = (d[volume_col] <= vma_long_cmp) if c5_vol_cmp_enabled else pd.Series(True, index=d.index)
    if not c5_vol_cmp_enabled:
        d["vma_long_cmp"] = pd.Series(np.nan, index=d.index)
    # 子模块2b：量均比较（短<=长）（未启用则诊断列 NaN）
    c5_vma_rel_ok = (vma_short <= vma_long) if c5_vma_rel_enabled else pd.Series(True, index=d.index)
    # 子模块3：近X日量连降（严格递减）
    # 计算“当日连降天数”与“是否满足近X日连降”（仅启用时）
    if c5_vol_down_enabled:
        dec = g_code[volume_col].diff(1) < 0  # 是否较前一日下降
        grp_key = (~dec).groupby(d[code_col]).cumsum()
        vol_down_streak = dec.astype(int).groupby([d[code_col], grp_key]).cumsum().astype(int)
        d["vol_down_streak"] = vol_down_streak
        if c5_vol_decreasing_days <= 1:
            c5_down_ok = pd.Series(True, index=d.index)
        else:
            win = int(max(1, c5_vol_decreasing_days))
            c5_down_ok = (vol_down_streak >= win)
    else:
        d["vol_down_streak"] = pd.Series(np.nan, index=d.index)
        c5_down_ok = pd.Series(True, index=d.index)
    c5_down_ok = c5_down_ok if c5_vol_down_enabled else pd.Series(True, index=d.index)

    # 输出子模块标志（便于前端诊断）
    d["c5_ratio_ok"] = (c5_ratio_ok if c5_ratio_enabled else pd.Series(np.nan, index=d.index))
    d["c5_vol_cmp_ok"] = (c5_vol_cmp_ok if c5_vol_cmp_enabled else pd.Series(np.nan, index=d.index))
    d["c5_vma_rel_ok"] = (c5_vma_rel_ok if c5_vma_rel_enabled else pd.Series(np.nan, index=d.index))
    d["c5_down_ok"] = (c5_down_ok if c5_vol_down_enabled else pd.Series(np.nan, index=d.index))
    # 子模块0：非放量（VR1 ≤ 阈值）
    if c5_vr1_enabled:
        prevNmax = g_code[volume_col].shift(1).rolling(c5_recent_n, min_periods=1).max()
        d["c5_prevNmax"] = prevNmax
        if c5_vr1_max is None:
            c5_vr1_ok = pd.Series(True, index=d.index)
        else:
            c5_vr1_ok = (d[volume_col] <= (float(c5_vr1_max) * prevNmax))
    else:
        d["c5_prevNmax"] = np.nan
        c5_vr1_ok = pd.Series(True, index=d.index)
    d["c5_vr1_ok"] = c5_vr1_ok

    if has_new_flags:
        # 新式：仅 AND 已启用的子模块
        parts = []
        if c5_vr1_enabled: parts.append(c5_vr1_ok)
        if c5_ratio_enabled: parts.append(c5_ratio_ok)
        if c5_vol_cmp_enabled: parts.append(c5_vol_cmp_ok)
        if c5_vma_rel_enabled: parts.append(c5_vma_rel_ok)
        if c5_vol_down_enabled: parts.append(c5_down_ok)
        cond5_combined = parts[0] if parts else pd.Series(True, index=d.index)
        for p in parts[1:]:
            cond5_combined = cond5_combined & p
        d["cond5_ok"] = np.where(c5_enabled_dry, cond5_combined, True)
        d["dryness_recent_cnt_v2"] = np.nan
        # 期望/实际文本（简化描述）
        d["cond5_expect"] = pd.Series(
            ("干缩子模块：" +
             (f"比例≤{c5_dryness_ratio_max};" if c5_ratio_enabled else "") +
             (f"量≤VMA{c5_vol_cmp_long};" if c5_vol_cmp_enabled else "") +
             (f"近{c5_vol_decreasing_days}日量连降;" if c5_vol_down_enabled else "")), index=d.index)
        d["cond5_actual"] = ("ratio_ok=" + c5_ratio_ok.astype(bool).astype(str)
                             + ", vol≤vma_ok=" + c5_vol_cmp_ok.astype(bool).astype(str)
                             + ", vol_down_ok=" + c5_down_ok.astype(bool).astype(str))
    else:
        # 旧式：保持原有三者同时要求 + 近窗干缩次数
        cond5_ratio_ok = d["dryness_ratio"] <= float(c5_dryness_ratio_max)
        cond5_vol_ok = (d[volume_col] <= vma_long_cmp) if bool(
            _c5_dry.get("require_vol_le_vma10", require_vol_le_vma10)) else pd.Series(True, index=d.index)
        if c5_dryness_recent_window and c5_dryness_recent_min_days:
            d["_dry_flag_tmp"] = (cond5_ratio_ok & cond5_vol_ok).astype("int8")
            d["dryness_recent_cnt_v2"] = g_code["_dry_flag_tmp"].transform(
                lambda s: s.rolling(int(c5_dryness_recent_window), min_periods=1).sum()
            )
            cond5_recent_ok = d["dryness_recent_cnt_v2"] >= int(c5_dryness_recent_min_days)
            d.drop(columns=["_dry_flag_tmp"], inplace=True)
        else:
            d["dryness_recent_cnt_v2"] = np.nan
            cond5_recent_ok = pd.Series(True, index=d.index)
        d["cond5_ratio_ok"] = cond5_ratio_ok
        d["cond5_vol_le_vma10_ok"] = cond5_vol_ok
        d["cond5_recent_ok"] = cond5_recent_ok
        cond5_all = cond5_ratio_ok & cond5_vol_ok & cond5_recent_ok
        d["cond5_ok"] = np.where(c5_enabled_dry, cond5_all, True)
        # 期望/实际
        d["cond5_expect"] = (
                pd.Series(f"短期量均/长期量均≤{c5_dryness_ratio_max}", index=d.index)
                + ("" if bool(
            _c5_dry.get("require_vol_le_vma10", require_vol_le_vma10)) is False else f"；量≤VMA{c5_vol_cmp_long}")
                + ("" if not (c5_dryness_recent_window and c5_dryness_recent_min_days)
                   else f"；近{c5_dryness_recent_window}日干缩≥{c5_dryness_recent_min_days}日")
        )
        # 实际文本
        ratio_str = d["dryness_ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
        vol_part = ("" if bool(_c5_dry.get("require_vol_le_vma10", require_vol_le_vma10)) is False else (
                "，vol≤vma_long_cmp=" + cond5_vol_ok.astype(bool).astype(str)))
        win_part = ("" if not (c5_dryness_recent_window and c5_dryness_recent_min_days)
                    else ("，近窗干缩日=" + d["dryness_recent_cnt_v2"].fillna(0).astype(int).astype(str)))
        d["cond5_actual"] = "ratio=" + ratio_str + vol_part + win_part

    # ============================================================
    # 条件6：价稳（仅使用 cond6 参数）
    # ============================================================
    _c6 = cond6 or {}
    c6_enabled = bool(_c6.get("enabled", enable_cond6))
    c6_price_stable_mode = str(_c6.get("price_stable_mode", price_stable_mode))
    c6_max_drop_ratio = float(_c6.get("max_drop_ratio", max_drop_ratio))
    c6_use_atr_window = int(_c6.get("use_atr_window", use_atr_window))
    c6_atr_buffer = float(_c6.get("atr_buffer", atr_buffer))

    d["cond6_enabled"] = bool(c6_enabled)

    prev_close = g_code[close_col].shift(1)
    prev_min = d["low_prev_min"]

    if c6_price_stable_mode == "no_new_low":
        cond6_price_stable_ok = d[low_col] >= prev_min
        d["cond6_metric"] = (d[low_col] - prev_min)
        d["cond6_metric_name"] = "low - prev_min_low"
        d["cond6_expect"] = pd.Series("未创新低（low ≥ 段内前低）", index=d.index)

        low_s = d[low_col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        prev_s = prev_min.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        diff_s = d["cond6_metric"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        d["cond6_actual"] = "low=" + low_s + ", prev_min=" + prev_s + ", diff=" + diff_s

    elif c6_price_stable_mode == "ratio":
        thresh = (1.0 - float(c6_max_drop_ratio)) * prev_min
        cond6_price_stable_ok = d[low_col] >= thresh
        d["cond6_metric"] = (d[low_col] / prev_min) - (1.0 - float(c6_max_drop_ratio))
        d["cond6_metric_name"] = f"low/prev_min - (1-{c6_max_drop_ratio:.0%})"
        d["cond6_expect"] = pd.Series(f"low ≥ prev_min × (1-{c6_max_drop_ratio:.0%})", index=d.index)

        ratio_val = (d[low_col] / prev_min)
        ratio_s = ratio_val.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        thr_s = thresh.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        mar_s = d["cond6_metric"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        d["cond6_actual"] = "low/prev_min=" + ratio_s + ", thresh=" + thr_s + ", margin=" + mar_s

    elif c6_price_stable_mode == "atr":
        tr = pd.concat([
            (d[high_col] - d[low_col]).abs(),
            (d[high_col] - prev_close).abs(),
            (d[low_col] - prev_close).abs()
        ], axis=1).max(axis=1)
        d["atr"] = g_code[tr.name].transform(
            lambda s: s.rolling(int(c6_use_atr_window), min_periods=int(c6_use_atr_window)).mean())
        thresh = prev_min - float(c6_atr_buffer) * d["atr"]
        cond6_price_stable_ok = d[low_col] >= thresh
        d["cond6_metric"] = d[low_col] - thresh
        d["cond6_metric_name"] = f"low - (prev_min - {c6_atr_buffer}*ATR{c6_use_atr_window})"
        d["cond6_expect"] = pd.Series(f"low ≥ prev_min - {c6_atr_buffer}×ATR{c6_use_atr_window}", index=d.index)

        low_s = d[low_col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        thr_s = thresh.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        mar_s = d["cond6_metric"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        d["cond6_actual"] = "low=" + low_s + ", thresh=" + thr_s + ", margin=" + mar_s

    else:
        raise ValueError("price_stable_mode 只能是 'no_new_low' | 'ratio' | 'atr'。")

    d["cond6_price_stable_ok"] = cond6_price_stable_ok
    d["cond6_ok"] = np.where(c6_enabled, cond6_price_stable_ok, True)

    # ============================================================
    # 总体判定：仅计入“已启用”的条件
    # ============================================================
    overall_ok = (
            (~d["cond1_enabled"] | d["cond1_ok"]) &
            (~d["cond2_enabled"] | d["cond2_ok"]) &
            (~d["cond3_enabled"] | d["cond3_ok"]) &
            (~d["cond4_enabled"] | d["cond4_ok"]) &
            (~d["cond5_enabled"] | d["cond5_ok"]) &
            (~d["cond6_enabled"] | d["cond6_ok"])
    )
    # 条件1额外控制：是否允许一个A点对应多个B点（默认允许）
    c1_allow_multi = bool(_c1.get("allow_multi_b_per_a", allow_multi_b_per_a))
    if c1_allow_multi:
        d["B_point"] = overall_ok.astype(bool)
    else:
        pre = overall_ok.astype(bool)
        d["_pre_b"] = pre.astype("int8")
        gseg = d.groupby([code_col, "seg_id"], sort=False)
        csum = gseg["_pre_b"].transform(lambda s: s.cumsum())
        d["B_point"] = (pre & (csum == 1)).astype(bool)

    # ============================================================
    # 文本解释（可关）
    # ============================================================
    if with_explain_strings:
        def _desc(row):
            if not row["B_point"]:
                return None
            dt = pd.to_datetime(row[date_col]).strftime("%Y年%m月%d日")
            parts = [
                f"条件1: {row['cond1_expect']}（实际：{row['cond1_actual']}）",
                f"条件2: {row['cond2_expect']}（实际：{row['cond2_actual']}）",
                f"条件3: {row['cond3_expect']}（实际：{row['cond3_actual']}）",
                f"条件4: {row['cond4_expect']}（实际：{row['cond4_actual']}）",
                f"条件5: {row['cond5_expect']}（实际：{row['cond5_actual']}）",
                f"条件6: {row['cond6_expect']}（实际：{row['cond6_actual']}）",
            ]
            return f"{row[code_col]} 在{dt}为B点，满足：" + "；".join(parts)

        def _status(row):
            met, unmet = [], []

            def mark(enabled, ok, name):
                if not enabled:
                    met.append(f"{name}(未启用)")
                else:
                    (met if ok else unmet).append(name)

            mark(bool(row["cond1_enabled"]), bool(row["cond1_ok"]), "条件1")
            mark(bool(row["cond2_enabled"]), bool(row["cond2_ok"]), "条件2")
            mark(bool(row["cond3_enabled"]), bool(row["cond3_ok"]), "条件3")
            mark(bool(row["cond4_enabled"]), bool(row["cond4_ok"]), "条件4")
            mark(bool(row["cond5_enabled"]), bool(row["cond5_ok"]), "条件5")
            mark(bool(row["cond6_enabled"]), bool(row["cond6_ok"]), "条件6")
            return "满足: " + "、".join(met) + "；不满足: " + ("、".join(unmet) if unmet else "无")

        d["B_point_desc"] = d.apply(_desc, axis=1)
        d["B_point_status"] = d.apply(_status, axis=1)

    # —— 清理临时列 ——
    d.drop(columns=[c for c in ["_seg_cum", "_maN_not_above", "_maN_above_int"] if c in d.columns],
           inplace=True, errors="ignore")

    return d


def find_c_points(
        df: pd.DataFrame,
        *,
        code_col: str = "code",
        date_col: str = "date",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        # 条件1：时间（距最近B点不超过A日）
        enable_cond1: bool = True,
        max_days_from_b: int = 60,
        # 条件2：放量（当日量 ≥ X × 前N日最大量）
        enable_cond2: bool = True,
        # 模块1：VR1 放量（默认启用）
        recent_n: int = 10,
        vol_multiple: float = 2.0,
        # 模块2：量均线上方比较（默认关闭）
        vma_short_days: int = 5,
        vma_long_days: int = 10,
        # 模块3：连涨量（默认关闭）
        vol_increasing_days: int = 3,
        # 条件3：价格在Y日均线上方（默认用最高价）
        enable_cond3: bool = True,
        price_field: str = "high",  # 'close' | 'high' | 'low'
        ma_days: int = 60,
        relation: str = "ge",  # 'ge' or 'gt'
        with_explain_strings: bool = False,
        cond1: Optional[dict] = None,
        cond2: Optional[dict] = None,
        cond3: Optional[dict] = None,
) -> pd.DataFrame:
    """
    在已包含 A_point/B_point 的数据上查找 C 点。

    - cond1: 距最近 B 点的天数 ≤ max_days_from_b。
    - cond2: 当日成交量 ≥ vol_multiple × 过去 recent_n 日（不含当日）的最大成交量。
    - cond3: 当日 price_field 在 ma_days 日均线之上（>= 或 >）。
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d.sort_values([code_col, date_col], inplace=True)
    g_code = d.groupby(code_col, sort=False)

    # —— 最近B点（按代码维度，全局最近；可跨 A 段复用，直到出现新的 B） ——
    if "B_point" not in d.columns:
        d["B_point"] = False
    d["_idx_in_code"] = g_code.cumcount()
    b_idx_series = d["_idx_in_code"].where(d["B_point"], np.nan)
    d["last_B_idx"] = b_idx_series.groupby(d[code_col]).ffill()
    d["days_since_B"] = d["_idx_in_code"] - d["last_B_idx"]

    # 条件1：时间窗口
    _c1 = cond1 or {}
    c1_enabled = bool(_c1.get("enabled", enable_cond1))
    c1_max_days_from_b = int(_c1.get("max_days_from_b", max_days_from_b))
    cond1_ok = (pd.notna(d["last_B_idx"]) & (d["days_since_B"] >= 1) & (d["days_since_B"] <= c1_max_days_from_b))
    d["cond1_ok"] = np.where(c1_enabled, cond1_ok, True)

    # 条件2：放量
    _c2 = cond2 or {}
    c2_enabled = bool(_c2.get("enabled", enable_cond2))
    # 子模块开关
    vr1_enabled = bool(_c2.get("vr1_enabled", True))
    vma_cmp_enabled = bool(_c2.get("vma_cmp_enabled", False))
    vol_up_enabled = bool(_c2.get("vol_up_enabled", False))
    # 模块1：VR1 放量
    c2_recent_n = int(_c2.get("recent_n", recent_n))
    c2_multiple = float(_c2.get("vol_multiple", vol_multiple))
    prevNmax = g_code[volume_col].shift(1).rolling(c2_recent_n, min_periods=1).max()
    d["prevNmax_vol"] = prevNmax
    d["vol_ratio_vs_prevNmax"] = d[volume_col] / prevNmax
    c2_vr1_ok = (d[volume_col] >= (c2_multiple * prevNmax)) if vr1_enabled else pd.Series(True, index=d.index)
    d["c2_vr1_ok"] = c2_vr1_ok
    # 模块2：量均线比较（短在长上）
    c2_vma_short = int(_c2.get("vma_short_days", vma_short_days))
    c2_vma_long = int(_c2.get("vma_long_days", vma_long_days))
    d["vma_short"] = g_code[volume_col].transform(lambda s: s.rolling(c2_vma_short, min_periods=c2_vma_short).mean())
    d["vma_long"] = g_code[volume_col].transform(lambda s: s.rolling(c2_vma_long, min_periods=c2_vma_long).mean())
    c2_vma_ok = (d["vma_short"] > d["vma_long"]) if vma_cmp_enabled else pd.Series(True, index=d.index)
    d["c2_vma_ok"] = c2_vma_ok
    # 模块3：前X日(含当日)成交量递增
    inc_days = int(_c2.get("vol_increasing_days", vol_increasing_days))
    if inc_days <= 1:
        c2_up_ok = pd.Series(True, index=d.index)
    else:
        # 严格递增：要求最近 inc_days 天体现在 (inc_days-1) 次相邻差分均 > 0
        win = int(max(1, inc_days - 1))
        inc = g_code[volume_col].diff(1) > 0  # Series[bool]，按代码分组计算
        # 在每个代码分组内，对 inc 做 rolling(win) 并要求“全部为 True”
        c2_up_ok = inc.groupby(d[code_col]).transform(lambda s: s.rolling(win, min_periods=win).sum() == win)
        c2_up_ok = c2_up_ok.fillna(False)
    d["c2_up_ok"] = (c2_up_ok if vol_up_enabled else pd.Series(True, index=d.index))
    # cond2 汇总：仅对启用的子模块取与
    enabled_mask = (
            (vr1_enabled & True) | (vma_cmp_enabled & True) | (vol_up_enabled & True)
    )
    if not enabled_mask:
        cond2_ok = pd.Series(True, index=d.index)
    else:
        parts = []
        if vr1_enabled: parts.append(d["c2_vr1_ok"])
        if vma_cmp_enabled: parts.append(d["c2_vma_ok"])
        if vol_up_enabled: parts.append(d["c2_up_ok"])
        cond2_ok = parts[0]
        for p in parts[1:]:
            cond2_ok = cond2_ok & p
    d["cond2_ok"] = np.where(c2_enabled, cond2_ok, True)

    # 条件3：价格与均线
    _c3 = cond3 or {}
    c3_enabled = bool(_c3.get("enabled", enable_cond3))
    c3_field = str(_c3.get("price_field", price_field))
    c3_ma_days = int(_c3.get("ma_days", ma_days))
    c3_rel = str(_c3.get("relation", relation))
    maY = g_code[close_col].transform(lambda s: s.rolling(c3_ma_days, min_periods=c3_ma_days).mean())
    d["maY"] = maY
    price_map = {"close": close_col, "high": high_col, "low": low_col}
    cmp_col = price_map.get(c3_field, high_col)
    if c3_rel == "gt":
        cond3_ok = d[cmp_col] > maY
    else:
        cond3_ok = d[cmp_col] >= maY
    d["cond3_ok"] = np.where(c3_enabled, cond3_ok, True)

    # 综合 C 点
    d["C_point"] = d["cond1_ok"] & d["cond2_ok"] & d["cond3_ok"]

    if with_explain_strings:
        d["cond1_expect"] = f"距最近B点 ≤ {c1_max_days_from_b}日"
        d["cond1_actual"] = ("days_since_B=" + d["days_since_B"].fillna(-1).astype(int).astype(str))
        d["cond2_expect"] = f"volume ≥ {c2_multiple} × 前{c2_recent_n}日最大量"
        d["cond2_actual"] = ("ratio=" + d["vol_ratio_vs_prevNmax"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan"))
        d["cond3_expect"] = f"{c3_field} {'>' if c3_rel == 'gt' else '≥'} MA{c3_ma_days}"
        d["cond3_actual"] = (
                f"{c3_field}=" + d[cmp_col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan") +
                ", ma=" + maY.map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
        )

    # 清理临时列
    d.drop(columns=[c for c in ["_idx_in_seg", "_idx_in_code"] if c in d.columns], inplace=True, errors="ignore")
    return d


if __name__ == '__main__':
    df = pd.read_parquet("/Users/chenjunming/Desktop/KevinGit/PyFinance/WF/backend/data/000001.parquet")
    print(df)
