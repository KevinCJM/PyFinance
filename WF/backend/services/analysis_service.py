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
        required_shorts: Optional[Iterable[int]] = None,  # 必须满足的短均线集合；None=等于 short_windows
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
        enable_cond1: bool = True,
        enable_cond2: bool = True,
        enable_cond3: bool = True,
        enable_cond4: bool = False,

        # ① 时间与“在 MA60 上”：改为 MA_N vs MA60
        min_days_from_a: int = 60,  # 从A点开始的最小天数，默认为60天
        enable_above_maN_rule: bool = True,  # 是否启用高于均线N的规则，默认为True
        above_maN_window: int = 5,  # 高于均线N的计算窗口期，默认为5日均线
        above_maN_days: int = 15,  # 高于均线N的天数要求，默认为30天
        above_maN_consecutive: bool = False,  # 是否要求高于均线N的天数必须连续，默认为False
        max_maN_below_days: int = 5,  # 均线N以下的最大天数，默认为5天

        # ② 触及/击破 MA60 + “缩量下行”的价格侧
        touch_price: str = "low",  # "low" 或 "close"
        touch_relation: str = "le",  # "le" 或 "lt"
        require_bearish: bool = False,  # 是否要求阴线
        require_close_le_prev: bool = False,  # 是否要求收盘价 ≤ 昨收
        vr1_max: Optional[float] = None,

        # ③ 干缩
        dryness_ratio_max: float = 0.8,  # VMA5/VMA10 ≤ 阈值
        require_vol_le_vma10: bool = True,
        dryness_recent_window: int = 0,  # 0 = 不启用
        dryness_recent_min_days: int = 0,

        # ④ 价稳
        price_stable_mode: str = "no_new_low",  # "no_new_low" | "ratio" | "atr"
        max_drop_ratio: float = 0.0,
        use_atr_window: int = 14,
        atr_buffer: float = 0.5,

        # 文本说明
        with_explain_strings: bool = True,
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

    # —— 成交量均线 ——
    if "vma_5" not in d.columns:
        d["vma_5"] = g_code[volume_col].transform(lambda s: s.rolling(5, min_periods=5).mean())
    if "vma_10" not in d.columns:
        d["vma_10"] = g_code[volume_col].transform(lambda s: s.rolling(10, min_periods=10).mean())

    # ============================================================
    # 条件1：A→B ≥ N 日 + （可选）MA_N 在 MA60 上
    # ============================================================
    d["cond1_enabled"] = bool(enable_cond1)

    cond1_time_ok = (d["seg_id"] > 0) & (d["days_since_A"] >= int(min_days_from_a))

    if enable_above_maN_rule and enable_cond1:
        maN_col = f"ma_{int(above_maN_window)}"
        if maN_col not in d.columns:
            d[maN_col] = g_code[close_col].transform(
                lambda s: s.rolling(int(above_maN_window), min_periods=int(above_maN_window)).mean()
            )
        maN_above = d[maN_col] >= d[ma60_col]
        g_seg = d.groupby([code_col, "seg_id"], sort=False)

        if above_maN_consecutive:
            d["_maN_not_above"] = (~maN_above).astype("int8")
            grp_break = g_seg["_maN_not_above"].transform(lambda s: s.cumsum())
            consec = maN_above.groupby([d[code_col], d["seg_id"], grp_break]).cumcount() + 1
            consec = pd.Series(np.where(maN_above.values, consec, 0), index=d.index)
            d["maN_consec_len_prev"] = consec.shift(1).fillna(0).astype(int)
            cond1_maN_ok = d["maN_consec_len_prev"] >= int(above_maN_days)
            maN_expect_str = f"MA{above_maN_window}在MA60上连续≥{above_maN_days}日（A..t-1）"
            maN_actual_text = "MA{}在MA60上(连续)=".format(above_maN_window) + \
                              d["maN_consec_len_prev"].astype(int).astype(str) + "天"
        else:
            d["_maN_above_int"] = maN_above.astype("int8")
            above_cum = g_seg["_maN_above_int"].transform(lambda s: s.cumsum())
            above_at_A = g_seg["_maN_above_int"].transform("first")
            d["maN_above_since_A_tminus1"] = (above_cum.shift(1).fillna(0) - above_at_A.fillna(0))
            span_len = (d["days_since_A"] - 1).clip(lower=0)
            d["maN_below_since_A_tminus1"] = (span_len - d["maN_above_since_A_tminus1"]).clip(lower=0)
            cond1_maN_ok = (d["maN_above_since_A_tminus1"] >= int(above_maN_days)) & \
                           (d["maN_below_since_A_tminus1"] <= int(max_maN_below_days))
            maN_expect_str = f"MA{above_maN_window}在MA60上累计≥{above_maN_days}日且下方≤{max_maN_below_days}日（A..t-1）"
            maN_actual_text = ("MA{}在MA60上(累计)=".format(above_maN_window)
                               + d["maN_above_since_A_tminus1"].fillna(0).astype(int).astype(str)
                               + "天, 下方=" + d["maN_below_since_A_tminus1"].fillna(0).astype(int).astype(str) + "天")
    else:
        cond1_maN_ok = pd.Series(True, index=d.index)
        maN_expect_str = "（MA_N规则未启用）"
        maN_actual_text = pd.Series("（未启用）", index=d.index)

    d["cond1_time_ok"] = cond1_time_ok
    d["cond1_maN_rule_ok"] = cond1_maN_ok
    cond1_ok_series = (cond1_time_ok & cond1_maN_ok)
    d["cond1_ok"] = np.where(enable_cond1, cond1_ok_series, True)

    # 期望/实际（向量化）
    d["cond1_expect"] = pd.Series(f"A→B≥{min_days_from_a}日；{maN_expect_str}", index=d.index)
    d["cond1_actual"] = ("seg_id=" + d["seg_id"].astype("Int64").astype(str)
                         + ", days_since_A=" + d["days_since_A"].astype("Int64").astype(str)
                         + "；" + (maN_actual_text if isinstance(maN_actual_text, pd.Series)
                                  else pd.Series(maN_actual_text, index=d.index)))

    # ============================================================
    # 条件2：阴线/收≤昨收 + 触及/击破 MA60 +（可选 VR1）
    # ============================================================
    d["cond2_enabled"] = bool(enable_cond2)
    prev_close = g_code[close_col].shift(1)

    cond2_bearish_ok = (d[close_col] < d[open_col]) if require_bearish else pd.Series(True, index=d.index)
    cond2_close_le_prev_ok = (d[close_col] <= prev_close) if require_close_le_prev else pd.Series(True, index=d.index)

    cmp_left = d[low_col] if touch_price == "low" else d[close_col]
    cond2_touch_ok = (cmp_left <= d[ma60_col]) if touch_relation == "le" else (cmp_left < d[ma60_col])

    cond2_vr1_ok = ((d["vr1"] <= float(vr1_max)) if (vr1_max is not None and "vr1" in d.columns)
                    else pd.Series(True, index=d.index))

    d["cond2_bearish_ok"] = cond2_bearish_ok
    d["cond2_close_le_prev_ok"] = cond2_close_le_prev_ok
    d["cond2_touch_ok"] = cond2_touch_ok
    d["cond2_vr1_ok"] = cond2_vr1_ok

    cond2_all = cond2_bearish_ok & cond2_close_le_prev_ok & cond2_touch_ok & cond2_vr1_ok
    d["cond2_ok"] = np.where(enable_cond2, cond2_all, True)

    _bear_txt = "阴线" if require_bearish else "不强制阴线"
    _prev_txt = "收≤昨收" if require_close_le_prev else "不强制收≤昨收"
    _touch_txt = f"{touch_price}{'≤' if touch_relation == 'le' else '<'}MA60"
    _vr_txt = (f" 且 VR1≤{vr1_max}" if (vr1_max is not None and "vr1" in d.columns) else "")
    d["cond2_expect"] = pd.Series(f"{_bear_txt}；{_prev_txt}；{_touch_txt}{_vr_txt}", index=d.index)

    d["cond2_actual"] = ("bearish=" + cond2_bearish_ok.astype(bool).astype(str)
                         + ", close≤prev=" + cond2_close_le_prev_ok.astype(bool).astype(str)
                         + ", " + _touch_txt + "=" + cond2_touch_ok.astype(bool).astype(str)
                         + ", vr1_ok=" + cond2_vr1_ok.astype(bool).astype(str))

    # ============================================================
    # 条件3：干缩
    # ============================================================
    d["cond3_enabled"] = bool(enable_cond3)

    d["dryness_ratio"] = d["vma_5"] / d["vma_10"]
    cond3_ratio_ok = d["dryness_ratio"] <= float(dryness_ratio_max)
    cond3_vol_ok = (d[volume_col] <= d["vma_10"]) if require_vol_le_vma10 else pd.Series(True, index=d.index)

    if dryness_recent_window and dryness_recent_min_days:
        d["_dry_flag_tmp"] = (cond3_ratio_ok & cond3_vol_ok).astype("int8")
        d["dryness_recent_cnt_v2"] = g_code["_dry_flag_tmp"].transform(
            lambda s: s.rolling(int(dryness_recent_window), min_periods=1).sum()
        )
        cond3_recent_ok = d["dryness_recent_cnt_v2"] >= int(dryness_recent_min_days)
        d.drop(columns=["_dry_flag_tmp"], inplace=True)
    else:
        d["dryness_recent_cnt_v2"] = np.nan
        cond3_recent_ok = pd.Series(True, index=d.index)

    d["cond3_ratio_ok"] = cond3_ratio_ok
    d["cond3_vol_le_vma10_ok"] = cond3_vol_ok
    d["cond3_recent_ok"] = cond3_recent_ok
    cond3_all = cond3_ratio_ok & cond3_vol_ok & cond3_recent_ok
    d["cond3_ok"] = np.where(enable_cond3, cond3_all, True)

    # 期望文本
    d["cond3_expect"] = (
            pd.Series(f"VMA5/VMA10≤{dryness_ratio_max}", index=d.index)
            + ("" if not require_vol_le_vma10 else "；量≤VMA10")
            + ("" if not (dryness_recent_window and dryness_recent_min_days)
               else f"；近{dryness_recent_window}日干缩≥{dryness_recent_min_days}日")
    )

    # 实际文本（逐元素格式化）
    ratio_str = d["dryness_ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
    vol_part = ("" if not require_vol_le_vma10 else ("，vol≤vma10=" + cond3_vol_ok.astype(bool).astype(str)))
    win_part = ("" if not (dryness_recent_window and dryness_recent_min_days)
                else ("，近窗干缩日=" + d["dryness_recent_cnt_v2"].fillna(0).astype(int).astype(str)))
    d["cond3_actual"] = "ratio=" + ratio_str + vol_part + win_part

    # ============================================================
    # 条件4：价稳
    # ============================================================
    d["cond4_enabled"] = bool(enable_cond4)

    prev_close = g_code[close_col].shift(1)
    prev_min = d["low_prev_min"]

    if price_stable_mode == "no_new_low":
        cond4_price_stable_ok = d[low_col] >= prev_min
        d["cond4_metric"] = (d[low_col] - prev_min)
        d["cond4_metric_name"] = "low - prev_min_low"
        d["cond4_expect"] = pd.Series("未创新低（low ≥ 段内前低）", index=d.index)

        low_s = d[low_col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        prev_s = prev_min.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        diff_s = d["cond4_metric"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        d["cond4_actual"] = "low=" + low_s + ", prev_min=" + prev_s + ", diff=" + diff_s

    elif price_stable_mode == "ratio":
        thresh = (1.0 - float(max_drop_ratio)) * prev_min
        cond4_price_stable_ok = d[low_col] >= thresh
        d["cond4_metric"] = (d[low_col] / prev_min) - (1.0 - float(max_drop_ratio))
        d["cond4_metric_name"] = f"low/prev_min - (1-{max_drop_ratio:.0%})"
        d["cond4_expect"] = pd.Series(f"low ≥ prev_min × (1-{max_drop_ratio:.0%})", index=d.index)

        ratio_val = (d[low_col] / prev_min)
        ratio_s = ratio_val.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        thr_s = thresh.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        mar_s = d["cond4_metric"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        d["cond4_actual"] = "low/prev_min=" + ratio_s + ", thresh=" + thr_s + ", margin=" + mar_s

    elif price_stable_mode == "atr":
        tr = pd.concat([
            (d[high_col] - d[low_col]).abs(),
            (d[high_col] - prev_close).abs(),
            (d[low_col] - prev_close).abs()
        ], axis=1).max(axis=1)
        d["atr"] = g_code[tr.name].transform(
            lambda s: s.rolling(int(use_atr_window), min_periods=int(use_atr_window)).mean())
        thresh = prev_min - float(atr_buffer) * d["atr"]
        cond4_price_stable_ok = d[low_col] >= thresh
        d["cond4_metric"] = d[low_col] - thresh
        d["cond4_metric_name"] = f"low - (prev_min - {atr_buffer}*ATR{use_atr_window})"
        d["cond4_expect"] = pd.Series(f"low ≥ prev_min - {atr_buffer}×ATR{use_atr_window}", index=d.index)

        low_s = d[low_col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        thr_s = thresh.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        mar_s = d["cond4_metric"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        d["cond4_actual"] = "low=" + low_s + ", thresh=" + thr_s + ", margin=" + mar_s

    else:
        raise ValueError("price_stable_mode 只能是 'no_new_low' | 'ratio' | 'atr'。")

    d["cond4_price_stable_ok"] = cond4_price_stable_ok
    d["cond4_ok"] = np.where(enable_cond4, cond4_price_stable_ok, True)

    # ============================================================
    # 总体判定：仅计入“已启用”的条件
    # ============================================================
    overall_ok = (
            (~d["cond1_enabled"] | d["cond1_ok"]) &
            (~d["cond2_enabled"] | d["cond2_ok"]) &
            (~d["cond3_enabled"] | d["cond3_ok"]) &
            (~d["cond4_enabled"] | d["cond4_ok"])
    )
    d["B_point"] = overall_ok.astype(bool)

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
            return "满足: " + "、".join(met) + "；不满足: " + ("、".join(unmet) if unmet else "无")

        d["B_point_desc"] = d.apply(_desc, axis=1)
        d["B_point_status"] = d.apply(_status, axis=1)

    # —— 清理临时列 ——
    d.drop(columns=[c for c in ["_seg_cum", "_maN_not_above", "_maN_above_int"] if c in d.columns],
           inplace=True, errors="ignore")

    return d
