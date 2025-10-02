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

        # ==== 其他可参数化选项 ====
        ma_full_window: bool = True,  # 均线 min_periods：True=等于窗口；False=1
        cross_requires_prev_below: bool = True,  # 当日上穿是否必须“前一日不在上方”
        confirm_min_crosses: int = 1,  # 条件3 需要的最少上穿次数
        vr1_lookback: int = 10,  # VR1 的回看窗口长度（取 max 之前）
        eps: float = 0.0,  # 比较时的容忍度，例如 > 变为 > + eps
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
    if required_shorts is None:
        required_shorts = tuple(short_windows)
    else:
        required_shorts = tuple(sorted(set(required_shorts)))

    # ---------- 预处理 ----------
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d.sort_values([code_col, date_col], inplace=True)
    g = d.groupby(code_col, sort=False)

    # ---------- 均线 ----------
    needed_ma_windows = set(short_windows) | {long_window}
    if confirm_ma_window:
        needed_ma_windows.add(confirm_ma_window)
    for k in sorted(needed_ma_windows):
        minp = k if ma_full_window else 1
        d[f"ma_{k}"] = g[close_col].transform(lambda s: s.rolling(k, min_periods=minp).mean())

    ma_long = f"ma_{long_window}"

    # ---------- 条件1：长均线下跌 ----------
    d["cond1_enabled"] = bool(enable_cond1)

    maL_t1 = g[ma_long].shift(1)
    maL_t1_L = g[ma_long].shift(1 + down_lookback)
    cond1_ok_series = (maL_t1.notna() & maL_t1_L.notna()) & ((maL_t1 + eps) < maL_t1_L)
    d["ma_long_down"] = cond1_ok_series
    d["cond1_ok"] = np.where(enable_cond1, cond1_ok_series, True)

    # 期望/实际文本（逐元素格式化）
    d["cond1_expect"] = pd.Series(
        f"MA{long_window}(t-1) < MA{long_window}(t-1-{down_lookback})", index=d.index
    )
    ma_t1_str = maL_t1.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    ma_t1L_str = maL_t1_L.map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")

    # ✅ 用向量化字符串拼接，不再使用 .format 放 Series
    prefix1 = f"MA{long_window}(t-1)="
    prefix2 = f", MA{long_window}(t-1-{down_lookback})="
    d["cond1_actual"] = prefix1 + ma_t1_str + prefix2 + ma_t1L_str

    # ---------- 条件2：短均线上穿（仅当日触发） ----------
    d["cond2_enabled"] = bool(enable_cond2)

    # 为每条短均线生成 above_k / crossup_k / recent_cross_k
    for k in short_windows:
        ma_s = f"ma_{k}"
        above_col = f"above_{k}"
        cross_col = f"crossup_{k}"

        d[above_col] = (d[ma_s] + eps) > d[ma_long]
        if cross_requires_prev_below:
            prev_above = (g[ma_s].shift(1) + eps) > g[ma_long].shift(1)
            d[cross_col] = d[above_col] & (~prev_above.fillna(False))
        else:
            d[cross_col] = d[above_col]

    # 当日至少一条 required_shorts 发生上穿
    today_any_cross = None
    for k in required_shorts:
        c = d[f"crossup_{k}"]
        today_any_cross = c if today_any_cross is None else (today_any_cross | c)
    if today_any_cross is None:
        today_any_cross = pd.Series(False, index=d.index)

    # （仅在 require_all=True 时使用）最近 cross_window 内 required_shorts 是否“每条均已发生过上穿”
    if require_all:
        recent_all = None
        for k in required_shorts:
            rc = g[f"crossup_{k}"].transform(lambda s: s.rolling(cross_window, min_periods=1).max()) > 0
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

    d["cond2_ok"] = np.where(enable_cond2, cond2_core, True)

    # 期望/实际（条件2）
    req_list_str = "、".join([f"MA{k}" for k in required_shorts])
    if require_all:
        cond2_expect_str = (
            f"当日至少一条({req_list_str})上穿MA{long_window}；"
            f"最近{cross_window}日内上述每条均出现过上穿；"
            f"且当日这些均线均在MA{long_window}上方"
        )
    else:
        cond2_expect_str = f"当日至少一条({req_list_str})上穿MA{long_window}"

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
    d["cond3_enabled"] = bool(enable_cond3)

    if (confirm_lookback_days > 0) and (confirm_ma_window is not None) and enable_cond3:
        price_col = confirm_price_col
        if price_col not in d.columns:
            raise ValueError(f"confirm_price_col='{price_col}' 不在 df 列中。")
        ma_c = f"ma_{confirm_ma_window}"

        prev_below = g[price_col].shift(1) < g[ma_c].shift(1)
        cross_price_up = (d[price_col] >= d[ma_c]) & prev_below.fillna(False)
        d["_cross_price_up"] = cross_price_up.astype("int8")

        # 统计 [t-N, t-1] 内的上穿次数
        confirm_cnt = g["_cross_price_up"].transform(
            lambda s: s.shift(1).rolling(int(confirm_lookback_days), min_periods=int(confirm_lookback_days)).sum()
        )
        d["confirm_cross_cnt"] = confirm_cnt
        cond3_ok_series = (confirm_cnt >= int(confirm_min_crosses))
        cond3_expect_str = f"A点前{confirm_lookback_days}日内，{price_col}由下上穿MA{confirm_ma_window}≥{confirm_min_crosses}次（区间[t-N,t-1]）"
        # 实际：给出次数
        cnt_str = confirm_cnt.fillna(0).astype(int).astype(str)
        d["cond3_actual"] = "cross_cnt=" + cnt_str
    else:
        cond3_ok_series = pd.Series(True, index=d.index)  # 未启用则视为通过
        cond3_expect_str = "（未启用）"
        d["confirm_cross_cnt"] = np.nan
        d["_cross_price_up"] = np.nan
        d["cond3_actual"] = "（未启用）"

    d["cond3_ok"] = np.where(enable_cond3, cond3_ok_series, True)
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


def explain_ab_pairs(
        df_code: pd.DataFrame,
        *,
        code_col: str = "code",
        date_col: str = "date",
        volume_col: str = "volume",
        low_col: str = "low",
):
    """
    输入：单只股票的 DataFrame（建议已先后运行 find_a_points 与 find_b_points）。
    输出：按 A1/B1、A2/B2… 排列的中文说明列表（仅输出“有对应B点”的 A）。
    说明策略：
      - A端：优先读取 A_point_desc；若无则做最小兜底（基于 crossup_/above_/ma_long_down）。
      - B端：优先读取现有格式列 condX_enabled/ok/expect/actual；否则退回 B_point_desc（或最小兜底）。
    """
    d = df_code.copy()
    # 基础清洗
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.sort_values(date_col)
    if d[code_col].nunique() != 1:
        raise ValueError("explain_ab_pairs: df_code 必须只包含单一股票的数据。")
    code = str(d[code_col].iloc[0])

    # 分段：若无 seg_id，退化为按 A_point 累积
    if "seg_id" not in d.columns:
        if "A_point" not in d.columns:
            raise ValueError("缺少 seg_id 与 A_point，无法配对 A/B。")
        d["seg_id"] = d["A_point"].astype(bool).cumsum()

    # B 点列名（兼容旧/新）
    b_flag_col = "B_point" if "B_point" in d.columns else ("B_point_v2" if "B_point_v2" in d.columns else None)
    if b_flag_col is None:
        raise ValueError("未找到 B 点标记列（B_point 或 B_point_v2）。")

    out: List[str] = []
    pair_idx = 1

    # -------- helpers --------
    def _fmt_cond_line(row: pd.Series, i: int) -> Optional[str]:
        """
        尝试用“现有格式”输出：条件i: expect（实际：actual） —— 满足/未满足/未启用
        若缺列，返回 None。
        """
        en_key = f"cond{i}_enabled"
        ok_key = f"cond{i}_ok"
        ex_key = f"cond{i}_expect"
        ac_key = f"cond{i}_actual"
        if ex_key in row.index or ac_key in row.index or ok_key in row.index or en_key in row.index:
            enabled = bool(row.get(en_key, True))
            expect = row.get(ex_key, "（无期望）")
            actual = row.get(ac_key, "（无实际）")
            ok_val = row.get(ok_key, None)
            tail = ""
            if not enabled:
                tail = "（未启用）"
            elif ok_val is not None:
                tail = " —— 满足" if bool(ok_val) else " —— 未满足"
            return f"条件{i}: {expect}（实际：{actual}）{tail}"
        return None

    def _fmt_a_minimal(rowA: pd.Series) -> str:
        # 最小兜底：仅在 A_point_desc 缺失时使用，避免口径偏差
        dtA = pd.to_datetime(rowA[date_col]).strftime("%Y年%m月%d日")
        # 条件1
        c1 = "条件1: 前期MA60下跌" if bool(rowA.get("ma_long_down", False)) else "条件1: 前期MA60下跌 未满足"
        # 条件2：当日上穿的短均线列表（若存在）
        crossed = []
        for k in (5, 10, 20, 30, 60):
            col = f"crossup_{k}"
            if col in rowA.index and bool(rowA[col]):
                crossed.append(f"MA{k}")
        c2 = f"条件2: 当日短均线上穿MA60: " + ("、".join(crossed) if crossed else "（无）")
        # 条件3：当日位置
        pos = []
        for k in (5, 10, 20, 30, 60):
            col = f"above_{k}"
            if col in rowA.index:
                pos.append(f"MA{k}{'>' if bool(rowA[col]) else '<='}MA60")
        c3 = ("条件3: 当日位置 " + "，".join(pos)) if pos else None
        parts = [c1, c2] + ([c3] if c3 else [])
        return f"{code}股票在{dtA}, 有A{pair_idx}点. " + "；".join(parts)

    def _fmt_b_minimal(rowB: pd.Series) -> str:
        # 最小兜底：仅在 condX_* 与 B_point_desc 均缺失时使用
        dtB = pd.to_datetime(rowB[date_col]).strftime("%Y年%m月%d日")
        parts = []
        if "pos_below_ma60" in rowB.index:
            parts.append(f"条件1: 当日 MA5/MA10 均≤MA60({'满足' if bool(rowB['pos_below_ma60']) else '未满足'})")
        if "days_since_A" in rowB.index:
            parts.append(f"条件(时间): A→B={int(rowB['days_since_A'])}天")
        if "dryness_ratio" in rowB.index and "vma_10" in rowB.index:
            ratio = rowB["dryness_ratio"]
            vma10 = rowB["vma_10"]
            vol = rowB.get(volume_col, np.nan)
            parts.append(
                f"条件(干缩): ratio={ratio:.2f if pd.notna(ratio) else 'nan'}, 当日量≤VMA10={bool(vol <= vma10) if pd.notna(vma10) and pd.notna(vol) else 'nan'}")
        if "cond4_metric_name" in rowB.index and "cond4_metric" in rowB.index:
            parts.append(f"条件(价稳): {rowB['cond4_metric_name']} = {rowB['cond4_metric']}")
        return f"{code}股票在{dtB}, 有B{pair_idx}点. " + "；".join(parts)

    # -------- 遍历每个段，组装 A/B 说明 --------
    for seg, sub in d.groupby("seg_id", sort=True):
        if seg <= 0:
            continue
        # A 行
        if "A_point" not in sub.columns:
            continue
        a_mask = sub["A_point"].astype(bool).to_numpy()
        if not a_mask.any():
            continue
        posA = int(np.where(a_mask)[0][0])
        rowA = sub.iloc[posA]

        # B 行（A 之后的第一个 B）
        if b_flag_col not in sub.columns:
            continue
        b_idx_all = np.where(sub[b_flag_col].astype(bool).to_numpy())[0]
        b_idx_after = b_idx_all[b_idx_all > posA]
        if b_idx_after.size == 0:
            continue
        posB = int(b_idx_after[0])
        rowB = sub.iloc[posB]

        # ===== A 端输出 =====
        if "A_point_desc" in rowA.index and pd.notna(rowA["A_point_desc"]):
            out.append(str(rowA["A_point_desc"]).replace("为A点", f"有A{pair_idx}点"))
        else:
            out.append(_fmt_a_minimal(rowA))

        # ===== B 端输出 =====
        dtB = pd.to_datetime(rowB[date_col]).strftime("%Y年%m月%d日")

        # 优先：现有格式 condX_*（B 函数生成）
        have_cond = any(k in rowB.index for k in [
            "cond1_enabled", "cond1_ok", "cond1_expect", "cond1_actual",
            "cond2_enabled", "cond2_ok", "cond2_expect", "cond2_actual",
            "cond3_enabled", "cond3_ok", "cond3_expect", "cond3_actual",
            "cond4_enabled", "cond4_ok", "cond4_expect", "cond4_actual",
        ])
        if have_cond:
            lines = []
            for i in (1, 2, 3, 4):
                line = _fmt_cond_line(rowB, i)
                if line is not None:
                    lines.append(line)
            if lines:
                out.append(f"{code}股票在{dtB}, 有B{pair_idx}点. " + "；".join(lines))
                pair_idx += 1
                continue  # 已输出

        # 次优：已有描述列
        desc_col = "B_point_desc" if "B_point_desc" in rowB.index and pd.notna(rowB["B_point_desc"]) else \
            ("B_point_desc_v2" if "B_point_desc_v2" in rowB.index and pd.notna(rowB["B_point_desc_v2"]) else None)
        if desc_col:
            out.append(str(rowB[desc_col]).replace("为B点", f"有B{pair_idx}点"))
            pair_idx += 1
            continue

        # 兜底：最小字段拼出
        out.append(_fmt_b_minimal(rowB))
        pair_idx += 1

    return out


def plot_kline_brush_with_ab(
        df_code: pd.DataFrame,
        *,
        date_col: str = "date",
        code_col: str = "code",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        ma5_col: str = "ma_5",
        ma10_col: str = "ma_10",
        ma60_col: str = "ma_60",
        a_flag_col: str = "A_point",
        b_flag_col: str = "B_point",
        a_desc_col: Optional[str] = "A_point_desc",  # 若无则自动降级
        b_desc_col: Optional[str] = "B_point_desc",

        title: Optional[str] = None,
        width: str = "1200px",
        height: str = "780px",

        a_point_at: str = "low",  # "low" 或 "close"
        b_point_at: str = "low",  # "low" 或 "close"
) -> Grid:
    """
    仅画图。要求 df_code 为【单只股票】的 DataFrame，且已包含 A_point / B_point / seg_id 等字段。
    可选显示说明列 a_desc_col / b_desc_col（若存在）。
    """
    d = df_code.copy()
    # --- 基础清洗 ---
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, open_col, high_col, low_col, close_col]).sort_values(date_col)
    if d.empty:
        raise ValueError("输入数据为空或 OHLC 列存在大量 NaN。")

    # 单票检查（可按需移除）
    if d[code_col].nunique() != 1:
        raise ValueError("df_code 必须只包含单一股票的数据。")

    # 轴与基础数组
    x = d[date_col].dt.strftime("%Y-%m-%d").tolist()
    op = d[open_col].astype(float).to_numpy()
    hi = d[high_col].astype(float).to_numpy()
    lo = d[low_col].astype(float).to_numpy()
    cl = d[close_col].astype(float).to_numpy()
    vol = d[volume_col].astype(float).fillna(0.0).to_numpy()
    up_mask = (cl >= op)

    # ---------------- 主图（K线） ----------------
    kdata = d[[open_col, close_col, low_col, high_col]].astype(float).values.tolist()
    kline = (
        Kline(init_opts=opts.InitOpts(width=width, height=height))
        .add_xaxis(x)
        .add_yaxis(
            "K", kdata,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a", color0="#14b143",
                border_color="#ef232a", border_color0="#14b143",
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title or f"{d[code_col].iloc[0]} — K线/MA/量/A-B"),
            legend_opts=opts.LegendOpts(pos_top="1%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1], range_start=70, range_end=100),
                opts.DataZoomOpts(type_="slider", xaxis_index=[0, 1], pos_top="85%"),
            ],
            brush_opts=opts.BrushOpts(
                tool_box=["rect", "lineX", "lineY", "keep", "clear"],
                brush_type="lineX", brush_mode="single",
                x_axis_index="all", y_axis_index="all",
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category", boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
            ),
            yaxis_opts=opts.AxisOpts(type_="value", is_scale=True),  # 兼容旧版
        )
    )

    # ---------------- 均线（若存在才画） ----------------
    def _add_ma_line(name: str, y: pd.Series):
        if y.name in d.columns or isinstance(y, pd.Series):
            kline.overlap(
                Line()
                .add_xaxis(x)
                .add_yaxis(name, y.astype(float).round(4).tolist(),
                           is_smooth=True, is_symbol_show=False,
                           linestyle_opts=opts.LineStyleOpts(width=1.2, opacity=0.9))
            )

    if ma5_col in d.columns:  _add_ma_line("MA5", d[ma5_col])
    if ma10_col in d.columns: _add_ma_line("MA10", d[ma10_col])
    if ma60_col in d.columns: _add_ma_line("MA60", d[ma60_col])

    # ---------------- 只保留“有对应 B 点”的 A 点 ----------------
    if "seg_id" not in d.columns:
        d["seg_id"] = d[a_flag_col].cumsum()

    ab_pairs = []  # (idxA, idxB)
    for seg, sub in d.groupby("seg_id", sort=True):
        if seg <= 0:
            continue
        idxs = sub.index.to_numpy()
        maskA = sub[a_flag_col].astype(bool).to_numpy()
        maskB = sub[b_flag_col].astype(bool).to_numpy() if (b_flag_col in sub.columns) else np.zeros_like(maskA, bool)
        if not maskA.any() or not maskB.any():
            continue

        # A 的位置（段内第1个 A）
        posA_all = np.where(maskA)[0]
        posA = posA_all[0]

        # 选择“在 A 之后”的第一个 B（严格 >）
        posB_all = np.where(maskB)[0]
        posB_after = posB_all[posB_all > posA]
        if posB_after.size == 0:
            continue
        posB = posB_after[0]

        # 回到原始行索引
        iA = idxs[posA]
        iB = idxs[posB]
        ab_pairs.append((iA, iB))

    # 若无任何 A-B 配对，A/B 均不显示
    show_A = show_B = len(ab_pairs) > 0

    # ---------------- A/B 散点 ----------------
    if show_A:
        A_idx = [iA for (iA, _) in ab_pairs]
        A_x = d.loc[A_idx, date_col].dt.strftime("%Y-%m-%d").tolist()
        A_y = (d.loc[A_idx, low_col] * 0.995 if a_point_at == "low" else d.loc[A_idx, close_col]).astype(float).tolist()

        kline = kline.overlap(
            Scatter()
            .add_xaxis(A_x)
            .add_yaxis(
                "A点(配对)", A_y,
                symbol="triangle-up", symbol_size=14,
                itemstyle_opts=opts.ItemStyleOpts(color="#2563eb", border_color="#1e40af", border_width=1),
                label_opts=opts.LabelOpts(is_show=False),
                tooltip_opts=opts.TooltipOpts(formatter="A点<br/>{b}: {c}")
            )
        )

    if show_B:
        B_idx = [iB for (_, iB) in ab_pairs]
        B_x = d.loc[B_idx, date_col].dt.strftime("%Y-%m-%d").tolist()
        B_y = (d.loc[B_idx, low_col] * 0.995 if b_point_at == "low" else d.loc[B_idx, close_col]).astype(float).tolist()

        kline = kline.overlap(
            Scatter()
            .add_xaxis(B_x)
            .add_yaxis(
                "B点", B_y,
                symbol="triangle-down", symbol_size=14,
                itemstyle_opts=opts.ItemStyleOpts(color="#f59e0b", border_color="#b45309", border_width=1),
                label_opts=opts.LabelOpts(is_show=False),
                tooltip_opts=opts.TooltipOpts(formatter="B点<br/>{b}: {c}")
            )
        )

    # ---------------- A-B 虚线连接 ----------------
    # 用位置映射，避免 get_loc 在重复索引时的歧义
    pos_map = pd.Series(np.arange(len(d)), index=d.index)

    if ab_pairs:
        N = len(x)
        for iA, iB in ab_pairs:
            posA = int(pos_map[iA])
            posB = int(pos_map[iB])

            yA = float(d.at[iA, low_col] * 0.995 if a_point_at == "low" else d.at[iA, close_col])
            yB = float(d.at[iB, low_col] * 0.995 if b_point_at == "low" else d.at[iB, close_col])

            y_line = [None] * N
            y_line[posA] = yA
            y_line[posB] = yB

            kline = kline.overlap(
                Line()
                .add_xaxis(x)
                .add_yaxis(
                    "A-B", y_line,
                    is_symbol_show=False,
                    linestyle_opts=opts.LineStyleOpts(width=1.2, type_="dashed", opacity=0.9),
                )
            )

    # ---------------- 量能副图 ----------------
    bar_items = [
        opts.BarItem(
            name=x[i],
            value=float(vol[i]),
            itemstyle_opts=opts.ItemStyleOpts(color=("#ef232a" if up_mask[i] else "#14b143"))
        )
        for i in range(len(x))
    ]
    volume_bar = (
        Bar()
        .add_xaxis(x)
        .add_yaxis("Volume", bar_items, xaxis_index=1, yaxis_index=1, label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="category", grid_index=1),
            yaxis_opts=opts.AxisOpts(grid_index=1, type_="value", is_scale=True, split_number=2),
            legend_opts=opts.LegendOpts(pos_top="91%"),
        )
    )

    # ---------------- 拼接 Grid ----------------
    grid = Grid(init_opts=opts.InitOpts(width=width, height=height))
    grid.add(kline, grid_opts=opts.GridOpts(pos_left="5%", pos_right="3%", pos_top="8%", pos_bottom="28%"))
    grid.add(volume_bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="3%", pos_top="76%", pos_bottom="8%"))
    return grid


if __name__ == '__main__':
    from pathlib import Path
    code = "600121"
    data_dir = Path(__file__).resolve().parent.parent / "data"
    # 读取文件
    print("读取的 DataFrame：")
    stock_df = pd.read_parquet(data_dir / f'{code}.parquet')
    print(stock_df.tail())
    stock_df.rename(columns={"股票代码": "code", "日期": "date", "开盘": "open", "收盘": "close",
                             "最高": "high", "最低": "low", "成交量": "volume"}, inplace=True)
    stock_df = stock_df[["date", "code", "open", "high", "low", "close", "volume"]]
    print("转换后的 DataFrame：")
    print(stock_df.tail())

    # 找到 A 点
    stock_df = find_a_points(stock_df, short_windows=(5, 10), long_window=60,
                             required_shorts=(5,), require_all=True,
                             # A 点前 5 天内，最高价“上穿”5 日均线至少一次
                             confirm_lookback_days=5, confirm_ma_window=5, confirm_price_col="high"
                             )
    print(f"\n{code} 的 A 点：")
    print(stock_df)

    # 找到 B 点
    stock_df = find_b_points(stock_df)
    print(f"\n{code} 的 B 点：")
    print(stock_df)

    # df_code: 单只股票的 DataFrame，已包含 A_point/B_point/seg_id 等列（你的“现有字段”）
    grid = plot_kline_brush_with_ab(
        stock_df,
        title=f"{stock_df['code'].iloc[0]} — A/B配对",
        a_point_at="low",
        b_point_at="low",
    )
    grid.render("kline_ab_pairs.html")

    # 打印 A-B 配对的说明
    l = explain_ab_pairs(stock_df)
    print("\nA-B 配对说明：")
    for line in l:
        print(line)
