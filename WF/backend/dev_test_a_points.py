from __future__ import annotations

import pandas as pd
from pathlib import Path
from services.analysis_service import find_a_points


def main():
    data_path = Path(__file__).resolve().parent / "data" / "300008.parquet"
    if not data_path.exists():
        raise SystemExit(f"测试数据不存在: {data_path}")

    df_cn = pd.read_parquet(data_path)
    # 构造最小所需字段（与后端接口一致）
    d = pd.DataFrame({
        "code": "300008",
        "date": pd.to_datetime(df_cn["日期"], errors="coerce"),
        "close": pd.to_numeric(df_cn.get("收盘"), errors="coerce"),
        "high": pd.to_numeric(df_cn.get("最高"), errors="coerce"),
        "volume": pd.to_numeric(df_cn.get("成交量"), errors="coerce"),
    }).dropna(subset=["date"]).sort_values("date")

    # 条件参数（原始逻辑，独立控制）
    cond1 = {"enabled": True, "long_window": 60, "down_lookback": 30}
    cond2 = {"enabled": True, "short_windows": (5, 10), "long_window": 60, "cross_window": 3, "required_shorts": None, "require_all": True}
    cond3 = {"enabled": False, "confirm_lookback_days": 0, "confirm_ma_window": None, "confirm_price_col": "high"}

    out = find_a_points(
        d,
        code_col="code", date_col="date", close_col="close", volume_col="volume",
        with_explain_strings=False,
        cond1=cond1, cond2=cond2, cond3=cond3,
    )
    a = out[out["A_point"] == True]
    print(f"总行数: {len(out)}; A点行数: {len(a)}")
    print(a[["date","A_point"]].tail(10))


if __name__ == "__main__":
    main()

