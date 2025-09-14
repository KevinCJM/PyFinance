"""ETF 数据抓取脚本（多线程 + 重试 + Parquet 输出）

使用 Tushare Pro 拉取 ETF 基本信息与净值日行情：
- 使用线程池并发抓取 `fund_nav`。
- 失败自动重试 n 次（带退避）。
- 将结果合并保存为 Parquet 文件。

注意：请通过环境变量 `TUSHARE_TOKEN` 提供访问令牌。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import tushare as ts

# 可调参数
MAX_WORKERS = int(os.getenv("ETF_FETCH_MAX_WORKERS", "8"))
MAX_RETRIES = int(os.getenv("ETF_FETCH_MAX_RETRIES", "3"))
RETRY_BACKOFF_SEC = float(os.getenv("ETF_FETCH_BACKOFF_SEC", "1.5"))
OUTPUT_DIR = os.getenv("ETF_OUTPUT_DIR", os.path.join("data", "processed"))
ETF_INFO_XLSX = os.path.join(OUTPUT_DIR, "etf_info_df.xlsx")
ETF_NAV_PARQUET = os.path.join(OUTPUT_DIR, "etf_daily_df.parquet")


def _require_token() -> str:
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "未检测到 TUSHARE_TOKEN 环境变量，请先设置后再运行。"
        )
    return token


def fetch_etf_list(token: str) -> pd.DataFrame:
    pro = ts.pro_api(token)
    df = pro.fund_basic(market="E")
    # 规范列名 & 保留常用字段
    keep = ["ts_code", "name", "management", "found_date"]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy() if cols else df.copy()


def fetch_nav_once(token: str, ts_code: str) -> Optional[pd.DataFrame]:
    pro = ts.pro_api(token)
    df = pro.fund_nav(ts_code=ts_code)
    if df is None or df.empty:
        return None
    # 只保留需要的列
    keep = ["ts_code", "nav_date", "unit_nav", "accum_nav", "adj_nav"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()
    # 转换日期
    if "nav_date" in df.columns:
        df["nav_date"] = pd.to_datetime(df["nav_date"])
        df["date"] = df["nav_date"]
    return df


def fetch_nav_with_retry(token: str, ts_code: str, name: str, max_retries: int) -> Optional[pd.DataFrame]:
    last_err: Optional[Exception] = None
    backoff = RETRY_BACKOFF_SEC
    for attempt in range(1, max_retries + 1):
        try:
            df = fetch_nav_once(token, ts_code)
            if df is not None and not df.empty:
                df["name"] = name
                df = df.sort_values("nav_date").reset_index(drop=True)
                return df
            # 空数据也视为失败，进行重试
            last_err = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001
            last_err = e
        # 重试等待
        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2
    # 最终失败
    print(f"[WARN] 拉取 {ts_code} - {name} 失败，错误：{last_err}")
    return None


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    token = _require_token()
    ensure_output_dir(OUTPUT_DIR)

    # 1) 获取 ETF 列表
    etf_info_df = fetch_etf_list(token)
    # 可选：保存元数据（便于排查），受 .gitignore 保护
    try:
        etf_info_df.to_excel(ETF_INFO_XLSX, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 写入 ETF 基本信息 Excel 失败：{e}")

    etf_dict: Dict[str, str] = {i: j for i, j in zip(etf_info_df.get("ts_code", []), etf_info_df.get("name", []))}
    total = len(etf_dict)
    print(f"准备抓取 {total} 只 ETF 的净值数据，线程数={MAX_WORKERS}，重试={MAX_RETRIES}")

    # 2) 并发抓取净值
    results: List[pd.DataFrame] = []
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_code = {
            executor.submit(fetch_nav_with_retry, token, code, name, MAX_RETRIES): (code, name)
            for code, name in etf_dict.items()
        }
        for idx, future in enumerate(as_completed(future_to_code), 1):
            code, name = future_to_code[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results.append(df)
                else:
                    failures.append(code)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 任务异常 {code}-{name}：{e}")
                failures.append(code)
            if idx % 50 == 0 or idx == total:
                print(f"进度：{idx}/{total}，成功 {len(results)}，失败 {len(failures)}")

    if not results:
        raise RuntimeError("抓取失败：未获得任何 ETF 净值数据。")

    # 3) 合并与保存
    etf_daily_df = pd.concat(results, axis=0, ignore_index=True)
    # 去重（以 ts_code + date 唯一）
    if {"ts_code", "date"}.issubset(etf_daily_df.columns):
        etf_daily_df = etf_daily_df.drop_duplicates(subset=["ts_code", "date"]).reset_index(drop=True)

    etf_daily_df.to_parquet(ETF_NAV_PARQUET, index=False)
    print(f"已保存 Parquet：{ETF_NAV_PARQUET}，共 {len(etf_daily_df)} 行；失败 {len(failures)} 支。")


if __name__ == "__main__":
    main()
