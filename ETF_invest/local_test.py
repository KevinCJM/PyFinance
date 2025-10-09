# -*- encoding: utf-8 -*-
"""
@File: local_test.py
@Modify Time: 2025/10/7 08:26       
@Author: Kevin-Chen
@Descriptions: 
"""
# -*- coding: utf-8 -*-
"""
获取“指数信息API”数据，并保存为 Excel
API: https://open.lixinger.com/api/cn/index  (POST)
"""
import time
from typing import Iterable, List, Optional, Dict, Any
import requests
import pandas as pd

API_URL = "https://open.lixinger.com/api/cn/index"


def _post_with_retry(session: requests.Session, url: str, json: Dict[str, Any],
                     timeout: float = 15.0, max_retries: int = 3, backoff: float = 1.5):
    """
    轻量重试：网络闪断或 5xx 时重试；4xx 不重试（多半是参数问题/Token 失败）
    """
    for i in range(max_retries):
        try:
            resp = session.post(url, json=json, timeout=timeout)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"Server error {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                # 明确抛出，方便用户看到错误信息（如 token 失效/配额不足/参数错误）
                raise requests.HTTPError(f"Client error {resp.status_code}: {resp.text[:200]}")
            return resp.json()
        except Exception as e:
            if i == max_retries - 1:
                raise
            time.sleep(backoff ** (i + 1))


def _to_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将返回的 list[dict] 规范化为 DataFrame，并做少量清洗
    """
    if not items:
        return pd.DataFrame()
    df = pd.json_normalize(items)

    # 常见字段补齐（不存在则保持缺失）
    preferred_cols = [
        "name", "stockCode", "areaCode", "market",
        "fsTableType", "source", "currency", "series",
        "launchDate", "rebalancingFrequency", "caculationMethod",
    ]
    # 将 launchDate 解析为日期；其余保持字符串
    if "launchDate" in df.columns:
        df["launchDate"] = pd.to_datetime(df["launchDate"], errors="coerce")

    # 排列列顺序：优先常用字段，其次其它字段
    other_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[[c for c in preferred_cols if c in df.columns] + other_cols]

    # 去重（以 stockCode+market 作为主键的常见选择）
    key_cols = [c for c in ["stockCode", "market"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    return df


def fetch_index_info(token: str,
                     stock_codes: Optional[Iterable[str]] = None,
                     batch_size: int = 200) -> pd.DataFrame:
    """
    调用指数信息API：
    - 不传 stock_codes：全量
    - 传入 stock_codes：按代码分批请求
    """
    session = requests.Session()
    all_rows: List[Dict[str, Any]] = []

    if not stock_codes:
        payload = {"token": token}
        data = _post_with_retry(session, API_URL, json=payload)
        # 规范的返回形如 {"code": 1, "message":"success", "data":[...]}
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise ValueError(f"Unexpected response: {str(data)[:200]}")
        all_rows.extend(items)
    else:
        stock_codes = list(stock_codes)
        for s in range(0, len(stock_codes), batch_size):
            sub = stock_codes[s: s + batch_size]
            payload = {"token": token, "stockCodes": sub}
            data = _post_with_retry(session, API_URL, json=payload)
            items = data.get("data") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise ValueError(f"Unexpected response: {str(data)[:200]}")
            all_rows.extend(items)

    return _to_dataframe(all_rows)


def save_to_excel(df: pd.DataFrame, path: str) -> None:
    """
    保存为 Excel（带基本格式）
    """
    if df.empty:
        print("没有获取到任何数据，未生成文件。")
        return
    # 为了兼容性，显式指定 engine
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        df.to_excel(writer, sheet_name="index_info", index=False)
        # 自动列宽
        worksheet = writer.sheets["index_info"]
        for i, col in enumerate(df.columns):
            # 取列名与样本数据的最大宽度
            series = df[col].astype(str)
            max_len = max([len(col)] + [len(x) for x in series.head(500)])  # 采样前500行估计宽度
            worksheet.set_column(i, i, min(max_len + 2, 60))


if __name__ == "__main__":
    # ======= 使用示例 =======
    # 1) 仅填你的 token，即可获取“全量指数信息”
    TOKEN = "请把这里替换为你的Token"

    # 2) 若仅想要部分指数，填 codes；留空则拉取全量
    CODES = []  # 例如: ["000016", "000300", "000905"]

    df_idx = fetch_index_info(TOKEN, stock_codes=CODES or None, batch_size=200)
    print(f"共获取 {len(df_idx)} 条指数信息，示例：")
    print(df_idx.head(10))

    save_to_excel(df_idx, "index_info.xlsx")
    print("已保存到 index_info.xlsx")
