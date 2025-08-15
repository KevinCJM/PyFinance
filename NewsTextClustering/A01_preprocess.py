# -*- coding: utf-8 -*-
"""
A01_preprocess.py

一个“专门用于新闻预处理”的独立脚本：
- 仅做文本规范化与清洗（不做“相同新闻”识别）。
- 面向中文新闻，使用向量化 pandas 字符串操作与预编译正则，尽量避免低效循环与链式赋值警告。
- 输出规范化列：title_norm, text_norm, doc_norm；并剔除空文本行。

用法示例：
    python A01_preprocess.py \
        --input test_news.parquet \
        --out-parquet test_news_clean.parquet \
        --out-excel test_news_clean.xlsx

可选开关：
    --drop-dup-title    清洗后按 title_norm 去重（仅预处理阶段的“显式重复”）
    --drop-dup-text     清洗后按 text_norm 去重
    --keep-original     输出时携带原始 title/text 列（默认也会保留）

注意：
- 这里不做“相同新闻/相似新闻”的聚类或阈值判断；那部分在下一步单独脚本实现。
- 依赖：pandas, numpy（与 re）。
"""

import os
import sys
import re
import argparse
from typing import Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# ---------- 1) 预编译正则与替换表 ----------

# 统一空白
RE_SPACES = re.compile(r'\s+')

# HTML 残留
RE_HTML_ENT = re.compile(r'&lt;/?div&gt;|&nbsp;')

# $ 后面不是数字的 $
RE_DOLLAR_NONNUM = re.compile(r'\$(?!\d)')

# 开头像 “IT之家6月3日消息,” 的头
RE_IT_HOUSE = re.compile(r'^IT之家\d{1,2}月\d{1,2}日消息,')

# 括号内噪声（来源、记者、原标题等）
RE_PAREN_NOISE = re.compile(
    r'\([^)]*(记者|来源|特约通讯员|通讯员|图源|文/|以下信息从|以下内容从|原标题|摘要|科技消息|本报讯|V观财报|编者按|财华社讯|财经日历|第一财经|电鳗财经|法治周末|科创板日报|金证研|检察日报|基金经理投资笔记|每日经济新闻)[^)]*\)'
)

# “本报…记者…报道”模板（含可选括号/地点）
RE_BAODAO = re.compile(
    r'本报(?:讯)?'  # 可选“本报讯”
    r'(?:（[^）]*）|\([^)]*\)|【[^】]*】|\[[^\]]*\])?'  # 可选括号块（全/半角/方括/书名号）
    r'[^,.;:，。；：]*?记者'  # 到“记者”为止（不跨句）
    r'[^,.;:，。；：]*?报道'  # 到“报道”为止（不跨句）'
)

# 压缩连续的句点
RE_MULTI_DOTS = re.compile(r'\.{2,}')

# 单字符映射（translate 更快）；包含半角化/统一标点/去装饰符
SINGLE_CHAR_MAP = {
    '、': ',', '，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':',
    '（': '(', '）': ')', '【': '(', '】': ')', '《': '(', '》': ')',
    '“': '"', '”': '"', '‘': '"', '’': '"', '〝': '"', '〞': '"',
    '・': '-', '～': '-', '—': '-', '…': '.',  # 多个点后续统一压缩
    '·': '', '●': '', '■': '', '▲': '', '▼': '', '▎': '', '▍': '',
    '▌': '', '▐': '', '□': '', '△': '', '▽': '', '►': '', '▶': '',
    '◀': '', '◆': '', '◇': '', '★': '', '☆': '', '※': '', '◎': '',
    '〓': '', '#': '', '*': '', '＊': '', '§': '', '©': '', '®': '',
    '[': '(', ']': ')', '{': '(', '}': ')', '<': '(', '>': ')'
}
TRANS_TABLE = str.maketrans(SINGLE_CHAR_MAP)

# 站点/口号类短语统一（可按需补充）
PHRASE_MAP = {
    '阿里巴巴集团': '阿里', '阿里巴巴': '阿里',
    '北京商报讯': '', '北京商报': '',
    'cninfo.com.cn': 'cninfo', 'cninfo.com': 'cninfo',
    'DoNews.com': 'DoNews', 'DoNews.cn': 'DoNews',
    'Wind数据显示,': '', 'Wind资讯,': '', 'Wind资讯': '',
    'YY点评:': '', 'YY点评': '',
    '👇点击下方👇组团集卡抽红包,省到就是赚到': '',
    'irm.cn': 'irm', 'e公司讯,': '',
    '????': '', '???': ''
}
if PHRASE_MAP:
    # 构造一次性正则（按长度降序，避免子串干扰）
    _phr_keys = sorted(PHRASE_MAP.keys(), key=len, reverse=True)
    RE_PHRASES = re.compile('|'.join(map(re.escape, _phr_keys)))
else:
    RE_PHRASES = None


# ---------- 基础工具 ----------

def _normalize_series(s: pd.Series) -> pd.Series:
    """向量化规范化：统一空白/标点，移除常见噪声模板。"""
    s = s.fillna('')
    s = s.str.replace(RE_SPACES, '', regex=True)
    s = s.str.replace(RE_HTML_ENT, '', regex=True)
    s = s.str.translate(TRANS_TABLE)
    if RE_PHRASES is not None:
        s = s.str.replace(RE_PHRASES, lambda m: PHRASE_MAP.get(m.group(0), ''), regex=True)
    s = s.str.replace(RE_DOLLAR_NONNUM, '', regex=True)
    s = s.str.replace(RE_PAREN_NOISE, '', regex=True)
    s = s.str.replace(RE_BAODAO, '', regex=True)
    s = s.str.replace(RE_IT_HOUSE, '', regex=True)
    s = s.str.replace(RE_MULTI_DOTS, '.', regex=True)
    return s


def preprocess_dataframe(df: pd.DataFrame,
                         drop_dup_title: bool = False,
                         drop_dup_text: bool = False,
                         keep_original: bool = True) -> pd.DataFrame:
    """
    对输入的 DataFrame 进行文本预处理，要求至少包含 ['title', 'text'] 列。

    预处理步骤包括：
    - 严格去除空值（将空字符串视作 NaN）；
    - 对 title 和 text 字段进行规范化处理，生成 title_norm 和 text_norm；
    - 合成 doc_norm 字段：title_norm + '。' + text_norm；
    - 可选地根据 title_norm 或 text_norm 去除重复行；
    - 返回预处理后的 DataFrame，不进行聚类等后续处理。

    参数：
        df (pd.DataFrame): 输入的 DataFrame，必须包含 'title' 和 'text' 列。
        drop_dup_title (bool): 是否根据 title_norm 去重，默认为 False。
        drop_dup_text (bool): 是否根据 text_norm 去重，默认为 False。
        keep_original (bool): 是否保留原始的 title 和 text 列，默认为 True。

    返回：
        pd.DataFrame: 预处理后的 DataFrame，包含 title_norm、text_norm 和 doc_norm 列，
                      若 keep_original 为 True，则还保留原始的 title 和 text 列。
    """
    if not {'title', 'text'}.issubset(df.columns):
        raise ValueError("输入 DataFrame 必须包含列：['title','text']")

    out_cols = ['title', 'text'] if keep_original else []
    # 严格去空（空串等价 NaN）
    df = df[['title', 'text']].copy()
    df['title'] = df['title'].replace('', np.nan)
    df['text'] = df['text'].replace('', np.nan)
    df = df.dropna(subset=['title', 'text']).reset_index(drop=True)

    # 规范化 title 和 text 列，生成对应的 *_norm 列
    title_norm = _normalize_series(df['title'])
    text_norm = _normalize_series(df['text'])

    # 拼接 title_norm 和 text_norm 得到 doc_norm
    doc_norm = title_norm.str.cat(text_norm, sep='。', na_rep='').str.strip()

    # 组装输出 DataFrame
    res = pd.DataFrame({
        **({'title': df['title']} if keep_original else {}),
        **({'text': df['text']} if keep_original else {}),
        'title_norm': title_norm,
        'text_norm': text_norm,
        'doc_norm': doc_norm
    })

    # 删除 doc_norm 为空的行
    res = res[res['doc_norm'].str.len() > 0].reset_index(drop=True)

    # 可选：按规范化后的 title 或 text 去重
    if drop_dup_title:
        res = res.loc[~res['title_norm'].duplicated()].reset_index(drop=True)
    if drop_dup_text:
        res = res.loc[~res['text_norm'].duplicated()].reset_index(drop=True)

    return res


# ---------- IO ----------

def read_any(input_path: str) -> pd.DataFrame:
    """根据扩展名自动读取 parquet/csv/json/jsonl。"""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ('.parquet', '.pq'):
        return pd.read_parquet(input_path)
    if ext in ('.csv',):
        return pd.read_csv(input_path)
    if ext in ('.jsonl', '.jsonl.gz'):
        return pd.read_json(input_path, lines=True)
    if ext in ('.json',):
        return pd.read_json(input_path)
    # 默认尝试 parquet，再退回 csv
    try:
        return pd.read_parquet(input_path)
    except Exception:
        return pd.read_csv(input_path)


def write_outputs(df: pd.DataFrame,
                  out_parquet: Optional[str],
                  out_excel: Optional[str]) -> None:
    if out_parquet:
        df.to_parquet(out_parquet, index=False)
        print(f"[INFO] 保存到 {out_parquet} 完成")
    if out_excel:
        df.to_excel(out_excel, index=False)
        print(f"[INFO] 保存到 {out_excel} 完成")


def main(drop_dup_title=True, drop_dup_text=True, keep_original=True,
         input_pq='test_news.parquet', out_pq='test_news_clean.parquet', out_excel='test_news_clean.xlsx'):
    print(f"[INFO] 读取：{input_pq}")
    df = read_any(input_pq)

    # 验证输入数据是否包含必需的列，并尝试自动适配大小写不同的列名
    if not {'title', 'text'}.issubset(df.columns):
        # 允许不同大小写/语言环境的列名，尝试自动适配
        lower_map = {c.lower(): c for c in df.columns}
        if 'title' in lower_map and 'text' in lower_map:
            df = df.rename(columns={lower_map['title']: 'title', lower_map['text']: 'text'})
        else:
            raise ValueError("输入数据必须包含列 'title' 与 'text'（大小写不敏感）。")

    print(f"[INFO] 原始条数：{len(df)}")

    # 执行数据预处理，包括去重等操作
    res = preprocess_dataframe(
        df,
        drop_dup_title=drop_dup_title,
        drop_dup_text=drop_dup_text,
        keep_original=keep_original  # 默认保留原始列，便于审计
    )

    print(f"[INFO] 预处理完成，剩余条数：{len(res)}")

    write_outputs(res, out_pq, out_excel)
    print(f"[INFO] 写出完成。")


if __name__ == '__main__':
    ''' 最终结果有4列: title, text, title_norm, text_norm, doc_norm
    分别表示: 标题, 正文, 规范化标题, 规范化正文, 合成的文档文本
    '''
    main(drop_dup_title=True, drop_dup_text=True, keep_original=True,
         input_pq='test_news.parquet',
         out_pq='test_news_clean.parquet', out_excel='test_news_clean.xlsx'
         )
