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

# —— 金融界净值快讯提取 ——
# 例：金融界2024年11月1日消息,万家颐和灵活配置混合A(519198)最新净值1.5545元,下跌1.87%.（或“增长”）
RE_JRJ_NAV = re.compile(
    r'^金融界'
    r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日消息,'  # 日期
    r'(?P<fund_name>[^()，,]+)\((?P<code>\d{6})\)'  # 基金名 + 6位代码
    r'最新净值(?P<nav>\d+(?:\.\d+)?)元,'  # 最新净值
    r'(?P<dir>增长|上涨|上升|下跌|下降)'  # 方向
    r'(?P<chg>-?\d+(?:\.\d+)?)%'  # 百分比
    r'[。\.，,]?',  # 末尾可选标点
    flags=re.UNICODE
)

# 例：该基金近1个月收益率0.06%,同类排名588|1911;近3个月收益率6.99%,同类排名1130|1900;今年来收益率11.62%,同类排名295|1870.
RE_JRJ_TAIL = re.compile(
    r'近1个月收益率(?P<ret_1m>-?\d+(?:\.\d+)?)%,同类排名(?P<rank_1m>\d+)\|(?P<base_1m>\d+);'
    r'近3个月收益率(?P<ret_3m>-?\d+(?:\.\d+)?)%,同类排名(?P<rank_3m>\d+)\|(?P<base_3m>\d+);'
    r'(?:今年来|年内)收益率(?P<ret_ytd>-?\d+(?:\.\d+)?)%,同类排名(?P<rank_ytd>\d+)\|(?P<base_ytd>\d+)',
    flags=re.UNICODE
)

DIR_SIGN_MAP = {'增长': 1, '上涨': 1, '上升': 1, '下跌': -1, '下降': -1}


def extract_jrj_nav_rows(res: pd.DataFrame) -> pd.DataFrame:
    """
    在预处理结果 res 上（包含 text_norm）向量化抽取“金融界净值类快讯”。
    返回含解析字段的 DataFrame；若无匹配，返回空表。
    """
    if 'text_norm' not in res.columns:
        return pd.DataFrame()

    t = res['text_norm']
    head = t.str.extract(RE_JRJ_NAV)

    mask = head['year'].notna()
    if not mask.any():
        return pd.DataFrame()

    # 基础列（保留原文便于审计）
    base = res.loc[mask, ['title', 'text', 'title_norm', 'text_norm', 'doc_norm']].reset_index(drop=True)

    # 头部结构化字段
    head = head.loc[mask].reset_index(drop=True)
    date_str = (
            head['year'].astype(str) + '-' +
            head['month'].astype(str).str.zfill(2) + '-' +
            head['day'].astype(str).str.zfill(2)
    )

    # 尾段结构化字段（可选）
    tail = t.loc[mask].str.extract(RE_JRJ_TAIL)

    out = pd.DataFrame({
        'date': pd.to_datetime(date_str, errors='coerce'),
        'fund_name': head['fund_name'],
        'fund_code': head['code'],
        'nav': pd.to_numeric(head['nav'], errors='coerce'),
        'direction': head['dir'],
        # 规范化涨跌幅：方向（±1）* |数值|
        'chg_pct': pd.to_numeric(head['chg'], errors='coerce').abs() *
                   head['dir'].map(DIR_SIGN_MAP).astype('float64')
    })

    # 追加尾段字段（若缺失则为 NaN）
    for col in ['ret_1m', 'ret_3m', 'ret_ytd']:
        out[col] = pd.to_numeric(tail.get(col), errors='coerce')
    for col in ['rank_1m', 'base_1m', 'rank_3m', 'base_3m', 'rank_ytd', 'base_ytd']:
        out[col] = pd.to_numeric(tail.get(col), errors='coerce', downcast='integer')

    # 合并回原文，便于人工核对
    out = pd.concat([base, out], axis=1)
    return out


# —— 重仓股快讯提取（最新披露数据显示…十大重仓股） ——
RE_TOP_HOLDINGS = re.compile(
    r'最新披露数据显示,'  # 起始锚点（允许前面还有别的话，extract会search）
    r'截(?:至|止)'  # 截至/截止 兼容
    r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日'
    r'(?:消息)?,'  # 可选“消息,”
    r'(?P<company>[^,;，；]+?)'  # 公司名（到“现身/出现在”之前）
    r'(?:现身|出现在)'  # 现身/出现在
    r'(?P<funds>\d+)只基金的(?:前)?十大重仓股(?:中)?,'  # “前十大”可选，“中”可选
    r'较上季度(?:'  # 同环比描述
    r'(?P<dir>增加|减少)(?:了)?(?P<delta>\d+)只'  # 增加/减少（可带“了”）
    r'|持平)'  # 或“持平”
    r'[;,，]'  # 分隔符
    r'合计持有(?P<shares>\d+(?:\.\d+)?)(?P<shares_unit>[万亿]?)股,'  # 股数 + 单位
    r'(?:持股|持仓)市值(?P<mv>\d+(?:\.\d+)?)(?P<mv_unit>万亿|万|亿)元'  # 金额 + 单位（含“万亿”）
    r'(?:|(?=,|，|。|\.))[,，]?'  # 后面可能还有逗号
    r'为公募基金第(?P<rank>\d+)大重仓股'
    r'(?:\((?:按)?[^)]*(?:市值|持股市值)[^)]*\))?',  # 括注里“排序/排名/按…市值…” 等都放宽
    flags=re.UNICODE
)

TOP_DIR_SIGN = {'增加': 1, '减少': -1}
UNIT_TO_NUM = {'': 1.0, '万': 1e4, '亿': 1e8, '万亿': 1e12}  # 金额/股数统一倍数表


def extract_top_holdings_rows(res: pd.DataFrame) -> pd.DataFrame:
    """
    在预处理结果 res（需含 text_norm）上抽取“最新披露数据显示…十大/前十大重仓股”类新闻。
    返回结构化字段；无匹配返回空表。
    """
    if 'text_norm' not in res.columns:
        return pd.DataFrame()

    ext = res['text_norm'].str.extract(RE_TOP_HOLDINGS)  # search 语义
    mask = ext['year'].notna()
    if not mask.any():
        return pd.DataFrame()

    ext = ext.loc[mask].reset_index(drop=True)

    # 日期
    date = pd.to_datetime(
        ext['year'].astype(str) + '-' +
        ext['month'].astype(str).str.zfill(2) + '-' +
        ext['day'].astype(str).str.zfill(2),
        errors='coerce'
    )

    # 基础数值
    funds = pd.to_numeric(ext['funds'], errors='coerce').astype('Int64')

    # 同环比变化：增加/减少 → ±delta；持平 → 0
    has_dir = ext['dir'].notna()
    qoq_change = pd.Series(pd.NA, index=ext.index, dtype='Int64')
    qoq_change.loc[has_dir] = (
            ext.loc[has_dir, 'dir'].map(TOP_DIR_SIGN).astype('float64') *
            pd.to_numeric(ext.loc[has_dir, 'delta'], errors='coerce')
    ).round().astype('Int64')
    qoq_change.loc[~has_dir] = 0  # “持平”分支

    # 股数：原值×单位
    shares_val = pd.to_numeric(ext['shares'], errors='coerce')
    shares_mul = ext['shares_unit'].map(UNIT_TO_NUM).astype('float64')
    shares_abs = shares_val * shares_mul  # 股

    # 金额：原值×单位（兼容 万亿/万/亿）
    mv_val = pd.to_numeric(ext['mv'], errors='coerce')
    mv_mul = ext['mv_unit'].map(UNIT_TO_NUM).astype('float64')
    mv_cny = mv_val * mv_mul  # 元

    rank = pd.to_numeric(ext['rank'], errors='coerce').astype('Int64')

    # 原文列（便于审计）
    base_cols = [c for c in ['title', 'text', 'title_norm', 'text_norm', 'doc_norm'] if c in res.columns]
    base = res.loc[mask, base_cols].reset_index(drop=True)

    out = pd.DataFrame({
        'date': date,
        'company': ext['company'],
        'fund_count': funds,
        'qoq_change': qoq_change,  # 较上季度变化：+/-N；持平=0
        'shares': shares_abs,  # 绝对股数（股）
        'shares_value': shares_val,  # 原值
        'shares_unit': ext['shares_unit'],  # '', '万', '亿'
        'market_value_cny': mv_cny,  # 绝对金额（元）
        'market_value_value': mv_val,  # 原值
        'market_value_unit': ext['mv_unit'],  # '万', '亿', '万亿'
        'rank_by_mktvalue': rank
    })

    out = pd.concat([base, out], axis=1)
    return out


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
    '👆如果您希望可以时常见面，欢迎标星🌟收藏哦~': '',
    '👆如果您希望可以时常见面,欢迎标星🌟收藏哦~': '',
    'irm.cn': 'irm', 'e公司讯,': '',
    '智通财经APP讯,': '', '智通财经APP获悉,': '', '智东西(公众号:zhidxcom)': '',
    '证券之星消息,': '', '证券时报网讯,': '', '证券时报企查查APP显示,近日,': '',
    '资讯正文/!normalize.cssv7.0.0|MITLicense|github.com/necolas/normalize.css/html(line-height:1.15;-ms-text-size-adjust:100%;-webkit-text-size-adjust:100%)body(margin:0)article,aside,footer,header,nav,section(display:block)h1(font-size:2em;margin:.67em0)figcaption,figure,main(display:block)figure(margin:1em40px)hr(-webkit-box-sizing:content-box;box-sizing:content-box;height:0;overflow:visible)pre(font-family:monospace,monospace;font-size:1em)a(background-color:rgba(0,0,0,0);-webkit-text-decoration-skip:objects)abbr(title)(border-bottom:none;text-decoration:underline;-webkit-text-decoration:underlinedotted;text-decoration:underlinedotted)b,strong(font-weight:inherit;font-weight:bolder)code,kbd,samp(font-family:monospace,monospace;font-size:1em)dfn(font-style:italic)mark(background-color:ff0;color:000)small(font-size:80%)sub,sup(font-size:75%;line-height:0;position:relative;vertical-align:baseline)sub(bottom:-.25em)sup(top:-.5em)audio,video(display:inline-block)audio:not((controls))(display:none;height:0)img(border-style:none)svg:not(:root)(overflow:hidden)button,input,optgroup,select,textarea(font-family:sans-serif;font-size:100%;line-height:1.15;margin:0)button,input(overflow:visible)button,select(text-transform:none)(type=reset),(type=submit),button,html(type=button)(-webkit-appearance:button)(type=button)::-moz-focus-inner,(type=reset)::-moz-focus-inner,(type=submit)::-moz-focus-inner,button::-moz-focus-inner(border-style:none;padding:0)(type=button):-moz-focusring,(type=reset):-moz-focusring,(type=submit):-moz-focusring,button:-moz-focusring(outline:1pxdottedButtonText)fieldset(padding:.35em.75em.625em)legend(-webkit-box-sizing:border-box;box-sizing:border-box;color:inherit;display:table;max-width:100%;padding:0;white-space:normal)progress(display:inline-block;vertical-align:baseline)textarea(overflow:auto)(type=checkbox),(type=radio)(-webkit-box-sizing:border-box;box-sizing:border-box;padding:0)(type=number)::-webkit-inner-spin-button,(type=number)::-webkit-outer-spin-button(height:auto)(type=search)(-webkit-appearance:textfield;outline-offset:-2px)(type=search)::-webkit-search-cancel-button,(type=search)::-webkit-search-decoration(-webkit-appearance:none)::-webkit-file-upload-button(-webkit-appearance:button;font:inherit)details,menu(display:block)summary(display:list-item)canvas(display:inline-block)template(display:none)(hidden)(display:none):root(font-size:50px)ul(padding:0;margin:0)ulli(list-style:none)ai-news-detail(color:333;font-family:PingFangSCRegular,PingFangSC-Regular,microsoftyahei,"\5B8B\4F53",tahoma,arial,simsun,sans-serif;padding:.32rem.32rem.48rem)ai-news-detail)p(font-size:.32rem;line-height:.44rem;margin:00.58rem)ai-news-detail.change-list)li(margin:00.52rem)ai-news-detail.change-list)lih2(font-size:.3rem;line-height:.42rem;margin:00.38rem;font-family:PingFangSC-Medium,PingFangSC-Regular,PingFangSCRegular,microsoftyahei,"\5B8B\4F53",tahoma,arial,simsun,sans-serif)ai-news-detail.change-list)li)ul(display:-webkit-box;display:-webkit-flex;display:-ms-flexbox;display:flex)ai-news-detail.change-list)li)ulli(-webkit-box-flex:1;-webkit-flex:1;-ms-flex:1;flex:1;padding:000.37rem)ai-news-detail.change-list)li)ulli:first-of-type(padding:0.37rem00;position:relative)ai-news-detail.change-list)li)ulli:first-of-type:after(content:"";position:absolute;top:50%;right:0;height:80%;width:1px;background:dfdfdf;-webkit-transform:scaleX(.5)translateY(-50%);-ms-transform:scaleX(.5)translateY(-50%);transform:scaleX(.5)translateY(-50%))ai-news-detail.change-list)li)ullih3(color:999;font-size:.26rem;line-height:.36rem;margin:00.16rem)ai-news-detail.change-list)li)ulli.compare-itemp(margin:0;line-height:.46rem;font-size:.28rem)': '',
    '资讯正文     /*! normalize.css v7.0.0 | MIT License | github.com/necolas/normalize.css */html{line-height:1.15;-ms-text-size-adjust:100%;-webkit-text-size-adjust:100%}body{margin:0}article,aside,footer,header,nav,section{display:block}h1{font-size:2em;margin:.67em 0}figcaption,figure,main{display:block}figure{margin:1em 40px}hr{-webkit-box-sizing:content-box;box-sizing:content-box;height:0;overflow:visible}pre{font-family:monospace,monospace;font-size:1em}a{background-color:rgba(0,0,0,0);-webkit-text-decoration-skip:objects}abbr[title]{border-bottom:none;text-decoration:underline;-webkit-text-decoration:underline dotted;text-decoration:underline dotted}b,strong{font-weight:inherit;font-weight:bolder}code,kbd,samp{font-family:monospace,monospace;font-size:1em}dfn{font-style:italic}mark{background-color:#ff0;color:#000}small{font-size:80%}sub,sup{font-size:75%;line-height:0;position:relative;vertical-align:baseline}sub{bottom:-.25em}sup{top:-.5em}audio,video{display:inline-block}audio:not([controls]){display:none;height:0}img{border-style:none}svg:not(:root){overflow:hidden}button,input,optgroup,select,textarea{font-family:sans-serif;font-size:100%;line-height:1.15;margin:0}button,input{overflow:visible}button,select{text-transform:none}[type=reset],[type=submit],button,html [type=button]{-webkit-appearance:button}[type=button]::-moz-focus-inner,[type=reset]::-moz-focus-inner,[type=submit]::-moz-focus-inner,button::-moz-focus-inner{border-style:none;padding:0}[type=button]:-moz-focusring,[type=reset]:-moz-focusring,[type=submit]:-moz-focusring,button:-moz-focusring{outline:1px dotted ButtonText}fieldset{padding:.35em .75em .625em}legend{-webkit-box-sizing:border-box;box-sizing:border-box;color:inherit;display:table;max-width:100%;padding:0;white-space:normal}progress{display:inline-block;vertical-align:baseline}textarea{overflow:auto}[type=checkbox],[type=radio]{-webkit-box-sizing:border-box;box-sizing:border-box;padding:0}[type=number]::-webkit-inner-spin-button,[type=number]::-webkit-outer-spin-button{height:auto}[type=search]{-webkit-appearance:textfield;outline-offset:-2px}[type=search]::-webkit-search-cancel-button,[type=search]::-webkit-search-decoration{-webkit-appearance:none}::-webkit-file-upload-button{-webkit-appearance:button;font:inherit}details,menu{display:block}summary{display:list-item}canvas{display:inline-block}template{display:none}[hidden]{display:none}:root{font-size:50px}ul{padding:0;margin:0}ul li{list-style:none}#ai-news-detail{color:#333;font-family:PingFang SC Regular,PingFangSC-Regular,microsoft yahei,"\5B8B\4F53",tahoma,arial,simsun,sans-serif;padding:.32rem .32rem .48rem}#ai-news-detail>p{font-size:.32rem;line-height:.44rem;margin:0 0 .58rem}#ai-news-detail .change-list>li{margin:0 0 .52rem}#ai-news-detail .change-list>li h2{font-size:.3rem;line-height:.42rem;margin:0 0 .38rem;font-family:PingFangSC-Medium,PingFangSC-Regular,PingFang SC Regular,microsoft yahei,"\5B8B\4F53",tahoma,arial,simsun,sans-serif}#ai-news-detail .change-list>li>ul{display:-webkit-box;display:-webkit-flex;display:-ms-flexbox;display:flex}#ai-news-detail .change-list>li>ul li{-webkit-box-flex:1;-webkit-flex:1;-ms-flex:1;flex:1;padding:0 0 0 .37rem}#ai-news-detail .change-list>li>ul li:first-of-type{padding:0 .37rem 0 0;position:relative}#ai-news-detail .change-list>li>ul li:first-of-type:after{content:"";position:absolute;top:50%;right:0;height:80%;width:1px;background:#dfdfdf;-webkit-transform:scaleX(.5) translateY(-50%);-ms-transform:scaleX(.5) translateY(-50%);transform:scaleX(.5) translateY(-50%)}#ai-news-detail .change-list>li>ul li h3{color:#999;font-size:.26rem;line-height:.36rem;margin:0 0 .16rem}#ai-news-detail .change-list>li>ul li .compare-item p{margin:0;line-height:.46rem;font-size:.28rem} ': '',
    '转自:上海证券报中国证券网上证报中国证券网讯': '', '转自:上海证券报': '', '转自:商务微新闻': '',
    '转自:山西发布': '', '转自:人民日报海外版': '', '转自:人民日报': '', '转自:人民政协网': '', '转自:人民政协报': '',
    '转自:上海证券报中国证券网讯': '', '转自:上观新闻': '', '转自:上海市基金同业公会': '', '转自:人民论坛': '',
    '转自:衢州日报': '', '转自:青蕉视频': '', '转自:秦皇岛新闻网': '', '转自:千龙网中新网': '', '转自:千龙网': '',
    '转自:起点新闻': '', '转自:齐鲁晚报': '', '转自:齐鲁壹点': '', '转自:齐鲁网': '', '转自:前瞻网': '',
    '转自:中国新闻网': '', '转自:中国网': '', '转自:期货日报': '', '转自:农民日报': '', '转自:内蒙古日报': '',
    '转自:南京晨报': '', '转自:南湖晚报': '', '转自:南方日报': '', '转自:辽宁日报': '', '转自:九派新闻': '',
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

    # —— 在预处理结果上提取“金融界净值类快讯”并单独导出 ——
    jrj_df = extract_jrj_nav_rows(res)
    if not jrj_df.empty:
        jrj_excel = "jrj_news_clean.xlsx"
        jrj_parquet = "jrj_news_clean.parquet"
        jrj_df.to_excel(jrj_excel, index=False)
        jrj_df.to_parquet(jrj_parquet, index=False)
        print(f"[INFO] 金融界净值类新闻匹配 {len(jrj_df)} 条，已另存为：{jrj_excel} & {jrj_parquet}")
        print(f"[INFO] 从原数据中剔除金融界基金新闻")
        res = res[~res['doc_norm'].isin(jrj_df['doc_norm'])].reset_index(drop=True)
        print(f"[INFO] 剩余条数：{len(res)}")
    else:
        print("[INFO] 未匹配到金融界净值类新闻。")

    # —— 抽取“最新披露数据显示…十大重仓股”并单独导出 ——
    top_df = extract_top_holdings_rows(res)
    if not top_df.empty:
        top_excel = "top_holdings_news_clean.xlsx"
        top_parquet = "top_holdings_news_clean.parquet"
        top_df.to_excel(top_excel, index=False)
        top_df.to_parquet(top_parquet, index=False)
        print(f"[INFO] 重仓股快讯匹配 {len(top_df)} 条，已另存：{top_excel} & {top_parquet}")
        print(f"[INFO] 从原数据中剔除重仓股快讯")
        res = res[~res['doc_norm'].isin(top_df['doc_norm'])].reset_index(drop=True)
        print(f"[INFO] 剩余条数：{len(res)}")
    else:
        print("[INFO] 未匹配到重仓股快讯。")

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
