# -*- coding:utf-8 -*-
"""
# 用途：配置项目
# 创建日期: 18-9-19 下午9:07
"""
import time
import os

ASSET_CODE_DICT = {
    "现金类": "01",
    "固收类": "02",
    "混合类": "03",
    "权益类": "04",
    "另类": "05"
}

CLAZZ_ORDER = ["现金类", "固收类", "混合类", "权益类", "另类"]

FUN_CLASS = [
    '现金类占比评分',
    '预期收益率评分',
    '预期波动率评分',
    '分散度评分'
]

DEBUG = False if os.getenv("DEBUG") is None else os.getenv("DEBUG").lower() == 'true'
# DEBUG = 'true'

if __name__ == "__main__":
    start_t = time.time()
    print("use time: %s" % (time.time() - start_t))
