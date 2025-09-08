# -*- encoding: utf-8 -*-
"""
@File: limit.py
@Modify Time: 2025/8/29 17:04       
@Author: Kevin-Chen
@Descriptions: 
"""

from wrapt_timeout_decorator import *
import time
import sys


def time_limit(func, *args, _timeout_limit=5, _timeout_back=None, _default_result=None, **kwargs):
    start_t = time.time()
    process_time = -1
    res = _default_result
    try:
        if "WIN" in sys.platform.upper():
            @timeout(_timeout_limit, False)
            def wrapper(func, *args, **kwargs):
                return func(*args, **kwargs)
        else:
            @timeout(_timeout_limit, True)
            def wrapper(func, *args, **kwargs):
                return func(*args, **kwargs)

        res = wrapper(func, *args, **kwargs)
        process_time = time.time() - start_t

    except TimeoutError as e:
        if _timeout_back:
            res = _timeout_back(*args, **kwargs)

    return res, process_time
