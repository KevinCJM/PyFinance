# -*- encoding: utf-8 -*-
"""
@File: errors.py
@Modify Time: 2025/8/25 14:43       
@Author: Kevin-Chen
@Descriptions: 
"""

import json


def request_error_msg(msg, error_code=0):
    """
    用于返回请求错误时的提示信息
    :param msg: 错误描述信息（字符串）
    :param error_code: 错误码
    :return: json 字符串
    """
    msg = str(msg)  # 如果不是字符串类型，强制转换
    msg_dic = {
        "code": error_code,
        "msg": msg
    }
    msg_json = json.dumps(msg_dic, ensure_ascii=False)
    return msg_json


def params_error_msg(msg, error_code=400):
    """
    用于返回参数校验错误时的提示信息
    :param msg: dict/list 错误描述（如 marshmallow 校验的 messages）
    :param error_code: 错误码
    :return: json 字符串
    """
    for k, v in msg.items():
        if isinstance(v, list):
            msg[k] = '/'.join(v)
        if isinstance(v, dict):
            inner_v = []
            for inner_k, inner_val in v.items():
                inner_v.append(f"{inner_k}:{'/'.join(inner_val)}")
            msg[k] = ';'.join(inner_v)

    msg_dic = {
        "code": error_code,
        "msg": msg
    }
    msg_json = json.dumps(msg_dic, ensure_ascii=False)
    return msg_json


def request_success_msg(data, code=200):
    """
    用于返回成功时的提示信息
    :param data: 返回数据
    :param code: 状态码
    :return: json 字符串
    """
    msg_dic = {
        "code": code,
        "msg": "success",
        "data": data
    }
    msg_json = json.dumps(msg_dic, ensure_ascii=False)
    return msg_json
