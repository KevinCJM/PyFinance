# -*- encoding: utf-8 -*-
"""
@File: base.py
@Modify Time: 2025/8/25 14:36       
@Author: Kevin-Chen
@Descriptions: 
"""
import json
import traceback

from flask import request, make_response, views
from marshmallow.exceptions import ValidationError
from utils.errors import request_error_msg, params_error_msg


def resp_to_json(res, error_status=0):
    """
    统一的 json 结果返回封装
    :param res: 返回的数据
    :param error_status: 错误码 (0 表示正常)
    :return: JSON 字符串
    """
    resp_dic = {
        "data": res,
        "code": error_status,
        "msg": ""
    }
    resp_json = json.dumps(resp_dic, ensure_ascii=False)
    return resp_json


def read_param(request):
    res = dict()

    # form/query 参数
    for k, v in request.values.lists():   # ✅ 替换 iterlists
        if len(v) <= 1:
            v = v[0]
        res[k] = v

    # JSON body
    if request.is_json:
        res.update(request.json)
    elif request.data != b'':
        try:
            res.update(json.loads(request.data.decode()))
        except Exception as e:
            print(traceback.format_exc())
            print("parse request.data error: %s" % e)

    # header token
    if request.headers.get('Authorization'):
        try:
            res['token'] = request.headers.get('Authorization').split()[1]
        except Exception:
            print(traceback.format_exc())
            res['token'] = ''

    # 文件上传
    if request.files:
        res.update({i: f for i, f in request.files.items()})

    return res


class BaseView(views.MethodView):
    def get(self):
        return self.all_method()

    def post(self):
        return self.all_method()

    def print_request_debug_info(self, self_obj, req):
        """计划用于进行 用户名 IP 等内容的日志记录"""
        pass

    def all_method(self):
        """统一入口：从 request 中解析参数 f，动态调用方法"""
        # 参数预处理
        params = read_param(request)
        # base_log.info(params)
        try:
            f = params.pop('f')
        except KeyError:
            print(traceback.format_exc())
            # base_log.warning("the request has no f, set f=home")
            f = "home"

        # 请求处理
        if hasattr(self, f):
            try:
                return getattr(self, f)(**params)
            except ValidationError as e:
                return make_response(params_error_msg(e.messages, 400), 400)
            except Exception as e:
                # base_log.error("server error about function {} and catch it.".format(f))
                # base_log.exception(e)
                # base_log.debug("args_dict: {}".format(params))
                print(traceback.format_exc())
                return make_response(
                    request_error_msg(
                        str(e) or "The server error, please contact to services providers.",
                        500
                    ), 500
                )
        else:
            return make_response(request_error_msg(f"{f} is not a support function"), 400)


class JsonResBaseView(BaseView):
    def all_method(self):
        params = read_param(request)
        f = params.pop('f')

        if hasattr(self, f):
            try:
                return resp_to_json(getattr(self, f)(**params))
            except ValidationError as e:
                return make_response(params_error_msg(e.messages, 400), 400)
            except Exception as e:
                print("server error about function {} and catch it.".format(f))
                print(e)
                print(traceback.format_exc())
                print(f"error_func: {f}, args_dict: {params}")
                return make_response(
                    request_error_msg(str(e) or "The server error, please contact to services providers.", 500),
                    500
                )
        else:
            return make_response(request_error_msg(f'{f} is not a support function'), 400)
