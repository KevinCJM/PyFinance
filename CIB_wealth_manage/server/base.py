# -*- encoding: utf-8 -*-
"""
@File: base.py
@Modify Time: 2025/8/25 14:33       
@Author: Kevin-Chen
@Descriptions: 
"""
import time
from flask import Flask
from server.service import Service

app = Flask(__name__)
s = Service()

raw_route_dic = {
    '/': '/',
    "wealth_manage": "/wealth_manage",
}

# 后续可能要对 route 做映射
route_dic = {k: v for k, v in raw_route_dic.items()}

# ====== 注册路由文件 ======
from views import wealth_manage
