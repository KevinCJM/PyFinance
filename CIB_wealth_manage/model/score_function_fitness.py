# -*- encoding: utf-8 -*-
"""
@File: score_function_fitness.py
@Modify Time: 2025/9/2 17:49       
@Author: Kevin-Chen
@Descriptions: 
"""
import math
import traceback
import numpy as np
from itertools import product
from scipy.optimize import minimize

algorithm = [
    'SLSQP', 'Powell',
    'Nelder-Mead',
    'CG', 'BFGS',
    'Newton-CG', 'L-BFGS-B',
    'TNC', 'trust-krylov',
    'trust-constr', 'dogleg',
    'trust-ncg', 'trust-exact',
]


def arctan(a, b, c, d, x):
    return a * np.arctan(b * x + c) + d


def atan(a, b, c, d, x):
    return np.array([a * math.atan(b * v + c) + d for v in x])


def morgan_mercer_flodin(a, b, c, d, x):
    return (b * c + a * np.power(x, d)) / (c + np.power(x, d))


def tanh_neg(a, b, c, d, x):
    return -1 * a * np.tanh(b * x + c) + d


def quadratic(a, b, x):
    c = 0
    return a * np.power(x, 2) + b * x + c


func_desc = {
    'atan': 'a*atan(b*x+c)+d',
    'arctan': 'a*arctan(b*x+c)+d',
    'morgan_mercer_flodin': 'b*c+a*x^d/(c+x^d)',
    'tanh_neg': '-1*a*tanh(b*x+c)+d',
    'quadratic': "a*x^2+b*x+c"
}


def obj_func(params, *args):
    w, x, y, f = args
    y_pred = f(*params, x)
    return np.sum((w * (y - y_pred)) ** 2)


def obj_func2(params, *args):
    w, x, y, f = args
    y_pred = f(*params, x)
    y_pred[y_pred < 0] = 10000
    y_pred[y_pred > 1] = 10000
    return np.sum((w * (y - y_pred)) ** 2)


class ScoreFunctionFitness:

    @staticmethod
    def cash_score_function_fitness(x_, y_):
        # 初始化解
        x0 = {
            'arctan': np.array([0.55908293, 26.55177979, -0.261595, 0.14304445]),
            'atan': np.array([0.55908293, 26.55177979, -0.261595, 0.14304445]),
        }

        x = x_
        y = y_
        res_min = np.inf
        params = None
        best_fun = None

        for i in range(len(x)):
            if i == 0 or i == len(x) - 1:
                continue

            if abs(x[i] - x[-1]) <= 0.01:
                x[i] = x[-1] - 0.02

            if abs(x[i] - x[0]) <= 0.01:
                x[i] = x[0] + 0.01

        w = np.ones(len(x_))

        for fun in [atan]:
            for method in algorithm:
                print(f"尝试使用函数 {fun.__name__} 和优化方法 {method} ==========================================")
                try:
                    res = minimize(obj_func, x0[fun.__name__], args=(w, x, y, fun), method=method)
                    if res['success'] != True:
                        print(fun.__name__, method, '波动率评分拟合失败')
                    else:
                        if res.fun < res_min:
                            params = res.x
                            y_predit = fun(*params, x)
                            res_min = res.fun
                            best_fun = fun
                except Exception as e:
                    print(fun.__name__, method, '波动率评分拟合异常', e)
                    print(traceback.format_exc())
        return best_fun.__name__, func_desc[best_fun.__name__], params, res_min, y_predit

    @staticmethod
    def return_score_function_fitness(x_, y_):

        def _search(c_start, c_end):
            a_ = [100]
            b_ = [0.99]
            c_ = np.power(10, np.linspace(c_start, c_end, 100))
            d_ = np.linspace(0, 5, 100)
            fun = morgan_mercer_flodin

            fmin = np.inf
            x_best = None

            for a, b, c, d in product(a_, b_, c_, d_):
                loss = obj_func(params=(a, b, c, d), *(w, x, y, fun))
                if fmin > loss:
                    fmin = loss
                    x_best = (a, b, c, d)
            y_predit = fun(*x_best, x)
            return fun.__name__, func_desc[fun.__name__], x_best, fmin, y_predit

        x = x_
        x = np.array([0] + list(x))
        x = np.array(list(x) + [x[-1] * 10])

        y = y_
        y = np.array([0] + list(y))
        y = np.array(list(y) + [100])

        w = np.ones(len(x))
        w[2] *= 2

        res = _search(c_start=-4, c_end=-2)
        for i in range(10):
            res = _search(c_start=np.log10(res[2][2] * 0.5), c_end=np.log10(res[2][2] * 1.5))

        return res

    @staticmethod
    def volatility_score_function_fitness(x_, y_):
        y_ = np.array([item / 100.0 for item in y_])
        x = np.array(x_)
        y = np.array(y_)

        if x[0] != 0:
            x = np.array([0] + list(x))
            y = np.array([1] + list(y))

        if x.shape[0] <= 2:
            x = np.array(list(x) + [1.0])
            y = np.array(list(y) + [0])

        last_x, last_y = x[-1], y[-1]
        for i in range(1, 5):
            x = np.array(list(x) + [last_x * 5 * i])
            y = np.array(list(y) + [last_y / 5 / i])

        w = np.ones(len(x))
        w[0] = 5
        w[1] = 3

        res_min = np.inf
        params = None
        best_fun = None

        for fun in [tanh_neg]:
            for method in algorithm:
                print(f"尝试使用函数 {fun.__name__} 和优化方法 {method} ==========================================")
                try:
                    res = minimize(obj_func2, [1, 1, 1, 1], args=(w, x, y, fun), method=method)
                    if res['success'] != True:
                        print(fun.__name__, method, '波动率评分拟合失败')
                        continue
                    else:
                        if res.fun < res_min:
                            params = res.x
                            params[0] *= 100
                            params[-1] *= 100
                            y_predit = fun(*params, x)
                            res_min = res.fun
                            best_fun = fun
                except Exception as e:
                    print(fun.__name__, method, '波动率评分拟合异常', e)
                    # print(traceback.format_exc())

        return best_fun.__name__, func_desc[best_fun.__name__], params, res_min, y_predit

    @staticmethod
    def disperse_score_function_fitness(x_, y_):
        w = np.ones(len(x_))
        x = x_
        y = y_

        a_ = [100]
        b_ = np.linspace(0, 100, 101)
        c_ = np.power(10, np.linspace(-10, 0, 11))
        d_ = np.linspace(0, 5, 101)

        fun = morgan_mercer_flodin
        fmin = np.inf
        x_best = None

        for a, b, c, d in product(a_, b_, c_, d_):
            loss = obj_func((a, b, c, d), *(w, x, y, fun))
            if fmin > loss:
                fmin = loss
                x_best = (a, b, c, d)

        y_predit = fun(*x_best, x)
        return fun.__name__, func_desc[fun.__name__], x_best, fmin, y_predit

    @staticmethod
    def disperse_value_function_fitness():
        # 离散度计算公式拟合，将 -x*log(x) 拟合成 a*x^2 + b*x
        x = np.linspace(1e-6, 1, 100)
        y = -1 * x * np.log(x)
        w = np.ones(x.shape)

        fun = quadratic
        fun_min = np.inf
        bestParams = None

        for method in ['Powell', 'Nelder-Mead', 'CG', 'BFGS']:  # 'L-BFGS-B', 'COBYLA', 'SLSQP'
            try:
                res = minimize(obj_func, [-1, 1], args=(w, x, y, quadratic), method=method)
                if (res['success'] != True):
                    print(fun.__name__, method, '优化失败')
                    continue
                else:
                    if fun_min > res.fun:
                        bestParams = res.x
                        fun_min = res.fun
            except:
                continue

        y_predit = fun(*bestParams, x)
        return fun.__name__, func_desc[fun.__name__], bestParams, np.sum(y_predit - y), y_predit
