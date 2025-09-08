# -*- encoding: utf-8 -*-
"""
@File: service.py
@Modify Time: 2025/8/25 15:09
@Author: Kevin-Chen
@Descriptions:
"""
import traceback
import datetime as dt
from model.score_function_fitness import ScoreFunctionFitness


class Service(object):
    def __new__(cls, *args, **kwargs):
        """
        单例模式：处理单例逻辑，减少创建消耗
        :param args:
        :param kwargs:
        :return: Service 单例对象
        """
        if hasattr(cls, "__instance__"):
            return cls.__instance__
        cls.__instance__ = object.__new__(cls)
        return cls.__instance__

    # ================== 财务规划分析 ==================

    def financial_plan_analysis_for_pension(self, current_age, retirement_age, years, cost, *args, **kwargs):
        """养老金规划分析"""
        return {"plan": "pension", "current_age": current_age, "retirement_age": retirement_age}

    def financial_plan_analysis_for_housepurchase(self, year, down_payment, present_value, invest_freq, *args,
                                                  **kwargs):
        """购房规划分析"""
        return {"plan": "housepurchase", "year": year, "down_payment": down_payment}

    def financial_plan_analysis_for_education(self, year, college_term, cost_school, cost_living, *args, **kwargs):
        """教育规划分析"""
        return {"plan": "education", "year": year, "college_term": college_term}

    def financial_plan_analysis_for_investment(self, year, present_value, period_invest, invest_freq, style, goal,
                                               *args, **kwargs):
        """投资规划分析"""
        return {"plan": "investment", "year": year, "goal": goal}

    def financial_plan_analysis_for_investment_user(self, years, present_value, period_invest, invest_freq, style, goal,
                                                    *args, **kwargs):
        """用户投资规划分析"""
        return {"plan": "investment_user", "years": years, "goal": goal}

    # ================== 静态方法 ==================

    @staticmethod
    def get_financial_plan_analysis_by_asset_weight(years: int, present_value: float, period_invest: float, *args,
                                                    **kwargs):
        """按资产权重分析财务计划"""
        return {"plan": "asset_weight", "years": years, "present_value": present_value}

    @staticmethod
    def productClassCfg(prods, srcProds, *args, **kwargs):
        """产品分类配置"""
        return {"func": "productClassCfg", "prods": prods}

    @staticmethod
    def clazzCfgDefScore(srcClazzCfg, clazzCfg, *args, **kwargs):
        """计算类配置得分"""
        return {"func": "clazzCfgDefScore", "srcClazzCfg": srcClazzCfg}

    @staticmethod
    def clazzCfgMaxAbsDiff(srcClazzCfg, clazzCfg, *args, **kwargs):
        """计算类配置最大差异"""
        return {"func": "clazzCfgMaxAbsDiff", "srcClazzCfg": srcClazzCfg}

    @staticmethod
    def callClazzCfg(prods, clazzMap, *args, **kwargs):
        """调用类配置"""
        return {"func": "callClazzCfg", "clazzMap": clazzMap}

    # ================== 投资组合相关 ==================

    def build_investment_portfolio(self, *args, **kwargs):
        """建立投资组合"""
        return {"func": "build_investment_portfolio"}

    def build_investment_portfolio_individual(self, *args, **kwargs):
        """建立个人投资组合"""
        return {"func": "build_investment_portfolio_individual"}

    def build_investment_portfolio_individual_v2(self, *args, **kwargs):
        return {"func": "build_investment_portfolio_individual_v2"}

    def build_investment_portfolio_individual_m4(self, *args, **kwargs):
        """
        构建个人投资组合M4版本

        参数:
            *args: 可变位置参数，传递给投资组合构建函数
            **kwargs: 可变关键字参数，传递给投资组合构建函数

        返回值:
            tuple: 包含以下元素的元组
                - portfolios: 投资组合数据
                - msg: 消息列表
                - indicators: 指标数据
                - msgList: 包含版本信息、参数信息的列表，或错误信息列表
        """
        try:
            # 导入M4版本的个人投资组合构建模块
            from calculate.investment_portfolios_individual_m4 import \
                build_portfolio as build_portfolio_individual

            # 调用底层投资组合构建函数
            print("调用M4版本的个人投资组合构建函数: build_portfolio_individual")
            portfolios, msg, indicators, info = build_portfolio_individual(*args, **kwargs)
            # 在信息列表前添加版本信息
            info = ['版本:v1.1'] + info
            # 构建包含版本信息和参数信息的消息列表
            msgList = [info, args, kwargs]
            return portfolios, msg, indicators, msgList

        except Exception as e:
            # 异常处理：打印异常堆栈信息
            traceback.print_exc()
            # 构建包含错误信息和参数信息的错误消息列表
            msgList = [traceback.format_exc(), args, kwargs]
            return [], ['优化失败'], [], msgList

    # ================== 其他 ==================

    def score_function_fitness(self, *args, **kwargs):
        try:
            func, x, y = kwargs['func'], kwargs['x'], kwargs['y']
            print(dt.datetime.now())
            print(func, x, y)
            if func == 'cash_score_function_fitness':
                all_ret = []
                for i in range(0, 101):
                    new_x = list(x)
                    new_x[1] = i / 100
                    func_name, func_content, params, perc, y_predit = getattr(ScoreFunctionFitness, func)(new_x, y)
                    all_ret.append((func_name, func_content, list(params), perc, list(y_predit)))
                    print(func_name, func_content, params, perc, y_predit)
                return all_ret

            else:
                func_name, func_content, params, perc, y_predit = getattr(ScoreFunctionFitness, func)(x, y)
                print(func_name, func_content, params, perc, y_predit)
                return func_name, func_content, list(params), perc, list(y_predit)

        except Exception as e:
            traceback.print_exc()
            print(traceback.format_exc())
            return {"error": str(e)}

    def kyc_cpk_select(self, *args, **kwargs):
        """KYC 选择"""
        return {"func": "kyc_cpk_select"}

    def backtest(self, actions, productUnivNavList, curUnivList, amount, *args, **kwargs):
        """回测"""
        return {"func": "backtest", "actions": actions, "amount": amount}


s = Service()
