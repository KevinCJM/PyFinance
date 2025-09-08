# -*- encoding: utf-8 -*-
"""
@File: service.py
@Modify Time: 2025/8/25 15:09       
@Author: Kevin-Chen
@Descriptions: 
"""


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

    def build_investment_portfolio_individual_m4(self, *args, **kwargs):
        """建立个人投资组合 M4 版本"""
        return {"func": "build_investment_portfolio_individual_m4"}

    # ================== 其他 ==================

    def score_function_fitness(self, *args, **kwargs):
        """打分函数"""
        return {"func": "score_function_fitness"}

    def kyc_cpk_select(self, *args, **kwargs):
        """KYC 选择"""
        return {"func": "kyc_cpk_select"}

    def backtest(self, actions, productUnivNavList, curUnivList, amount, *args, **kwargs):
        """回测"""
        return {"func": "backtest", "actions": actions, "amount": amount}
