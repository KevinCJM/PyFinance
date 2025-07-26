# -*- encoding: utf-8 -*-
"""
@File: A04_AnalysisAgent.py
@Modify Time: 2025/7/16 10:30
@Author: Kevin-Chen
@Descriptions: 负责对因子评估报告进行分析、解读，并提出迭代方向的智能体。
"""

import json
import traceback

# 从公用工具模块导入大模型调用函数
from AutoFactorCreator.B02_AgentTools import call_llm_api


class AnalysisAgent:
    """因子分析师智能体：负责解读因子评估报告，并提出改进建议。"""

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
        self.sys_prompt = f"""你是一位顶级的量化分析师（Quant Analyst），你的核心任务是深入分析一个金融因子的性能，并为其未来的优化方向提供清晰、可执行的建议。

你将收到一个包含两部分的JSON输入：
1.  `factor_ast`: 因子的数学结构，以抽象语法树（AST）的形式表示。
2.  `evaluation_report`: 该因子的详细回测评估报告。

你的分析需要结合这两部分信息。例如，你需要思考AST的结构是否合理（如时间尺度是否匹配、是否包含非平稳序列），并从评估报告中寻找证据来支撑你的观点（如IC值低、单调性差等）。

你的输出必须严格遵循以下JSON格式，不包含任何额外的解释或Markdown标记：

```json
{{
  "因子性能分析": "（在这里填写你对因子整体表现的专业、简洁的分析。你需要综合评估其预测能力、稳定性、风险收益特征和交易成本等。）",
  "因子升级建议": "（在这里填写具体、可操作的改进建议。例如：修改AST的某个部分、尝试不同的算子、增加数据预处理步骤等。）"
}}
```

**关键评估指标解读指南:**
- **ic_mean / rank_ic_mean**: 预测能力的核心。绝对值越高越好，通常认为 > 0.03 才具备初步研究价值。
- **icir**: 预测能力的稳定性。越高越好，通常 > 0.5 被认为是稳定的。
- **p_value**: 统计显著性。通常 < 0.05 才有意义，但必须结合IC的绝对值来看。
- **monotonicity**: 因子排序的有效性。`true` 表示因子值越高的资产，其未来收益也越高（或越低），这是非常理想的特性。
- **sharpe_ratio**: 多空组合的风险调整后收益。越高越好，> 1.0 通常被认为是优秀的。
- **mean_turnover**: 换手率，代表交易成本。越低越好。

请基于以上指南，进行专业、深刻的分析。
        """

    def analyze_factor_performance(self, factor_ast: dict, evaluation_report: dict) -> dict:
        """
        分析因子表现并生成报告。

        Args:
            factor_ast (dict): 因子的AST表达式。
            evaluation_report (dict): A05_CalFactors.py生成的因子评估报告。

        Returns:
            dict: 包含分析和建议的字典。
        """
        # 构建一个清晰的用户提示，将AST和评估报告整合在一起
        user_prompt = f"""请分析以下金融因子：

### 1. 因子计算逻辑 (AST):
```json
{json.dumps(factor_ast, indent=2)}
```

### 2. 因子评估报告:
```json
{json.dumps(evaluation_report, indent=2)}
```

请根据你的角色和任务指示，返回一份包含深入分析和具体改进建议的JSON报告。"""

        print("\n--- 因子分析师智能体正在分析报告... ---")
        try:
            llm_response_str = call_llm_api(
                sys_prompt=self.sys_prompt,
                prompt=user_prompt,
                temperature=self.temperature
            )
            # 直接解析LLM返回的JSON字符串
            analysis_json = json.loads(llm_response_str)
            return analysis_json
        except Exception as e:
            print(f"Error processing analysis from LLM: {e}")
            print(traceback.format_exc())
            return {"error": str(e), "raw_response": llm_response_str if 'llm_response_str' in locals() else ""}


if __name__ == '__main__':
    # 1. 准备您提供的示例输入数据
    mock_factor_ast = {
        "func": "subtract",
        "args": {
            "a": {
                "func": "cumulative_sum",
                "args": {
                    "data": {
                        "func": "excess_return",
                        "args": {
                            "data": {"var": "log_return"},
                            "benchmark_data": {"var": "benchmark_ew"},
                            "axis": 0
                        }
                    },
                    "axis": 0
                }
            },
            "b": {
                "func": "rolling_sum",
                "args": {
                    "data": {
                        "func": "subtract",
                        "args": {
                            "a": {
                                "func": "exponential_moving_average",
                                "args": {"data": {"var": "log_return"}, "span": 20, "axis": 0}
                            },
                            "b": {
                                "func": "ts_std",
                                "args": {"data": {"var": "log_return"}, "window": 20, "axis": 0}
                            }
                        }
                    },
                    "window": 5,
                    "axis": 0
                }
            }
        }
    }

    mock_evaluation_report = {
        "ic_analysis": {
            "ic_mean": 0.009719802648003992,
            "ic_std": 0.30304356079845873,
            "icir": 0.0320739454829341,
            "t_statistic": 1.6335685835362843,
            "p_value": 0.10247087044808578
        },
        "rank_ic_analysis": {
            "rank_ic_mean": 0.02071189884704673,
            "rank_ic_std": 0.3535736419127014,
            "rank_icir": 0.05857874114994288
        },
        "turnover_analysis": {
            "mean_turnover": 0.027199959716040962
        },
        "group_return_analysis": {
            "mean_group_returns": {
                "0.0": 0.00010294309918678793,
                "1.0": 1.2460425488458934e-05,
                "2.0": 0.00027306643382554214,
                "3.0": 0.0002292898127415189,
                "4.0": 0.000185913999500191
            }
        },
        "long_short_portfolio_analysis": {
            "sharpe_ratio": 0.39898983599972926
        }
    }

    # 2. 实例化分析师智能体
    analyst_agent = AnalysisAgent()

    # 3. 调用分析方法
    final_analysis_report = analyst_agent.analyze_factor_performance(
        factor_ast=mock_factor_ast,
        evaluation_report=mock_evaluation_report
    )

    # 4. 打印最终的JSON分析报告
    print("\n--- 最终分析报告 (JSON) ---")
    print(json.dumps(final_analysis_report, indent=2, ensure_ascii=False))
