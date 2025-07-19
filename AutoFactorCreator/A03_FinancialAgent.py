# -*- encoding: utf-8 -*-
"""
@File: A03_FinancialAgent.py
@Modify Time: 2025/7/16 11:00
@Author: Kevin-Chen
@Descriptions: 包含“金融数学家智能体”的定义和逻辑。
"""

import os
import re
import sys
import json
import inspect
import traceback
import importlib.util

# 从新的公用工具模块导入大模型调用函数
from AutoFactorCreator.B02_AgentTools import call_llm_api

# 确保A02_OperatorLibrary在Python路径中，以便inspect可以找到它
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _get_operator_descriptions():
    """
    动态读取 A02_OperatorLibrary.py 文件，提取所有算子函数的名称、文档字符串和参数信息，并进行分类和格式化。

    Returns:
        dict: 包含两类算子的字典，以及格式化的字符串描述。
    """
    operator_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A02_OperatorLibrary.py")

    if not os.path.exists(operator_file_path):
        return {"error": "A02_OperatorLibrary.py not found."}

    spec = importlib.util.spec_from_file_location("A02_OperatorLibrary", operator_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    factor_calculation_operators = []
    preprocessing_operators = []

    # 提取源码内容用于注释分类
    with open(operator_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    factor_calc_section = re.search(r'# --- 基础数学算子 ---\n(.*?)(?=# --- 数据预处理与因子结果处理算子 ---\n|\Z)',
                                    content, re.DOTALL)
    preprocessing_section = re.search(
        r'# --- 数据预处理与因子结果处理算子.*?---\n(.*?)(?=\n# |\Z)',
        content, re.DOTALL)

    factor_calc_names = set()
    if factor_calc_section:
        factor_calc_names.update(re.findall(r'def (\w+)\(', factor_calc_section.group(1)))

    preprocessing_names = set()
    if preprocessing_section:
        preprocessing_names.update(re.findall(r'def (\w+)\(', preprocessing_section.group(1)))

    def format_docstring(name, obj):
        doc = inspect.getdoc(obj) or "(No description available)"
        signature = inspect.signature(obj)
        params = ", ".join(
            [f"{p.name}: {p.annotation.__name__ if p.annotation != inspect.Parameter.empty else 'Any'}"
             for p in signature.parameters.values()]
        )

        lines = [f"- {name}({params}):"]
        doc_lines = doc.splitlines()

        desc_lines = []
        param_lines = []
        return_lines = []
        section = 'desc'

        for line in doc_lines:
            line = line.strip()
            if not line:
                continue  # 忽略空行

            if line.startswith("参数:") or line.startswith("参数说明:"):
                section = 'param'
                continue
            elif line.startswith("返回:") or line.startswith("返回值:"):
                section = 'return'
                continue

            if section == 'desc':
                desc_lines.append(line)
            elif section == 'param':
                param_lines.append("    - " + line)
            elif section == 'return':
                return_lines.append("    - " + line)

        if desc_lines:
            lines.append("  - 功能描述: " + " ".join(desc_lines))
        if param_lines:
            lines.append("  - 参数:")
            lines.extend(param_lines)
        if return_lines:
            lines.append("  - 返回:")
            lines.extend(return_lines)

        return "\n".join(lines)

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name in ['neutralize', 'winsorize', 'clip', 'fill_na', 'cross_sectional_rank', 'cross_sectional_scale']:
            continue  # 跳过这些函数，因为它们是预处理算子
        if not name.startswith("_") and obj.__module__ == module.__name__:
            formatted = format_docstring(name, obj)
            if name in factor_calc_names:
                factor_calculation_operators.append(formatted)
            elif name in preprocessing_names:
                preprocessing_operators.append(formatted)
            else:
                factor_calculation_operators.append(formatted)

    description_str = """
> 可用于因子计算的算子 (Factor Calculation Operators), 这些算子可以直接用于构建金融因子计算逻辑:

""" + "\n\n".join(factor_calculation_operators)

    return {
        "factor_calculation_operators": factor_calculation_operators,
        "preprocessing_operators": preprocessing_operators,
        "description": description_str
    }


def _convert_string_numbers_to_actual_numbers(obj):
    """
    递归地将AST中特定键的值从字符串转换为数字。
    """
    if isinstance(obj, dict):
        return {k: _convert_string_numbers_to_actual_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_string_numbers_to_actual_numbers(elem) for elem in obj]
    elif isinstance(obj, str) and obj.isdigit():
        return int(obj)
    try:
        return float(obj)
    except (ValueError, TypeError):
        return obj


def _pre_process_llm_response_string(llm_response_str: str) -> str:
    """
    预处理LLM响应字符串，以处理JSON解析前的潜在问题。
    """
    # 移除Markdown代码块标记
    processed_str = re.sub(r"^```json\n|```$", "", llm_response_str.strip(), flags=re.MULTILINE)
    # 修复可能存在的双重转义
    processed_str = processed_str.replace("\\", "\\")
    return processed_str


def _format_analysis_for_prompt(analysis_report: dict) -> str:
    """
    将分析师智能体返回的JSON报告格式化为清晰的Markdown，用于提示。
    """
    if not isinstance(analysis_report, dict) or "因子性能分析" not in analysis_report:
        return "- 上一轮分析无效或缺失。"

    performance = analysis_report.get("因子性能分析", "N/A")
    suggestions = analysis_report.get("因子升级建议", "N/A")

    # 将建议从字符串（可能包含换行）转换为Markdown列表
    if isinstance(suggestions, str):
        suggestion_list = [f"  - {line.strip()}" for line in suggestions.split('\n') if line.strip()]
        suggestions_md = "\n".join(suggestion_list)
    else:
        suggestions_md = "  - 建议格式不正确。"

    return f"""#### 因子性能分析:
- {performance}

#### 因子升级建议:
{suggestions_md}
    """


class FinancialMathematicianAgent:
    """金融数学家智能体：负责因子构思、逻辑生成和新算子需求提出。"""

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        self.operator_descriptions_dict = _get_operator_descriptions()
        self.operator_descriptions = self.operator_descriptions_dict.get("description", "")
        self.history_factor_results = []  # 存储历史因子成果摘要
        self.sys_prompt = f"""你是一个顶级的金融数学家，精通量化投资和因子模型。你的任务是根据提供的算子库和可用数据，构思用于预测未来收益的金融因子计算逻辑。因子逻辑需要足够的创新,但也要遵循金融学的基本原理和数学的计算逻辑。
你必须严格按照以下JSON格式返回你的构思：

## 1. 因子计算逻辑:
    ```json
    {{
      "des": "对因子逻辑的详细解释，包括其经济学含义和预期效果。",
      "ast": "AST语法树，表示因子计算的步骤，使用算子库中的函数名。例如：{{\"func\": \"add\", \"args\": {{\"a\": {{\"var\": \"close_df\"}}, \"b\": {{\"func\": \"std_dev\", \"args\": {{\"data\": {{\"var\": \"vol_df\"}}, \"axis\": \"0\", \"ddof\": \"1\"}}}}}}}}}}"
    }}
    ```
    AST语法树的结构必须是嵌套的字典，其中包含 'func' (函数名) 或 'var' (变量名)，以及 'args' (参数列表)。
    **重要提示：** `args` 必须是一个字典，其中键是算子的参数名，值是参数的具体内容（可以是嵌套的AST结构、变量或标量）。

## 2. 新算子需求:
    如果你认为现有算子不足以表达你的因子构思，你可以提出一个新算子的需求。请严格按照以下JSON格式返回你的需求：
    ```json
    {{
        "action": "CreateNewCalFunc",
        "function_name": "新函数名",
        "description": "新函数的功能描述，包括输入、输出、数学逻辑等详细信息。",
        "example_usage": "一个使用新函数的代码示例"
    }}
    ```

## 3. 可用的数据变量:
> 以下所有的数据变量都是二维数组: 行表示时间, 列表示产品。
- `log_return`: **个股**日对数收益率数据
- `high`: **个股**最高价数据
- `low`: **个股**最低价数据
- `vol`: **个股**成交量数据
- `amount`: **个股**成交额数据
- `close`: **个股**收盘价数据
- `open`: **个股**开盘价数据
- `benchmark_ew`: **等权重组合基准**的日对数收益率
- `benchmark_min_var`: **滚动最小方差组合基准**的日对数收益率
- `benchmark_erc`: **滚动等风险贡献组合基准**的日对数收益率

## 4. 可用的算子库:
{self.operator_descriptions}

## 5. 关于算子中的 `axis` 参数:
- `axis=0` 通常表示对时间序列（按行，即沿着日期轴）进行操作，例如计算过去N天的移动平均。
- `axis=1` 通常表示对横截面（按列，即沿着金融产品轴）进行操作，例如计算某个日期所有金融产品的排名。
请根据你的因子构思，合理选择 `axis` 参数的值。

## 6. 注意事项:
- 你的输出必须是有效的JSON格式，且只包含JSON内容，不要有任何额外文字。
- 优先使用现有算子构建因子。只有在现有算子确实无法表达你的构思时，才提出新算子需求。
- 构思的因子应尽可能简洁但有效，避免过度复杂化。
- 确保AST语法树中的函数名和变量名与算子库中提供的名称完全一致。
        """
        print(self.sys_prompt)

    def propose_factor_or_operator(self, initial_idea: str = None) -> dict:
        """
        构思新的因子计算逻辑或提出新算子需求。

        Args:
            initial_idea (str, optional): 一个可选的初始研究方向，仅在没有历史记录时使用。

        Returns:
            dict: 包含因子计算逻辑 (AST) 或新算子需求 (CreateNewCalFunc)。
        """
        if self.history_factor_results:
            # --- Case 1: 有历史记录，进行迭代 ---
            history_md = "\n\n---\n".join(self.history_factor_results)
            user_prompt = f"""### 历史因子分析与迭代建议摘要：\n{history_md}\n\n请严格遵照你的角色指示，基于以上分析，提出一个全新的、更优的因子计算逻辑。"""
        elif initial_idea:
            # --- Case 2: 没有历史记录，但有初始研究方向 ---
            user_prompt = f"""### 请围绕以下研究方向，构思一个全新的金融因子：\n- **研究方向**: {initial_idea}\n\n请严格遵照你的角色指示，设计一个与此方向相关的、具有创新性的因子计算逻辑。"""
        else:
            # --- Case 3: 没有历史记录，也没有初始方向，自由发挥 ---
            user_prompt = "### 请发挥你的创造力，构思一个全新的、具有潜力的金融因子。请严格遵照你的角色指示，返回你的构思。"

        print("\n--- 发送给LLM的完整提示: ---")
        print(user_prompt)
        print("\n--- 金融数学家智能体正在构思... ---")
        print("---------------------------------")
        try:
            llm_response_str = call_llm_api(
                sys_prompt=self.sys_prompt,
                prompt=user_prompt,
                temperature=self.temperature
            )
            processed_str = _pre_process_llm_response_string(llm_response_str)
            response_json = json.loads(processed_str)
            return _convert_string_numbers_to_actual_numbers(response_json)
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            print(traceback.format_exc())
            return {"error": str(e)}

    def add_history_factor_result(self, factor_ast: dict, analysis_report_dic: dict):
        """
        将包含深度分析的历史因子成果添加到历史记录中。
        """
        formatted_analysis = _format_analysis_for_prompt(analysis_report_dic)
        history_entry = f"""**上一轮因子结构 (AST):**
```json
{json.dumps(factor_ast, indent=2)}
```
**分析师评估与建议:**
{formatted_analysis}
"""
        self.history_factor_results.append(history_entry)


if __name__ == '__main__':
    # 模拟 A04_AnalysisAgent 的功能
    from AutoFactorCreator.A04_AnalysisAgent import AnalysisAgent

    print("\n--- 金融数学家智能体与分析师智能体协同测试启动 ---")

    # 1. 实例化智能体
    fm_agent = FinancialMathematicianAgent(temperature=0.7)
    analyst_agent = AnalysisAgent(temperature=0.5)

    # 2. 模拟三轮因子生成 -> 评估 -> 分析 -> 迭代的完整循环
    # 初始的因子AST，用于启动第一轮
    current_ast = {
        "func": "subtract",
        "args": {
            "a": {"var": "log_return"},
            "b": {"var": "benchmark_ew"}
        }
    }

    for i in range(1, 4):
        print(f"\n================== 第 {i} 轮完整循环 ==================")

        # 2.1. (模拟) 调用 A06_CalFactors.py 对 current_ast 进行评估
        print(f"\n--- 步骤1: (模拟)评估因子: {json.dumps(current_ast, indent=2)} ---")
        # 这里使用您提供的示例评估报告作为模拟结果
        mock_evaluation_report = {
            "ic_analysis": {"ic_mean": 0.0088, "ic_std": 0.226, "icir": 0.039, "p_value": 0.045},
            "rank_ic_analysis": {"rank_ic_mean": 0.0176, "rank_ic_std": 0.251, "rank_icir": 0.070},
            "turnover_analysis": {"mean_turnover": 0.0064},
            "group_return_analysis": {"monotonicity": False},
            "long_short_portfolio_analysis": {"sharpe_ratio": -0.82}
        }
        print("模拟评估报告 (来自A05):", json.dumps(mock_evaluation_report, indent=2, ensure_ascii=False))

        # 2.2. 调用分析师智能体对评估报告进行深度分析
        print("\n--- 步骤2: 分析师智能体进行深度分析 ---")
        analysis_report = analyst_agent.analyze_factor_performance(
            factor_ast=current_ast,
            evaluation_report=mock_evaluation_report
        )
        print("分析师报告 (来自A04):", json.dumps(analysis_report, indent=2, ensure_ascii=False))

        # 2.3. 将分析师的报告添加到金融数学家的历史记录中
        fm_agent.add_history_factor_result(current_ast, analysis_report)
        print("\n--- 步骤3: 金融数学家接收分析报告并构思新因子 ---")

        # 2.4. 金融数学家基于深度分析，提出新的因子构思
        new_factor_proposal = fm_agent.propose_factor_or_operator()
        print("\n金融数学家的新构思:", json.dumps(new_factor_proposal, indent=2, ensure_ascii=False))

        if 'ast' in new_factor_proposal:
            # 更新当前AST，用于下一轮循环
            current_ast = new_factor_proposal['ast']
        else:
            print("金融数学家提出了新算子需求或返回了错误，测试终止。")
            break

    print("\n--- 协同测试结束 ---")
