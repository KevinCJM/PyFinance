# -*- encoding: utf-8 -*-
"""
@File: A03_Agents.py
@Modify Time: 2025/7/10 10:05       
@Author: Kevin-Chen
@Descriptions: 
"""

import os
import re
import sys
import json
import random  # 导入random模块用于模拟因子效果
import inspect
import requests
import traceback
import importlib.util
from openai import OpenAI

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
        return {"error": "A02_OperatorLibrary.py not found.", "description": "Error: A02_OperatorLibrary.py not found."}

    spec = importlib.util.spec_from_file_location("A02_OperatorLibrary", operator_file_path)
    if spec is None:
        return {"error": "Could not load A02_OperatorLibrary spec.",
                "description": "Error: Could not load A02_OperatorLibrary spec."}

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(traceback.format_exc())
        return {"error": f"Error executing A02_OperatorLibrary.py: {e}",
                "description": f"Error executing A02_OperatorLibrary.py: {e}"}

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
    递归地将AST中特定键（如'axis', 'ddof', 'window', 'span', 'halflife', 'q'）的值从字符串转换为数字。
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k in ["axis", "ddof", "window", "span", "halflife"] and isinstance(v, str) and v.isdigit():
                new_obj[k] = int(v)
            elif k == "q" and isinstance(v, str):
                try:
                    new_obj[k] = float(v)
                except ValueError:
                    new_obj[k] = v  # Keep as string if not a valid float
            else:
                new_obj[k] = _convert_string_numbers_to_actual_numbers(v)
        return new_obj
    elif isinstance(obj, list):
        return [_convert_string_numbers_to_actual_numbers(elem) for elem in obj]
    else:
        return obj


def _pre_process_llm_response_string(llm_response_str: str) -> str:
    """
    预处理LLM响应字符串，以处理JSON解析前的潜在问题。
    包括：
    1. 修复LaTeX中可能存在的双重转义反斜杠。
    2. 尝试将特定键的未加引号的数字转换为带引号的字符串。
       这是一种脆弱的基于正则表达式的方法，应谨慎使用。
       理想情况下，LLM应直接将这些值作为字符串输出。
    """
    # 1. 修复LaTeX中可能存在的双重转义反斜杠
    if "\\" in llm_response_str:
        llm_response_str = llm_response_str.replace("\\", "\\")

    # 2. 尝试将特定键的未加引号的数字转换为带引号的字符串。
    #    这是针对LLM可能不严格遵循AST示例中数字字符串表示的变通方法。
    keys_to_quote = ["axis", "ddof", "window", "span", "halflife", "q"]
    for key in keys_to_quote:
        # 此正则表达式查找：
        #   - 双引号后跟键名
        #   - 可选的空白字符
        #   - 冒号
        #   - 可选的空白字符
        #   - 一个数字（整数或浮点数）
        #   - 一个前瞻断言，确保其后跟逗号或闭合括号/方括号
        # 这仍然不是万无一失的，但尝试更具体。
        llm_response_str = re.sub(
            rf'("{key}"\s*:\s*)(\d+(\.\d+)?)(?=[\],}}])',
            r'\1"\2"',
            llm_response_str
        )
    return llm_response_str


def call_llm_api(sys_prompt: str, prompt: str, temperature: float = 0.7) -> str:
    """
    调用大模型API接口。

    Args:
        sys_prompt (str): System角色的提示。
        prompt (str): User角色的提示。
        temperature (float): 模型生成时的温度参数，控制随机性。

    Returns:
        str: 大模型返回的文本内容。

    Raises:
        FileNotFoundError: 如果config.json文件不存在。
        KeyError: 如果config.json中缺少必要的配置项。
        Exception: 如果API调用失败或返回非预期结果。
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model = config.get("LLM_MODEL")
    api_key = config.get("API_KEY")
    base_url = config.get("LLM_URL")

    if not all([model, api_key, base_url]):
        raise KeyError("Missing one or more required keys (LLM_MODEL, API_KEY, LLM_URL) in config.json")

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    if 'lightcode-uis' in base_url:
        # 通过lightcode-ui接口调用大模型API
        print("Using lightcode-ui API...")
        headers = {'Accept': "*/*", 'Authorization': f"Bearer {api_key}", 'Content-Type': "application/json"}
        payload = {"model": model, "messages": messages, "stream": False}
        resp = None
        try:
            resp = requests.post(base_url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()  # 检查HTTP响应状态码
            response_data = resp.json()
            # 解析 返回结果
            json_inner = response_data["choices"][0]["message"]["content"]
            return re.sub(r"^```json\n|```$", "", json_inner.strip(), flags=re.MULTILINE)
        except requests.exceptions.RequestException as e:
            raise Exception(f"lightcode-ui API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected response format from lightcode-ui API: {e} Response: {resp.text}")

    else:
        # 通过OpenAI接口调用大模型API
        print("Using OpenAI API...")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
            llm_response_content = resp.choices[0].message.content
            # 移除Markdown代码块标记
            if llm_response_content.startswith("```json") and llm_response_content.endswith("```"):
                llm_response_content = llm_response_content[len("```json"): -len("```")].strip()
            return llm_response_content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")


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
        "example_usage": "一个使用新函数的代码示例，例如：new_func(data, param1=10)"
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

    def propose_factor_or_operator(self) -> dict:
        """
        构思新的因子计算逻辑或提出新算子需求。

        Args:
            current_evaluation_result (dict): 当前因子的评估结果。

        Returns:
            dict: 包含因子计算逻辑 (AST, LaTeX) 或新算子需求 (CreateNewCalFunc)。
        """

        user_prompt = """### 请构思一个新的金融因子。"""

        if self.history_factor_results:
            user_prompt += (f"\n\n### 历史因子成果摘要："
                            f"\n{json.dumps(self.history_factor_results, indent=2, ensure_ascii=False)}")

        user_prompt += "\n请根据这些结果，提出一个改进的因子或一个全新的因子。"
        print(user_prompt)
        print("\n--- 金融数学家智能体正在构思... ---")
        llm_response_str = None
        try:
            llm_response_str = call_llm_api(
                sys_prompt=self.sys_prompt,
                prompt=user_prompt,
                temperature=self.temperature
            )
            # 在尝试解析JSON之前，对LLM响应字符串进行预处理
            llm_response_str = _pre_process_llm_response_string(llm_response_str)
            print(llm_response_str)
            response_json = json.loads(llm_response_str)
            # 将AST中的字符串数字转换为实际数字
            response_json = _convert_string_numbers_to_actual_numbers(response_json)

            # 验证JSON结构
            if "action" in response_json and response_json["action"] == "CreateNewCalFunc":
                required_keys = ["function_name", "description", "example_usage"]
                if not all(key in response_json for key in required_keys):
                    raise ValueError("Invalid CreateNewCalFunc JSON format.")
                print("金融数学家智能体提出了新算子需求。")
                return response_json
            elif "des" in response_json and "ast" in response_json:
                print("金融数学家智能体提出了新的因子计算逻辑。")
                # 可以在这里添加更复杂的AST结构验证
                return response_json
            else:
                raise ValueError("LLM response is not a valid factor or CreateNewCalFunc JSON.")

        except json.JSONDecodeError as e:
            print(f"Error: LLM response is not valid JSON. {e}\nResponse: {llm_response_str}")
            print(traceback.format_exc())
            return {"error": "Invalid JSON response from LLM", "raw_response": llm_response_str}
        except ValueError as e:
            print(f"Error: Invalid LLM response structure. {e}\nResponse: {llm_response_str}")
            print(traceback.format_exc())
            return {"error": "Invalid LLM response structure", "raw_response": llm_response_str}
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            print(f"Response: {llm_response_str}")
            print(traceback.format_exc())
            return {"error": f"LLM API call failed: {e}"}

    def add_history_factor_result(self, factor_result: dict):
        """
        添加历史因子成果摘要，用于LLM的迭代学习。
        """
        # 可以在这里对factor_result进行摘要处理，避免历史信息过长导致token超限
        self.history_factor_results.append(factor_result)


# 示例用法 (仅用于测试)
if __name__ == "__main__":
    # --- 测试 _get_operator_descriptions ---
    print("\n--- Testing _get_operator_descriptions ---")
    ops_desc = _get_operator_descriptions()
    print(ops_desc)

    # --- 金融数学家智能体三轮模拟 ---
    print("\n--- 金融数学家智能体三轮模拟 ---")
    fm_agent = FinancialMathematicianAgent(temperature=0.7)

    for i in range(1, 4):  # 模拟3轮
        print(f"\n--- 第 {i} 轮因子构思 ---")
        proposed_output = fm_agent.propose_factor_or_operator()
        print(proposed_output)
        # 提取des和ast
        the_ast = proposed_output.get("ast")
        the_des = proposed_output.get("des")
        print("金融数学家智能体输出:", json.dumps(proposed_output, indent=2, ensure_ascii=False))

        # 模拟因子评估结果
        rank_ic = round(random.uniform(0.01, 0.15), 4)  # 模拟Rank IC
        comment = "表现良好" if rank_ic > 0.05 else "表现一般"
        current_eval_result = {
            "factor_name": f"Factor_Round_{i - 1}",
            "rank_ic": rank_ic,
            "t_stat": round(random.uniform(1.5, 3.0), 2),
            "comment": comment
        }
        # 将本次构思的因子（如果不是新算子需求）添加到历史记录
        fm_agent.add_history_factor_result({
            "factor_name": f"Factor_Round_{i}",
            "des": the_des,
            "ast": the_ast,
            "rank_ic": current_eval_result["rank_ic"] if current_eval_result else None,  # 第一次没有评估结果
            "comment": current_eval_result["comment"] if current_eval_result else None,  # 第一次没有评估结果
        })
