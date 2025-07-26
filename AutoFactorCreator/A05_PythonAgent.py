# -*- encoding: utf-8 -*-
"""
@File: A05_PythonAgent.py
@Modify Time: 2025/7/16 12:52
@Author: Kevin-Chen
@Descriptions: 负责代码审查、缺陷修改、新算子开发与测试的智能体。
"""
import re
import json
import traceback
from AutoFactorCreator.B02_AgentTools import call_llm_api, FileOps


class PythonEngineerAgent:
    """资深Python工程师智能体：负责代码的自动修复与开发。"""

    def __init__(self, temperature: float = 0.3):
        self.temperature = temperature
        self.sys_prompt = f"""你是一位顶级的资深Python工程师，尤其擅长调试和修复数据科学与量化金融领域的代码。你的任务是分析错误信息和相关代码，并提供精确、可执行的修复方案。

你必须严格遵循以下的JSON格式输出你的修复建议，不包含任何额外的解释或Markdown标记：

```json
{{
  "analysis": "（在这里详细分析错误的根本原因，例如：哪个变量类型不匹配，哪个函数调用缺少了参数，或者哪个逻辑判断有缺陷。）",
  "suggestions": [
    {{
      "action": "REPLACE",
      "file_path": "/path/to/your/file.py",
      "old_code_snippet": "（这里是需要被替换的、精确的、包含上下文的多行代码片段。）",
      "new_code_snippet": "（这里是用于替换的、新的、完整的代码片段。）"
    }},
    {{
      "action": "INSERT",
      "file_path": "/path/to/your/file.py",
      "line_number": <line_number_to_insert_after>,
      "code_to_insert": "（这里是要插入的新代码。）"
    }},
    {{
      "action": "DELETE",
      "file_path": "/path/to/your/file.py",
      "line_start": <line_number_to_start_deleting>,
      "line_end": <line_number_to_end_deleting>
    }}
  ]
}}
```

**重要指令:**
1.  **精确性:** `old_code_snippet` 必须与文件中的原始代码完全一致（包括缩进和换行），以确保替换操作能够成功。
2.  **完整性:** `new_code_snippet` 必须是完整的、可以直接替换旧代码的块。
3.  **上下文:** 提供的代码片段应包含足够的上下文，以避免歧义。
4.  **简洁性:** 只提供必要的修复操作。如果一个 `REPLACE` 操作可以解决问题，就不要同时提供 `INSERT` 或 `DELETE`。
5.  **专注:** 你的任务是修复代码，而不是重构或优化。只解决当前报告的错误。
"""

    def fix_code_error(self, traceback_info: str) -> dict:
        """
        分析错误堆栈信息并生成修复建议。

        Args:
            traceback_info (str): 完整的Python错误堆栈跟踪信息。

        Returns:
            dict: 包含代码分析和修复建议的字典。
        """
        # 1. 从traceback中解析出关键信息：错误文件路径和行号
        file_path, line_number, error_message = self._parse_traceback(traceback_info)
        if not file_path:
            return {"error": "无法从traceback中解析出有效的文件路径和行号。", "raw_traceback": traceback_info}

        # 2. 获取错误点周围的代码上下文
        code_context = FileOps.get_code_context(file_path, line_number)

        # 3. 构建用户提示
        user_prompt = f"""### Bug Report

**Error Message:**
```
{error_message}
```

**File Path:** `{file_path}`
**Line Number:** `{line_number}`

**Code Context:**
```python
{code_context}
```

**Full Traceback:**
```
{traceback_info}
```

请根据你的角色和任务指示，分析此错误并提供JSON格式的修复方案。"""

        print("\n--- Python工程师智能体正在分析错误... ---")
        try:
            llm_response_str = call_llm_api(
                sys_prompt=self.sys_prompt,
                prompt=user_prompt,
                temperature=self.temperature
            )
            # 直接解析LLM返回的JSON字符串
            fix_plan = json.loads(llm_response_str)
            return fix_plan
        except Exception as e:
            print(f"处理来自LLM的修复方案时出错: {e}")
            print(traceback.format_exc())
            return {"error": str(e), "raw_response": llm_response_str if 'llm_response_str' in locals() else ""}

    def _parse_traceback(self, traceback_info: str) -> (str, int, str):
        """
        一个简单的解析器，用于从Python的traceback字符串中提取文件路径、行号和错误信息。
        """
        # 匹配 `File "/path/to/file.py", line 123, in function_name`
        file_line_match = re.search(r'File "(.*?)", line (\d+), in', traceback_info)
        if not file_line_match:
            return None, None, traceback_info.strip().split('\n')[-1]

        file_path = file_line_match.group(1)
        line_number = int(file_line_match.group(2))

        # 提取最后一行作为主要的错误信息
        error_message = traceback_info.strip().split('\n')[-1]

        return file_path, line_number, error_message


if __name__ == '__main__':
    # 模拟一个从 A06_CalFactors.py 捕获的错误
    mock_traceback = """Traceback (most recent call last):
  File "/Users/chenjunming/Desktop/KevinGit/PyFinance/AutoFactorCreator/A06_CalFactors.py", line 588, in <module>
    json_report = generate_factor_report(
                  ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chenjunming/Desktop/KevinGit/PyFinance/AutoFactorCreator/A06_CalFactors.py", line 486, in generate_factor_report
    factor_values_array = calculate_factor_values(internal_ast, calculator)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chenjunming/Desktop/KevinGit/PyFinance/AutoFactorCreator/A06_CalFactors.py", line 365, in calculate_factor_values
    factor_values = calculator.calculate_factor(factor_ast)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chenjunming/Desktop/KevinGit/PyFinance/AutoFactorCreator/A06_CalFactors.py", line 52, in calculate_factor
    final_result = self._execute_dag_plan(execution_plan, dag_representation)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chenjunming/Desktop/KevinGit/PyFinance/AutoFactorCreator/A06_CalFactors.py", line 146, in _execute_dag_plan
    result = func(*args)
             ^^^^^^^^^^^
  File "/Users/chenjunming/Desktop/KevinGit/PyFinance/AutoFactorCreator/A02_OperatorLibrary.py", line 820, in rolling_corr
    raise ValueError("Inputs a and b must have the same shape.")
ValueError: Inputs a and b must have the same shape.
"""

    # 1. 实例化工程师智能体
    engineer_agent = PythonEngineerAgent()

    # 2. 调用修复方法
    fix_suggestion = engineer_agent.fix_code_error(mock_traceback)

    # 3. 打印最终的JSON修复建议
    print("\n--- Python工程师的修复建议 (JSON) ---")
    print(json.dumps(fix_suggestion, indent=2, ensure_ascii=False))
