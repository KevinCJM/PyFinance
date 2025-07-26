# -*- encoding: utf-8 -*-
"""
@File: B02_AgentTools.py
@Modify Time: 2025/7/16 10:00
@Author: Kevin-Chen
@Descriptions: 提供给智能体使用的公用工具, 包括: 大模型调用, 文件操作, 代码执行与测试等。
"""
import os
import re
import json
import requests
import subprocess
from openai import OpenAI


def call_llm_api(sys_prompt: str, prompt: str, temperature: float = 0.7) -> str:
    """
    调用大模型API接口。
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
        raise KeyError("Missing one or more required keys in config.json")

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    if 'lightcode-uis' in base_url:
        headers = {'Accept': "*/*", 'Authorization': f"Bearer {api_key}", 'Content-Type': "application/json"}
        payload = {"model": model, "messages": messages, "stream": False}
        try:
            resp = requests.post(base_url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()
            response_data = resp.json()
            json_inner = response_data["choices"][0]["message"]["content"]
            return re.sub(r"^```json\n|```$", "", json_inner.strip(), flags=re.MULTILINE)
        except Exception as e:
            raise Exception(f"lightcode-ui API request failed: {e}. Response: {resp.text if 'resp' in locals() else 'N/A'}")
    else:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(model=model, temperature=temperature, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")


class FileOps:
    """封装了所有文件操作的工具类"""

    @staticmethod
    def read(file_path: str) -> str:
        """读取并返回文件的全部内容。"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {file_path}: {e}"

    @staticmethod
    def write(file_path: str, content: str) -> str:
        """将内容完全写入（或覆写）到指定文件。"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing to file {file_path}: {e}"

    @staticmethod
    def replace_snippet(file_path: str, old_snippet: str, new_snippet: str) -> str:
        """在文件中查找并替换一个精确的代码片段。"""
        try:
            original_content = FileOps.read(file_path)
            if old_snippet not in original_content:
                return f"Error: old_snippet not found in {file_path}. Replacement failed."
            
            new_content = original_content.replace(old_snippet, new_snippet, 1)
            return FileOps.write(file_path, new_content)
        except Exception as e:
            return f"Error replacing snippet in {file_path}: {e}"

    @staticmethod
    def get_code_context(file_path: str, line_number: int, context_lines: int = 10) -> str:
        """从文件中读取并提取特定行号周围的代码片段。"""
        if not os.path.exists(file_path):
            return f"错误: 文件未找到于 {file_path}"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            start = max(0, line_number - 1 - context_lines)
            end = min(len(lines), line_number + context_lines)
            snippet = []
            for i in range(start, end):
                prefix = ">> " if i + 1 == line_number else "   "
                snippet.append(f"{prefix}{i + 1:4d}| {lines[i].rstrip()}")
            return "\n".join(snippet)
        except Exception as e:
            return f"错误: 读取文件或提取上下文时失败: {e}"


class CodeExecutor:
    """封装了代码执行与测试的工具类"""

    @staticmethod
    def run_script(script_path: str, args: dict = None) -> dict:
        """
        执行一个Python脚本并返回其结果。

        Args:
            script_path (str): 要执行的脚本的绝对路径。
            args (dict, optional): 传递给脚本的命令行参数字典。
                                   例如: {'--input': 'data.csv', '--rate': 0.1}

        Returns:
            dict: 一个包含执行结果的字典。
        """
        if not os.path.exists(script_path):
            return {"success": False, "return_code": -1, "stdout": "", "stderr": f"Error: Script not found at {script_path}"}

        command = ['python', script_path]
        if args:
            for key, value in args.items():
                command.append(str(key))
                command.append(str(value))

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )
            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr
            }
        except Exception as e:
            return {"success": False, "return_code": -1, "stdout": "", "stderr": f"Error executing script {script_path}: {e}"}
