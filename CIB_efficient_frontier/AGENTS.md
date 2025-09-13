# Repository Guidelines

## 项目结构与模块组织
- 根目录 Python 脚本为主要入口：
  - `B**_*.py`：投资组合/有效前沿计算（例：`B01_random_weights.py`）。
  - `T**_*.py`：工具与可视化（例：`T03_show_index_nv.py`）。
  - `local_test_*.py`：本地实验脚本。
- 数据：Excel 文件（如 `历史净值数据.xlsx` 及变体）。
- 输出：HTML 报告与导出表格（如 `efficient_frontier.html`）。

## 构建、测试与开发命令
- 解释器（默认）：`/Users/chenjunming/Desktop/myenv_312/bin/python3.12`
- `pip` 路径：`/Users/chenjunming/Desktop/myenv_312/bin/pip`
- 安装依赖（按需）：`/Users/chenjunming/Desktop/myenv_312/bin/pip install numpy pandas matplotlib scipy cvxpy`
- 运行脚本：`/Users/chenjunming/Desktop/myenv_312/bin/python3.12 B01_random_weights.py`
- 提示：将输入数据置于仓库根目录或通过参数/环境变量传入路径。

## 代码风格与命名规范
- Python 3.12；4 空格缩进；遵循 PEP 8。
- 函数/变量用 `snake_case`；类用 `CapWords`；常量用全大写加下划线。
- 模块应职责单一；偏向小而清晰的函数。
- 可选工具：`black -l 88` 与 `isort`（如需要可在单独 PR 中引入）。

## 测试指南
- 目前无正式测试套件；推荐引入 `pytest`：
  - 目录：`tests/`；命名：`test_*.py`。
  - 运行：`/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest`。
- 对随机过程设定固定种子；对关键指标添加断言；对 HTML/Excel 产出进行快照比对。

## 提交与 Pull Request 规范
- 提交信息使用祈使句、简洁、可中英混合：
  - 示例：`B07: 提升再平衡稳定性`，`T02: 修正收益率计算`，`B01: rename inputs`。
- PR 需包含：变更目的、方案概述、影响脚本、样例输出（截图/HTML）、数据假设/前置条件，并关联 Issue（如有）。

## 安全与配置提示
- 将 Excel 输入视为不可信：校验列名/日期/数值范围。
- 避免提交敏感数据；如需机密配置使用 `.env`（不提交）。
- 将路径、随机种子等通过参数或环境变量配置；默认固定种子以便复现。

## Agent 专用说明
- 沟通语言：中文。
- 默认 Python 解释器：`/Users/chenjunming/Desktop/myenv_312/bin/python3.12`。
- 变更应尽量“手术式”；不要随意重命名数据文件。如需重构，优先提议引入 `data/`（输入）与 `reports/`（输出）并在单独 PR 讨论。

## Git 忽略与提交流水线
- 已在 `.gitignore` 中忽略：`*.csv`、`*.xlsx`、`*.html`、`local_test_*.py`。
- 已提供 `pre-commit` 钩子阻止上述文件被提交：`.githooks/pre-commit`。
- 启用钩子：
  - `git config core.hooksPath .githooks`
  - `chmod +x .githooks/pre-commit`
  - 触发时会给出中文提示并阻止提交。
