# Repository Guidelines

## 项目结构与模块组织
此仓库围绕 Flask 服务构建，主入口 `start.py` 在 6001 端口暴露理财 API，逻辑按职责拆分成服务层、视图层与计算层，方便测试与扩展。
- `server/base.py` 负责应用初始化，`server/service.py` 保持单例服务对象。
- `views/wealth_manage.py` 定义主要 HTTP 路由，依赖服务层返回投资组合结果。
- `calculate/` 存放组合优化、约束和配置（如 `product_class.csv`）。
- `efficient_frontier_API/` 提供前沿曲线样例、脚本与独立文档，请在该目录遵循其 `AGENTS.md`。
- `model/` 汇集评分与适应度函数，`utils/` 放置通用小工具。
- 数据样本与临时日志位于 `data/`、`测试文件夹/`；测试建议放在根目录 `tests/` 下。

## 构建、测试与开发命令
- `python3 -m venv .venv && source .venv/bin/activate`：创建并激活虚拟环境。
- `pip install -r efficient_frontier_API/requirements.txt`：安装依赖（建议 Python 3.7–3.8）。
- `python start.py`：启动 Flask 服务，访问 `http://localhost:6001` 验证接口。
- `python efficient_frontier_API/A01_main_api.py`：运行前沿曲线演示脚本，校验算法输出。
- `pytest -q`：执行所有单测；如需覆盖率可加 `--cov` 选项。

## 编码风格与命名约定
Python 代码遵守 PEP 8 与四空格缩进；模块命名 `lower_snake_case.py`，类使用 `PascalCase`，函数与变量为 `lower_snake_case`，常量大写。优先补充类型注解和清晰的 docstring，复杂逻辑添加简短注释。避免硬编码路径，改用 `config_py.py` 或环境变量，并保持文件为 ASCII。

## 测试指南
统一使用 `pytest`；新增用例放入 `tests/test_*.py`，针对 Flask 视图利用测试客户端模拟请求。算法或优化逻辑需提供示例输入输出，并尽量对关键约束做快照测试。提交前运行 `pytest -q` 确认通过，必要时记录尚未覆盖的风险点。

## 提交与 Pull Request 指南
提交信息遵循 `[scope] verb: summary`（如 `[views] add M4 route`）。PR 描述需说明目的、关联 Issue、验证步骤（示例 JSON、cURL 或截图）及潜在性能或数据影响。涉及数据文件的修改请列出差异摘要。合并前确认依赖文件未误删，保证 lint/测试全部通过。

## 安全与配置提示
切勿提交真实数据库或凭据，示例配置参考 `efficient_frontier_API/Y01_db_config.py` 并在本地覆写。新增敏感文件前更新 `.gitignore`。本仓库默认使用简体中文交流，如在子目录遇到额外 `AGENTS.md`，请优先执行其中指引。
