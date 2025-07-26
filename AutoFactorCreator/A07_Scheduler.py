# -*- encoding: utf-8 -*-
"""
@File: A07_Scheduler.py
@Modify Time: 2025/7/16 12:54
@Author: Kevin-Chen
@Descriptions: 调度器, 协调各个智能体之间的任务安排, 并且分配计算任务
"""
import os
import json
import logging
import tempfile
import shlex

# 导入项目中的核心模块
from AutoFactorCreator.A03_FinancialAgent import FinancialMathematicianAgent
from AutoFactorCreator.A04_AnalysisAgent import AnalysisAgent
from AutoFactorCreator.A05_PythonAgent import PythonEngineerAgent
from AutoFactorCreator.B02_AgentTools import FileOps, CodeExecutor

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Scheduler:
    """
    自动化因子挖掘流程的调度器。
    负责协调金融数学家、计算引擎和分析师智能体，形成一个完整的、自动化的因子发现与迭代闭环。
    """

    def __init__(self, max_iterations: int, performance_thresholds: dict):
        """
        初始化调度器。
        """
        self.max_iterations = max_iterations
        self.performance_thresholds = performance_thresholds
        self.fm_agent = FinancialMathematicianAgent(temperature=0.7)
        self.analyst_agent = AnalysisAgent(temperature=0.5)
        self.engineer_agent = PythonEngineerAgent(temperature=0.3)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cal_factor_script_path = os.path.join(script_dir, 'A06_CalFactors.py')
        self.data_dir = os.path.join(script_dir, '..', 'Data', 'Processed_ETF_Data')

        if not os.path.exists(self.cal_factor_script_path):
            raise FileNotFoundError(f"计算脚本未找到: {self.cal_factor_script_path}")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录未找到: {self.data_dir}。请先运行 A01_DataPrepare.py。")

    def _execute_and_repair_factor_calculation(self, factor_ast: dict) -> (bool, dict):
        """
        通过命令行执行因子计算，并在出错时尝试自动修复。
        """
        max_attempts = 2  # 首次尝试 + 1次修复后重试
        temp_ast_file = None

        try:
            # 1. 将AST写入临时文件
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
                json.dump(factor_ast['ast'], f)
                temp_ast_file = f.name
            logging.info(f"因子AST已保存到临时文件: {temp_ast_file}")

            # 2. 构建命令行参数字典
            command_args = {
                "--ast_file": temp_ast_file,
                "--data_dir": self.data_dir,
                "--close_key": "close",
                "--return_type": "simple"
            }

            for attempt in range(1, max_attempts + 1):
                logging.info(f"--- 开始执行因子计算脚本 (尝试 #{attempt}) ---")
                
                # 3. 执行脚本
                exec_result = CodeExecutor.run_script(self.cal_factor_script_path, args=command_args)

                # 4. 处理执行结果
                if exec_result['success']:
                    logging.info("脚本成功执行。")
                    logging.debug(f"STDOUT:\n{exec_result['stdout']}")
                    try:
                        evaluation_report = json.loads(exec_result['stdout'])
                        return True, evaluation_report
                    except json.JSONDecodeError:
                        logging.error("解析因子评估报告JSON失败。")
                        return False, {"error": "JSON解析失败", "output": exec_result['stdout']}
                else:
                    logging.error("脚本执行失败。")
                    logging.error(f"STDERR:\n{exec_result['stderr']}")

                    if attempt >= max_attempts:
                        logging.error("已达到最大修复尝试次数，放弃当前因子。")
                        return False, {"error": "修复失败", "traceback": exec_result['stderr']}

                    # 5. 自动修复流程
                    logging.info("调用Python工程师智能体尝试自动修复...")
                    fix_plan = self.engineer_agent.fix_code_error(exec_result['stderr'])

                    if "suggestions" in fix_plan and fix_plan["suggestions"]:
                        suggestion = fix_plan["suggestions"][0]
                        if suggestion["action"] == "REPLACE":
                            logging.info(f"正在应用修复：替换文件 {suggestion['file_path']} 中的代码")
                            result = FileOps.replace_snippet(
                                suggestion['file_path'],
                                suggestion['old_code_snippet'],
                                suggestion['new_code_snippet']
                            )
                            logging.info(f"文件操作结果: {result}")
                            if "Error" in result:
                                logging.error("代码片段替换失败，修复终止。")
                                return False, {"error": "代码片段替换失败", "details": result}
                            logging.info("修复已应用，将重试计算...")
                        else:
                            logging.warning(f"收到不支持的修复操作: {suggestion['action']}。放弃修复。")
                            return False, {"error": "不支持的修复操作", "plan": fix_plan}
                    else:
                        logging.error("工程师智能体未能提供有效的修复方案。")
                        return False, {"error": "无修复方案", "plan": fix_plan}
        finally:
            # 6. 清理临时文件
            if temp_ast_file and os.path.exists(temp_ast_file):
                os.remove(temp_ast_file)
                logging.info(f"临时文件已删除: {temp_ast_file}")
        return False, {"error": "未知的执行循环错误"}

    def run_pipeline(self, initial_research_topic: str = None):
        """
        启动并执行完整的因子挖掘自动化流程。
        """
        logging.info("===== 自动化因子挖掘流程启动 =====")
        if initial_research_topic:
            logging.info(f"初始研究方向: {initial_research_topic}")

        logging.info("调用金融数学家智能体，构思初始因子...")
        current_ast = self.fm_agent.propose_factor_or_operator(initial_idea=initial_research_topic)
        if "ast" not in current_ast:
            logging.error(f"无法获取有效的初始因子，流程终止。响应: {current_ast}")
            return

        for i in range(1, self.max_iterations + 1):
            logging.info(f"\n{'=' * 20} 第 {i}/{self.max_iterations} 轮迭代开始 {'=' * 20}")
            logging.info(f"正在处理因子: {current_ast.get('des', 'N/A')}")
            logging.info(f"因子 AST: {json.dumps(current_ast.get('ast', {}), indent=2)}")

            is_success, report = self._execute_and_repair_factor_calculation(current_ast)

            if not is_success:
                logging.error("因子计算或修复失败，将此失败记录在案，并构思一个全新的因子。")
                error_report = {"因子性能分析": f"计算或修复失败: {report.get('error')}",
                                "因子升级建议": "检查AST或算子逻辑，或修复相关代码。"}
                self.fm_agent.add_history_factor_result(current_ast, error_report)
                current_ast = self.fm_agent.propose_factor_or_operator()
                continue

            evaluation_report = report

            if self._check_termination_conditions(evaluation_report):
                logging.info(f"因子性能已达到预设目标，流程终止。")
                break

            logging.info("调用分析师智能体进行深度分析...")
            analysis_report = self.analyst_agent.analyze_factor_performance(
                factor_ast=current_ast['ast'],
                evaluation_report=evaluation_report
            )
            logging.info(f"分析师报告: \n{json.dumps(analysis_report, indent=2, ensure_ascii=False)}")

            self.fm_agent.add_history_factor_result(current_ast, analysis_report)
            logging.info("分析报告已添加至金融数学家的历史记录中。")

            logging.info("调用金融数学家智能体，基于分析报告构思新因子...")
            new_proposal = self.fm_agent.propose_factor_or_operator()

            if "ast" in new_proposal:
                current_ast = new_proposal
            else:
                logging.error(f"金融数学家返回了未知格式的响应，流程终止: {new_proposal}")
                break

        logging.info(f"\n{'=' * 20} 自动化因子挖掘流程结束 {'=' * 20}")
        if i == self.max_iterations:
            logging.info(f"已达到最大迭代次数 ({self.max_iterations})。")

    def _check_termination_conditions(self, report: dict) -> bool:
        """
        检查是否满足任一性能终止条件。
        """
        rank_icir = report.get("rank_ic_analysis", {}).get("rank_icir", 0)
        sharpe = report.get("long_short_portfolio_analysis", {}).get("sharpe_ratio", 0)

        if rank_icir >= self.performance_thresholds.get("rank_icir", float('inf')):
            logging.info(f"性能达标: Rank ICIR ({rank_icir:.4f}) >= 阈值 ({self.performance_thresholds['rank_icir']})")
            return True
        if sharpe >= self.performance_thresholds.get("sharpe_ratio", float('inf')):
            logging.info(
                f"性能达标: Sharpe Ratio ({sharpe:.4f}) >= 阈值 ({self.performance_thresholds['sharpe_ratio']})")
            return True
        return False


if __name__ == '__main__':
    MAX_ITERATIONS = 50
    PERFORMANCE_THRESHOLDS = {
        "rank_icir": 0.15,
        "sharpe_ratio": 1.5
    }
    INITIAL_TOPIC = None

    scheduler = Scheduler(
        max_iterations=MAX_ITERATIONS,
        performance_thresholds=PERFORMANCE_THRESHOLDS
    )
    scheduler.run_pipeline(initial_research_topic=INITIAL_TOPIC)
