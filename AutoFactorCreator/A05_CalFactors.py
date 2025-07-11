# -*- encoding: utf-8 -*-
"""
@File: A05_CalFactors.py
@Modify Time: 2025/7/11 10:03       
@Author: Kevin-Chen
@Descriptions: This module is responsible for calculating factor values based on AST
              and evaluating their effectiveness.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from AutoFactorCreator import A02_OperatorLibrary as op_lib


class FactorCalculator:
    """
    A class to calculate and evaluate factors based on Abstract Syntax Trees (AST).
    """

    def __init__(self, data_dfs: dict):
        """
        Initializes the calculator with necessary market data.

        :param data_dfs: A dictionary of DataFrames, where keys are variable names
                         (e.g., 'open', 'close', 'vol') and values are the corresponding DataFrames.
        """
        self.data = data_dfs

    def calculate_factor(self, factor_ast: dict) -> pd.DataFrame:
        """
        Public method to calculate a factor from its AST representation.
        This version uses a DAG to handle common subexpressions efficiently.

        :param factor_ast: The factor's logic represented as an AST dictionary.
        :return: A DataFrame containing the calculated factor values.
        """
        execution_plan, dag_representation = self._build_dag_and_plan(factor_ast)
        final_result = self._execute_dag_plan(execution_plan, dag_representation)
        return final_result

    def _get_node_id(self, node: dict, known_nodes: dict) -> str:
        """
        Generates a unique, deterministic ID for any AST node.
        This is the core of common subexpression elimination.
        """
        node_type = node.get('type')
        if node_type == 'literal':
            return f"literal::{node['value']}"
        elif node_type == 'variable':
            return f"variable::{node['name']}"
        elif node_type == 'operator':
            child_ids = [self._get_node_id(child, known_nodes) for child in node['children']]
            node_id = f"operator::{node['func']}({', '.join(child_ids)})"
            return node_id
        else:
            raise ValueError(f"Unsupported AST node type: {node_type}")

    def _build_dag_and_plan(self, factor_ast: dict) -> (list, dict):
        """
        Builds a Directed Acyclic Graph (DAG) from the AST and returns a
        topologically sorted execution plan.
        """
        dag = {}

        def build_recursive(node):
            node_id = self._get_node_id(node, dag)
            if node_id in dag:
                return node_id
            node_type = node.get('type')
            if node_type in ['literal', 'variable']:
                dag[node_id] = {'type': node_type, 'node': node, 'deps': []}
                return node_id
            if node_type == 'operator':
                dependencies = [build_recursive(child) for child in node['children']]
                dag[node_id] = {'type': node_type, 'node': node, 'deps': dependencies}
                return node_id
            raise ValueError(f"Unsupported AST node type: {node_type}")

        build_recursive(factor_ast)
        plan = []
        visited = set()

        def topological_sort_util(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for dep_id in dag[node_id]['deps']:
                topological_sort_util(dep_id)
            plan.append(node_id)

        for node_id in dag:
            topological_sort_util(node_id)
        return plan, dag

    def _execute_dag_plan(self, plan: list, dag: dict) -> pd.DataFrame:
        """
        Executes a topologically sorted plan with a cache for intermediate results.
        """
        cache = {}
        for node_id in plan:
            if node_id in cache:
                continue
            node_info = dag[node_id]
            node_type = node_info['type']
            if node_type == 'literal':
                result = node_info['node']['value']
            elif node_type == 'variable':
                result = self.data[node_info['node']['name']].copy()
            elif node_type == 'operator':
                args = [cache[dep_id] for dep_id in node_info['deps']]
                func = getattr(op_lib, node_info['node']['func'])
                result = func(*args)
            else:
                raise ValueError(f"Unsupported node type in execution plan: {node_type}")
            cache[node_id] = result
        return cache[plan[-1]]

    def evaluate_factor(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame):
        """
        Runs a full suite of evaluations on a calculated factor.
        """
        # Ensure factor and returns are aligned before evaluation
        aligned_factor, aligned_returns = factor_df.align(forward_returns_df, join='inner', axis=0)

        results = {
            'ic_analysis': self._calculate_ic(aligned_factor, aligned_returns),
            'rank_ic_analysis': self._calculate_rank_ic(aligned_factor, aligned_returns),
            'turnover_analysis': self._calculate_turnover(aligned_factor),
            'group_return_analysis': self._calculate_group_returns(aligned_factor, aligned_returns),
            'long_short_portfolio_analysis': self._calculate_long_short_returns(aligned_factor, aligned_returns)
        }
        return results

    def _calculate_ic(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> dict:
        """
        Calculates Information Coefficient (IC) and related metrics (ICIR, t-test).
        """
        ic_dict = {}
        for date in factor_df.index:
            try:
                factor_slice = factor_df.loc[date].dropna()
                return_slice = forward_returns_df.loc[date].dropna()
                common_index = factor_slice.index.intersection(return_slice.index)
                if len(common_index) < 2:
                    ic_dict[date] = np.nan
                    continue
                ic = factor_slice.loc[common_index].corr(return_slice.loc[common_index], method='pearson')
                ic_dict[date] = ic
            except (ValueError, IndexError, KeyError) as e:
                print(f"Error calculating IC for date {date}: {e}")
                ic_dict[date] = np.nan
        ic_series = pd.Series(ic_dict, name='ic').reindex(factor_df.index)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std != 0 and not np.isnan(ic_std) else np.nan
        t_stat, p_value = stats.ttest_1samp(ic_series.dropna(), 0, nan_policy='omit')
        return {"ic_series": ic_series, "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir, "t_statistic": t_stat,
                "p_value": p_value}

    def _calculate_rank_ic(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> dict:
        """
        Calculates Rank Information Coefficient (Rank IC) and its IR.
        """
        rank_ic_dict = {}
        for date in factor_df.index:
            try:
                factor_slice = factor_df.loc[date].dropna()
                return_slice = forward_returns_df.loc[date].dropna()
                common_index = factor_slice.index.intersection(return_slice.index)
                if len(common_index) < 2:
                    rank_ic_dict[date] = np.nan
                    continue
                rank_ic = factor_slice.loc[common_index].corr(return_slice.loc[common_index], method='spearman')
                rank_ic_dict[date] = rank_ic
            except (ValueError, IndexError, KeyError) as e:
                print(f"Error calculating Rank IC for date {date}: {e}")
                rank_ic_dict[date] = np.nan
        rank_ic_series = pd.Series(rank_ic_dict, name='rank_ic').reindex(factor_df.index)
        rank_ic_mean = rank_ic_series.mean()
        rank_ic_std = rank_ic_series.std()
        rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 and not np.isnan(rank_ic_std) else np.nan
        return {"rank_ic_series": rank_ic_series, "rank_ic_mean": rank_ic_mean, "rank_ic_std": rank_ic_std,
                "rank_icir": rank_icir}

    def _calculate_turnover(self, factor_df: pd.DataFrame) -> dict:
        """
        Calculates the turnover of the factor.
        """
        if (factor_df < 0).any().any():
            factor_positive = factor_df.rank(axis=1, pct=True)
        else:
            factor_positive = factor_df
        weights = factor_positive.div(factor_positive.sum(axis=1), axis=0).fillna(0)
        turnover_series = weights.diff().abs().sum(axis=1) / 2
        mean_turnover = turnover_series.mean()
        return {"mean_turnover": mean_turnover, "turnover_series": turnover_series}

    def _calculate_group_returns(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame,
                                 num_quantiles=5) -> dict:
        """
        Calculates returns by factor quantiles and tests for monotonicity.
        """

        def get_quantile(row):
            return pd.qcut(row, num_quantiles, labels=False, duplicates='drop')

        factor_quantiles = factor_df.rank(axis=1, pct=True).apply(get_quantile, axis=1)
        aligned_quantiles = factor_quantiles.stack().rename('quantile')
        aligned_returns = forward_returns_df.stack().rename('returns')
        aligned_data = pd.concat([aligned_quantiles, aligned_returns], axis=1).dropna()
        mean_group_returns = aligned_data.groupby('quantile')['returns'].mean().rename_axis(f'Factor_Quantile')
        monotonicity = mean_group_returns.is_monotonic_increasing or mean_group_returns.is_monotonic_decreasing
        return {"mean_group_returns": mean_group_returns, "monotonicity": monotonicity}

    def _calculate_long_short_returns(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame,
                                      num_quantiles=5) -> dict:
        """
        Calculates the returns of a long-short portfolio.
        """

        def get_quantile(row):
            return pd.qcut(row, num_quantiles, labels=False, duplicates='drop')

        factor_quantiles = factor_df.rank(axis=1, pct=True).apply(get_quantile, axis=1)
        aligned_quantiles = factor_quantiles.stack().rename('quantile')
        aligned_returns = forward_returns_df.stack().rename('returns')
        aligned_data = pd.concat([aligned_quantiles, aligned_returns], axis=1).dropna()
        top_quantile_returns = aligned_data[aligned_data['quantile'] == num_quantiles - 1].groupby(level=0)[
            'returns'].mean()
        bottom_quantile_returns = aligned_data[aligned_data['quantile'] == 0].groupby(level=0)['returns'].mean()
        long_short_returns = top_quantile_returns.subtract(bottom_quantile_returns, fill_value=0)
        long_short_returns = long_short_returns.reindex(factor_df.index).fillna(0)
        cumulative_returns = (1 + long_short_returns).cumprod()
        sharpe_ratio = long_short_returns.mean() / long_short_returns.std() * np.sqrt(
            252) if long_short_returns.std() != 0 else 0
        return {"long_short_returns": long_short_returns, "cumulative_returns": cumulative_returns,
                "sharpe_ratio": sharpe_ratio}


def prepare_data_and_calculator(data_paths: dict, close_path_key='close'):
    """
    Loads all data, prepares a FactorCalculator instance and forward returns.
    """
    print("--- Loading and Preparing Data ---")

    def load_and_prepare_df(path, name):
        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
        elif path.endswith('.csv'):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format for {path}.")

        # Standardize index to datetime
        if 'date' in df.columns: df = df.set_index('date')
        if 'Date' in df.columns: df = df.set_index('Date')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        print(f"Loaded {name}, Index Type: {type(df.index)}, Shape: {df.shape}")
        return df

    loaded_data = {name: load_and_prepare_df(path, name) for name, path in data_paths.items()}

    if close_path_key not in loaded_data:
        raise ValueError(f"Close price data (key '{close_path_key}') not found.")

    print("Calculating forward returns...")
    forward_returns = loaded_data[close_path_key].pct_change(1).shift(-1)

    print("--- Initializing Factor Calculator ---")
    calculator = FactorCalculator(data_dfs=loaded_data)

    return calculator, forward_returns


def calculate_factor_values(factor_ast: dict, calculator: FactorCalculator) -> pd.DataFrame:
    """
    Calculates factor values from an AST using the provided calculator.
    """
    print("--- Calculating Factor from AST ---")
    factor_values = calculator.calculate_factor(factor_ast)
    print("Factor calculation complete.")
    return factor_values


def evaluate_factor_performance(factor_values: pd.DataFrame, calculator: FactorCalculator,
                                forward_returns: pd.DataFrame) -> dict:
    """
    Evaluates the performance of the calculated factor values.
    """
    print("--- Evaluating Factor ---")
    evaluation_results = calculator.evaluate_factor(factor_values, forward_returns)
    print("Factor evaluation complete.")
    return evaluation_results


def convert_ast_format(node):
    """
    A utility function to convert the user-provided AST format to the internal format.
    """
    if not isinstance(node, dict):
        return {'type': 'literal', 'value': node}
    if 'var' in node:
        return {'type': 'variable', 'name': node['var']}
    if 'func' in node:
        new_node = {'type': 'operator', 'func': node['func']}
        args = node.get('args', {})
        children = []
        arg_order = ['data', 'a', 'b', 'window', 'span', 'halflife', 'p', 'q', 'axis', 'ddof']
        for key in arg_order:
            if key in args:
                children.append(convert_ast_format(args[key]))
        new_node['children'] = children
        return new_node
    return {'type': 'literal', 'value': node}


def display_evaluation_report(results: dict):
    """Formats and prints the factor evaluation report."""
    print("--- FACTOR EVALUATION REPORT ---")
    print("=" * 35)
    ic_res = results['ic_analysis']
    print(f"IC Mean: {ic_res['ic_mean']:.4f}")
    print(f"IC Std Dev: {ic_res['ic_std']:.4f}")
    print(f"ICIR: {ic_res['icir']:.4f}")
    print(f"T-statistic: {ic_res['t_statistic']:.4f}, P-value: {ic_res['p_value']:.4f}")
    print("-" * 35)
    rank_ic_res = results['rank_ic_analysis']
    print(f"Rank IC Mean: {rank_ic_res['rank_ic_mean']:.4f}")
    print(f"Rank IC Std Dev: {rank_ic_res['rank_ic_std']:.4f}")
    print(f"Rank ICIR: {rank_ic_res['rank_icir']:.4f}")
    print("-" * 35)
    turnover_res = results['turnover_analysis']
    print(f"Mean Daily Turnover: {turnover_res['mean_turnover']:.4f}")
    print("-" * 35)
    group_res = results['group_return_analysis']
    print("Mean Returns by Factor Quantile:")
    print(group_res['mean_group_returns'].to_string())
    print(f"Monotonicity: {group_res['monotonicity']}")
    print("-" * 35)
    ls_res = results['long_short_portfolio_analysis']
    print("Long-Short Portfolio Performance:")
    print(f"Annualized Sharpe Ratio: {ls_res['sharpe_ratio']:.4f}")
    print("=" * 35)
    print("Analysis finished.")


if __name__ == '__main__':
    data_paths = {
        'amount': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_amount_df.parquet',
        'close': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_close_df.parquet',
        'high': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_high_df.parquet',
        'log_return': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_log_df.parquet',
        'low': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_low_df.parquet',
        'open': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_open_df.parquet',
        'vol': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_vol_df.parquet'
    }

    user_ast = {
        'func': 'divide',
        'args': {
            'a': {'var': 'vol'},
            'b': {
                'func': 'std_dev',
                'args': {
                    'data': {
                        'func': 'moving_average',
                        'args': {
                            'data': {'var': 'vol'},
                            'window': 20,
                            'axis': 0
                        }
                    },
                    'axis': 0,
                    'ddof': 1
                }
            }
        }
    }

    # 1. Convert user AST to internal format
    print("--- Converting AST format ---")
    internal_ast = convert_ast_format(user_ast)
    print("Converted AST:", json.dumps(internal_ast, indent=2))

    # 2. Prepare data and calculator
    calculator, forward_returns = prepare_data_and_calculator(data_paths, close_path_key='close')

    # 3. Calculate factor values
    factor_values_df = calculate_factor_values(internal_ast, calculator)

    # 4. DEBUG: Inspect the calculated factor DataFrame
    print(" --- Inspecting Calculated Factor DataFrame ---")
    print("DataFrame Info:")
    factor_values_df.info()
    print("DataFrame Head: ")
    print(factor_values_df.head())
    print("DataFrame Tail: ")
    print(factor_values_df.tail())
    print("--------------------------------------------")

    # 5. Evaluate factor performance
    final_results = evaluate_factor_performance(factor_values_df, calculator, forward_returns)

    # 6. Display final report
    display_evaluation_report(final_results)
