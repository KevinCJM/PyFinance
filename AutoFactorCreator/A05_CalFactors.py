# -*- encoding: utf-8 -*-
"""
@File: A05_CalFactors.py
@Modify Time: 2025/7/11 10:03       
@Author: Kevin-Chen
@Descriptions: This module is responsible for calculating factor values based on AST
              and evaluating their effectiveness.
"""
import pandas as pd
import numpy as np
from scipy import stats
# Assume A02_OperatorLibrary contains all the necessary calculation functions
from AutoFactorCreator import A02_OperatorLibrary as op_lib


class FactorCalculator:
    """
    A class to calculate and evaluate factors based on Abstract Syntax Trees (AST).
    """

    def __init__(self, open_df, high_df, low_df, close_df, volume_df, turn_df, returns_df):
        """
        Initializes the calculator with necessary market data.

        :param open_df: DataFrame of opening prices.
        :param high_df: DataFrame of high prices.
        :param low_df: DataFrame of low prices.
        :param close_df: DataFrame of close prices.
        :param volume_df: DataFrame of volume.
        :param turn_df: DataFrame of turnover.
        :param returns_df: DataFrame of forward returns for evaluation.
        """
        self.data = {
            'open': open_df,
            'high': high_df,
            'low': low_df,
            'close': close_df,
            'volume': volume_df,
            'turn': turn_df,
        }
        # Forward returns are crucial for IC, group analysis, etc.
        self.forward_returns = returns_df
        # self.op_lib = op_lib # Operator library instance

    def calculate_factor(self, factor_ast: dict) -> pd.DataFrame:
        """
        Public method to calculate a factor from its AST representation.

        :param factor_ast: The factor's logic represented as an AST dictionary.
        :return: A DataFrame where columns are assets and rows are dates,
                 containing the calculated factor values.
        """
        # The AST is expected to be a dictionary, e.g.,
        # {'type': 'operator', 'func': 'ts_mean', 'children': [
        #   {'type': 'variable', 'name': 'close'},
        #   {'type': 'literal', 'value': 10}
        # ]}
        return self._execute_ast(factor_ast)

    def _execute_ast(self, node: dict):
        """
        Recursively executes the AST to calculate factor values.

        :param node: The current node in the AST.
        :return: The result of the calculation at this node (can be a DataFrame or a literal).
        """
        node_type = node.get('type')

        if node_type == 'operator':
            # Get the function from the operator library
            func = getattr(self.op_lib, node['func'])
            # Recursively call children to get arguments
            args = [self._execute_ast(child) for child in node['children']]
            return func(*args)

        elif node_type == 'variable':
            # Return the corresponding data DataFrame
            return self.data[node['name']].copy()

        elif node_type == 'literal':
            # Return the literal value (e.g., for a window size)
            return node['value']

        else:
            raise ValueError(f"Unsupported AST node type: {node_type}")

    def evaluate_factor(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame):
        """
        Runs a full suite of evaluations on a calculated factor.

        :param factor_df: The DataFrame with factor values.
        :param forward_returns_df: The DataFrame with forward returns.
        :return: A dictionary containing all evaluation results.
        """
        results = {
            'ic_analysis': self._calculate_ic(factor_df, forward_returns_df),
            'rank_ic_analysis': self._calculate_rank_ic(factor_df, forward_returns_df),
            'turnover_analysis': self._calculate_turnover(factor_df),
            'group_return_analysis': self._calculate_group_returns(factor_df, forward_returns_df),
            'long_short_portfolio_analysis': self._calculate_long_short_returns(factor_df, forward_returns_df)
        }
        return results

    def _calculate_ic(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> dict:
        """
        Calculates Information Coefficient (IC) and related metrics (ICIR, t-test).
        IC is the Pearson correlation between factor values and subsequent returns.
        """
        # Align factor and returns data. Drop rows with NaNs in either.
        aligned_data = pd.concat([factor_df.stack(), forward_returns_df.stack()], axis=1).dropna()
        aligned_data.columns = ['factor', 'returns']

        # Calculate IC series (Pearson correlation for each time period)
        ic_series = aligned_data.groupby(level=0).apply(lambda x: x['factor'].corr(x['returns'], method='pearson'))

        # Calculate IC metrics
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std != 0 else np.nan

        # Perform t-test on the IC series
        t_stat, p_value = stats.ttest_1samp(ic_series.dropna(), 0)

        return {
            "ic_series": ic_series,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "t_statistic": t_stat,
            "p_value": p_value
        }

    def _calculate_rank_ic(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> dict:
        """
        Calculates Rank Information Coefficient (Rank IC) and its IR.
        Rank IC is the Spearman correlation between factor values and subsequent returns.
        """
        # Align factor and returns data. Drop rows with NaNs in either.
        aligned_data = pd.concat([factor_df.stack(), forward_returns_df.stack()], axis=1).dropna()
        aligned_data.columns = ['factor', 'returns']

        # Calculate Rank IC series (Spearman correlation for each time period)
        rank_ic_series = aligned_data.groupby(level=0).apply(
            lambda x: x['factor'].corr(x['returns'], method='spearman'))

        # Calculate Rank IC metrics
        rank_ic_mean = rank_ic_series.mean()
        rank_ic_std = rank_ic_series.std()
        rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else np.nan

        return {
            "rank_ic_series": rank_ic_series,
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_std": rank_ic_std,
            "rank_icir": rank_icir
        }

    def _calculate_turnover(self, factor_df: pd.DataFrame) -> dict:
        """
        Calculates the turnover of the factor.
        """
        # TODO: Implementation
        print("Calculating Factor Turnover...")
        return {"mean_turnover": 0.0}

    def _calculate_group_returns(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame,
                                 num_quantiles=5) -> dict:
        """
        Calculates returns by factor quantiles and tests for monotonicity.

        :param factor_df: The DataFrame with factor values.
        :param forward_returns_df: The DataFrame with forward returns.
        :param num_quantiles: The number of groups to divide the assets into.
        :return: A dictionary with mean returns for each quantile and a monotonicity metric.
        """

        # Function to get quantile for each row
        def get_quantile(row):
            return pd.qcut(row, num_quantiles, labels=False, duplicates='drop')

        # Calculate quantiles for each day
        factor_quantiles = factor_df.rank(axis=1, pct=True).apply(get_quantile, axis=1)

        # Align data
        aligned_quantiles = factor_quantiles.stack().rename('quantile')
        aligned_returns = forward_returns_df.stack().rename('returns')
        aligned_data = pd.concat([aligned_quantiles, aligned_returns], axis=1).dropna()

        # Calculate mean return for each quantile group over time
        mean_group_returns = aligned_data.groupby('quantile')['returns'].mean().rename_axis(f'Factor_Quantile')

        # Test for monotonicity
        # A simple check is to see if the difference between consecutive group returns is always positive or always negative.
        monotonicity = mean_group_returns.is_monotonic_increasing or mean_group_returns.is_monotonic_decreasing

        return {
            "mean_group_returns": mean_group_returns,
            "monotonicity": monotonicity
        }

    def _calculate_long_short_returns(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame,
                                      num_quantiles=5) -> dict:
        """
        Calculates the returns of a long-short portfolio based on top and bottom factor quantiles.

        :param factor_df: The DataFrame with factor values.
        :param forward_returns_df: The DataFrame with forward returns.
        :param num_quantiles: The number of groups to divide the assets into.
        :return: A dictionary with cumulative returns and Sharpe ratio of the long-short strategy.
        """

        # Function to get quantile for each row
        def get_quantile(row):
            return pd.qcut(row, num_quantiles, labels=False, duplicates='drop')

        # Calculate quantiles for each day
        factor_quantiles = factor_df.rank(axis=1, pct=True).apply(get_quantile, axis=1)

        # Align data
        aligned_quantiles = factor_quantiles.stack().rename('quantile')
        aligned_returns = forward_returns_df.stack().rename('returns')
        aligned_data = pd.concat([aligned_quantiles, aligned_returns], axis=1).dropna()

        # Calculate daily returns for top and bottom quantiles
        top_quantile_returns = aligned_data[aligned_data['quantile'] == num_quantiles - 1].groupby(level=0)[
            'returns'].mean()
        bottom_quantile_returns = aligned_data[aligned_data['quantile'] == 0].groupby(level=0)['returns'].mean()

        # Calculate long-short strategy returns
        long_short_returns = top_quantile_returns - bottom_quantile_returns

        # Fill missing daily returns with 0
        long_short_returns = long_short_returns.reindex(factor_df.index).fillna(0)

        # Calculate cumulative returns
        cumulative_returns = (1 + long_short_returns).cumprod()

        # Calculate Sharpe Ratio (assuming risk-free rate is 0)
        sharpe_ratio = long_short_returns.mean() / long_short_returns.std() * np.sqrt(252)  # Annualized

        return {
            "long_short_returns": long_short_returns,
            "cumulative_returns": cumulative_returns,
            "sharpe_ratio": sharpe_ratio
        }
