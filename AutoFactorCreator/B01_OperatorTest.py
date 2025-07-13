# -*- encoding: utf-8 -*-
"""
@File: B01_OperatorTest.py
@Modify Time: 2025/7/13 17:00
@Author: Kevin-Chen
@Descriptions: 基础算子自动测试脚本。
              使用pandas作为基准，测试A02_OperatorLibrary.py中算子的数值准确性。
"""
import unittest
import numpy as np
import pandas as pd
import AutoFactorCreator.A02_OperatorLibrary as op


class TestOperatorLibrary(unittest.TestCase):
    """
    测试 A02_OperatorLibrary 中的所有算子。
    通过将numpy实现的结果与pandas的实现进行比较来验证准确性。
    """

    def setUp(self):
        """在每个测试方法运行前设置测试环境。"""
        # 设置随机种子以保证测试的可复现性
        np.random.seed(42)
        # 定义测试数据的维度
        self.rows, self.cols = 100, 10
        # 创建包含NaN的随机测试数据
        self.data1 = self._create_random_data()
        self.data2 = self._create_random_data()
        # 创建对应的Pandas DataFrame
        self.df1 = pd.DataFrame(self.data1)
        self.df2 = pd.DataFrame(self.data2)
        # 定义滚动窗口大小
        self.window = 10

    def _create_random_data(self) -> np.ndarray:
        """生成包含NaN的随机numpy数组。"""
        data = np.random.rand(self.rows, self.cols) * 10
        # 随机插入约10%的NaN值
        nan_mask = np.random.rand(self.rows, self.cols) < 0.1
        data[nan_mask] = np.nan
        return data

    def _assert_results_almost_equal(self, custom_result, pandas_result):
        """
        断言两个结果（自定义实现和pandas实现）是否几乎相等。
        处理pandas结果到numpy数组的转换，并比较浮点数。
        """
        if isinstance(pandas_result, pd.Series):
            pandas_result = pandas_result.to_numpy()
        if isinstance(pandas_result, pd.DataFrame):
            pandas_result = pandas_result.to_numpy()

        # 使用numpy的testing工具进行浮点数数组的近似比较，并认为NaN是相等的
        np.testing.assert_allclose(custom_result, pandas_result, rtol=1e-5, atol=1e-8, equal_nan=True)

    # --- 测试基础数学算子 ---
    def test_basic_math_operators(self):
        self.df1_no_neg = self.df1.copy()
        self.df1_no_neg[self.df1_no_neg <= 0] = np.nan
        data1_no_neg = self.df1_no_neg.to_numpy()

        self._assert_results_almost_equal(op.add(self.data1, self.data2), self.df1 + self.df2)
        self._assert_results_almost_equal(op.subtract(self.data1, self.data2), self.df1 - self.df2)
        self._assert_results_almost_equal(op.multiply(self.data1, self.data2), self.df1 * self.df2)
        self._assert_results_almost_equal(op.divide(self.data1, self.data2), self.df1 / self.df2)
        self._assert_results_almost_equal(op.log(data1_no_neg), np.log(self.df1_no_neg))
        self._assert_results_almost_equal(op.abs_val(self.data1), self.df1.abs())
        self._assert_results_almost_equal(op.power(self.data1, 2), self.df1 ** 2)
        self._assert_results_almost_equal(op.sqrt(data1_no_neg), np.sqrt(self.df1_no_neg))

    # --- 测试累积算子 ---
    def test_cumulative_operators(self):
        for axis in [0, 1]:
            self._assert_results_almost_equal(op.cumulative_sum(self.data1, axis=axis),
                                              self.df1.cumsum(axis=axis, skipna=True))
            self._assert_results_almost_equal(op.cumulative_product(self.data1, axis=axis),
                                              self.df1.cumprod(axis=axis, skipna=True))
            self._assert_results_almost_equal(op.cumulative_max(self.data1, axis=axis),
                                              self.df1.cummax(axis=axis, skipna=True))
            self._assert_results_almost_equal(op.cumulative_min(self.data1, axis=axis),
                                              self.df1.cummin(axis=axis, skipna=True))

    # --- 测试滚动算子 ---
    def test_rolling_operators(self):
        for axis in [0, 1]:
            self._assert_results_almost_equal(op.rolling_sum(self.data1, self.window, axis=axis),
                                              self.df1.rolling(self.window, axis=axis).sum())
            # nanprod在pandas中没有直接的skipna=True选项，但默认行为与numpy的nanprod一致
            self._assert_results_almost_equal(op.rolling_product(self.data1, self.window, axis=axis),
                                              self.df1.rolling(self.window, axis=axis).apply(np.nanprod))
            self._assert_results_almost_equal(op.rolling_max(self.data1, self.window, axis=axis),
                                              self.df1.rolling(self.window, axis=axis).max())
            self._assert_results_almost_equal(op.rolling_min(self.data1, self.window, axis=axis),
                                              self.df1.rolling(self.window, axis=axis).min())

    # --- 测试统计算子 ---
    def test_statistical_operators(self):
        for axis in [0, 1]:
            self._assert_results_almost_equal(op.mean(self.data1, axis=axis), self.df1.mean(axis=axis))
            self._assert_results_almost_equal(op.std_dev(self.data1, axis=axis), self.df1.std(axis=axis))
            self._assert_results_almost_equal(op.variance(self.data1, axis=axis), self.df1.var(axis=axis))
            self._assert_results_almost_equal(op.max_val(self.data1, axis=axis), self.df1.max(axis=axis))
            self._assert_results_almost_equal(op.min_val(self.data1, axis=axis), self.df1.min(axis=axis))
            self._assert_results_almost_equal(op.median(self.data1, axis=axis), self.df1.median(axis=axis))
            self._assert_results_almost_equal(op.quantile(self.data1, 0.75, axis=axis),
                                              self.df1.quantile(0.75, axis=axis))

        # 测试 correlation 和 covariance
        self._assert_results_almost_equal(op.correlation(self.data1, self.data2, axis=1),
                                          self.df1.corrwith(self.df2, axis=1))
        # Pandas的covwith不存在，我们逐行/列计算来验证
        pd_cov_axis1 = [self.df1.iloc[i].cov(self.df2.iloc[i]) for i in range(self.rows)]
        self._assert_results_almost_equal(op.covariance(self.data1, self.data2, axis=1), np.array(pd_cov_axis1))

        pd_corr_axis0 = [self.df1[col].corr(self.df2[col]) for col in self.df1.columns]
        self._assert_results_almost_equal(op.correlation(self.data1, self.data2, axis=0), np.array(pd_corr_axis0))

        pd_cov_axis0 = [self.df1[col].cov(self.df2[col]) for col in self.df1.columns]
        self._assert_results_almost_equal(op.covariance(self.data1, self.data2, axis=0), np.array(pd_cov_axis0))

    # --- 测试时间序列算子 ---
    def test_time_series_operators(self):
        for axis in [0, 1]:
            self._assert_results_almost_equal(op.ts_delay(self.data1, self.window, axis=axis),
                                              self.df1.shift(self.window, axis=axis))
            self._assert_results_almost_equal(op.ts_delta(self.data1, self.window, axis=axis),
                                              self.df1.diff(self.window, axis=axis))
            self._assert_results_almost_equal(op.moving_average(self.data1, self.window, axis=axis),
                                              self.df1.rolling(self.window, axis=axis).mean())
            self._assert_results_almost_equal(op.ts_std(self.data1, self.window, axis=axis),
                                              self.df1.rolling(self.window, axis=axis).std())
            self._assert_results_almost_equal(op.ts_rank(self.data1, self.window, axis=axis),
                                              self.df1.rolling(self.window, axis=axis).rank(pct=True))

        # 测试只支持axis=0的算子
        self._assert_results_almost_equal(op.rolling_corr(self.data1, self.data2, self.window),
                                          self.df1.rolling(self.window).corr(self.df2))
        self._assert_results_almost_equal(op.rolling_cov(self.data1, self.data2, self.window),
                                          self.df1.rolling(self.window).cov(self.df2))
        self._assert_results_almost_equal(op.exponential_moving_average(self.data1, self.window),
                                          self.df1.ewm(span=self.window, adjust=False).mean())


if __name__ == '__main__':
    print("开始运行算子库自动化测试...")
    unittest.main(verbosity=2)
