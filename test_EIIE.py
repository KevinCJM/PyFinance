import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import gym
from gym import spaces
import quantstats as qs
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import copy
import warnings
import math

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# 模块一：EIIE 模型架构与辅助类 (来自您提供的 EIIE_model.py)
# ==============================================================================

class EIIE(nn.Module):
    """EIIE（Ensemble of Identical Independent Evaluators）策略网络"""

    def __init__(self, initial_features=3, k_size=3, conv_1_features=2, conv_2_features=20, time_window=50,
                 device="cpu"):
        super().__init__()
        self.device = device
        n_size = time_window - k_size + 1
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=initial_features, out_channels=conv_1_features, kernel_size=(1, k_size)),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_1_features, out_channels=conv_2_features, kernel_size=(1, n_size)),
            nn.ReLU(),
        )
        self.final_convolution = nn.Conv2d(in_channels=conv_2_features + 1, out_channels=1, kernel_size=(1, 1))
        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()
        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)
        output = self.sequential(observation)
        output = torch.cat([last_stocks, output], dim=1)
        output = self.final_convolution(output)
        output = torch.cat([cash_bias, output], dim=2)
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)
        output = self.softmax(output)
        return output

    def forward(self, observation, last_action):
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias


class PVM:
    """组合向量记忆（Portfolio Vector Memory）"""

    def __init__(self, capacity, portfolio_size):
        self.capacity = capacity
        self.portfolio_size = portfolio_size
        self.reset()

    def reset(self):
        self.memory = [np.array([1] + [0] * self.portfolio_size, dtype=np.float32)] * (self.capacity + 1)
        self.index = 0

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action


class ReplayBuffer:
    """回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self):
        buffer = list(self.buffer)
        self.buffer.clear()
        return buffer


class RLDataSet(IterableDataset):
    """强化学习数据集"""

    def __init__(self, buffer):
        self.buffer = buffer

    def __iter__(self):
        yield from self.buffer.sample()


# ==============================================================================
# 模块二：强化学习环境 (来自您提供的 DRL_env.py)
# ==============================================================================
class PortfolioOptimizationEnv(gym.Env):
    """一个用于 OpenAI gym 的投资组合优化环境"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, initial_amount, commission_fee_pct=0, time_window=1, features=None, **kwargs):
        self._time_window = time_window
        self._time_index = time_window - 1
        self._df = df
        self._initial_amount = initial_amount
        self._commission_fee_pct = commission_fee_pct
        self._features = ["close", "high", "low"] if features is None else features
        self._valuation_feature = "close"
        self._time_column = "trade_date"
        self._tic_column = "ts_code"
        self._cwd = kwargs.get("cwd", "./")

        self._preprocess_data()

        self._tic_list = self._df[self._tic_column].unique()
        self.portfolio_size = len(self._tic_list)
        self._sorted_times = sorted(set(self._df[self._time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        action_space_dim = 1 + self.portfolio_size
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space_dim,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self._features), self.portfolio_size, self._time_window))

        self._reset_memory()
        self._portfolio_value = self._initial_amount
        self._terminal = False

    def step(self, actions):
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            self._generate_final_report()
            return self._state, self._reward, self._terminal, {}
        else:
            weights = self._softmax_normalization(actions)
            self._actions_memory.append(weights)
            last_weights = self._final_weights[-1]

            # 交易成本 (TRF模型)
            mu = 1 - 2 * self._commission_fee_pct + self._commission_fee_pct ** 2
            last_mu = 1
            while abs(mu - last_mu) > 1e-10:
                last_mu = mu
                mu = (1 - self._commission_fee_pct * weights[0] - (
                        2 * self._commission_fee_pct - self._commission_fee_pct ** 2) * np.sum(
                    np.maximum(last_weights[1:] - mu * weights[1:], 0))) / (1 - self._commission_fee_pct * weights[0])

            self._portfolio_value *= mu
            self._asset_memory["initial"].append(self._portfolio_value)

            self._time_index += 1
            self._state, self._info = self._get_state_and_info_from_time_index(self._time_index)

            portfolio = self._portfolio_value * (weights * self._price_variation)
            self._portfolio_value = np.sum(portfolio)
            weights = portfolio / self._portfolio_value

            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(weights)
            self._date_memory.append(self._info["end_time"])

            rate_of_return = self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            self._portfolio_return_memory.append(rate_of_return - 1)
            self._reward = np.log(rate_of_return)
            self._portfolio_reward_memory.append(self._reward)

        return self._state, self._reward, self._terminal, self._info

    def reset(self):
        self._time_index = self._time_window - 1
        self._reset_memory()
        self._state, self._info = self._get_state_and_info_from_time_index(self._time_index)
        self._portfolio_value = self._initial_amount
        self._terminal = False
        return self._state

    def _get_state_and_info_from_time_index(self, time_index):
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]
        self._data = self._df[(self._df[self._time_column] >= start_time) & (self._df[self._time_column] <= end_time)]
        self._price_variation = self._df_price_variation[self._df_price_variation[self._time_column] == end_time][
            self._valuation_feature].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)

        state = np.array([tic_data[self._features].to_numpy().T for tic_data in
                          [self._data[self._data[self._tic_column] == tic] for tic in self._tic_list]])
        state = state.transpose((1, 0, 2))

        info = {"end_time": end_time, "trf_mu": 1.0, "price_variation": self._price_variation}
        return state, info

    def _preprocess_data(self):
        self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        self._df_price_variation = self._temporal_variation_df()
        self._df = self._normalize_dataframe()
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df[self._features] = self._df[self._features].astype(np.float32)

    def _temporal_variation_df(self, periods=1):
        df_temporal_variation = self._df.copy()
        for column in self._features:
            df_temporal_variation[f"prev_{column}"] = df_temporal_variation.groupby(self._tic_column)[column].shift(
                periods=periods)
            df_temporal_variation[column] = df_temporal_variation[column] / df_temporal_variation[f"prev_{column}"]
        return df_temporal_variation.drop(columns=[f"prev_{c}" for c in self._features]).fillna(1).reset_index(
            drop=True)

    def _normalize_dataframe(self):
        # EIIE 默认使用 v_t / v_{t-n+1} 的归一化方式
        return self._temporal_variation_df(self._time_window - 1)

    def _reset_memory(self):
        date_time = self._sorted_times[self._time_index]
        self._asset_memory = {"initial": [self._initial_amount], "final": [self._initial_amount]}
        self._portfolio_return_memory, self._portfolio_reward_memory = [0], [0]
        initial_weights = np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        self._actions_memory, self._final_weights = [initial_weights], [initial_weights]
        self._date_memory = [date_time]

    def _softmax_normalization(self, actions):
        return np.exp(actions) / np.sum(np.exp(actions))

    def _generate_final_report(self):
        print("=" * 60)
        print("回测完成，生成报告...")
        metrics_df = pd.DataFrame({
            "date": self._date_memory, "returns": self._portfolio_return_memory,
            "rewards": self._portfolio_reward_memory, "portfolio_values": self._asset_memory["final"]})
        metrics_df.set_index("date", inplace=True)

        qs.plots.snapshot(metrics_df["returns"], title='EIIE Strategy Performance', show=False,
                          savefig="EIIE_performance.png")
        print(f"最终资产净值: {self._portfolio_value:.2f}")
        print(f"夏普比率: {qs.stats.sharpe(metrics_df['returns']):.4f}")
        print(f"最大回撤: {qs.stats.max_drawdown(metrics_df['portfolio_values']):.4f}")
        print("=" * 60)


# ==============================================================================
# 模块三：强化学习智能体 (来自您提供的 DRL_agent.py)
# ==============================================================================
class PolicyGradient:
    """实现策略梯度算法以训练投资组合优化智能体"""

    def __init__(self, env, policy=EIIE, policy_kwargs=None, batch_size=100, lr=1e-3, action_noise=0.0, optimizer=AdamW,
                 device="cpu"):
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.batch_size, self.lr, self.action_noise, self.optimizer, self.device = batch_size, lr, action_noise, optimizer, device
        self._setup_train(env, policy)

    def _setup_train(self, env, policy):
        self.train_env = env
        self.train_policy = policy(**self.policy_kwargs).to(self.device)
        self.train_optimizer = self.optimizer(self.train_policy.parameters(), lr=self.lr)
        self.train_buffer = ReplayBuffer(capacity=self.batch_size)
        self.train_pvm = PVM(self.train_env.episode_length, env.portfolio_size)
        dataset = RLDataSet(self.train_buffer)
        self.train_dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def train(self, episodes=100):
        for _ in tqdm(range(1, episodes + 1), desc="Training Episodes"):
            obs, done = self.train_env.reset(), False
            self.train_pvm.reset()
            while not done:
                last_action = self.train_pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                action = self.train_policy(obs_batch, last_action_batch)
                self.train_pvm.add(action)
                next_obs, reward, done, info = self.train_env.step(action)
                exp = (obs, last_action, info["price_variation"], info["trf_mu"])
                self.train_buffer.append(exp)
                if len(self.train_buffer) == self.batch_size:
                    self._gradient_ascent()
                obs = next_obs
            self._gradient_ascent()

        return self.train_policy  # 返回训练好的模型

    def _gradient_ascent(self):
        if not self.train_buffer: return
        obs, last_actions, price_variations, trf_mu = next(iter(self.train_dataloader))
        obs, last_actions, price_variations, trf_mu = obs.to(self.device), last_actions.to(
            self.device), price_variations.to(self.device), trf_mu.unsqueeze(1).to(self.device)
        mu = self.train_policy.mu(obs, last_actions)
        policy_loss = -torch.mean(torch.log(torch.sum(mu * price_variations * trf_mu, dim=1)))
        self.train_policy.zero_grad();
        policy_loss.backward();
        self.train_optimizer.step()


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 参数配置 ---
    CONFIG = {
        'file_path': './Data/etf_daily.parquet',
        'fund_list': ['510050.SH', '159915.SZ', '159912.SZ', '512500.SH', '511010.SH',
                      '513100.SH', '513030.SH', '513080.SH', '513520.SH', '518880.SH',
                      '161226.SZ', '501018.SH', '159981.SZ', '159985.SZ', '159980.SZ', '511990.SH'],
        'features': ["close", "high", "low"],
        'train_start': '2020-01-01', 'train_end': '2023-12-31',
        'test_start': '2024-01-01', 'test_end': '2024-06-25',
        'time_window': 50, 'initial_amount': 100000.0, 'commission_fee_pct': 0.001,
        'learning_rate': 1e-4, 'episodes': 100, 'batch_size': 128, 'device': "cpu"
    }

    # --- 2. 数据准备 ---
    print("--- 开始数据准备 ---")
    try:
        df_full = pd.read_parquet(CONFIG['file_path'])
        df_full['trade_date'] = pd.to_datetime(df_full['trade_date'])
        df_filtered = df_full[df_full['ts_code'].isin(CONFIG['fund_list'])]

        # 统一所有资产的起止日期
        min_date = df_filtered.groupby('ts_code')['trade_date'].min().max()
        max_date = df_filtered.groupby('ts_code')['trade_date'].max().min()
        df_aligned = df_filtered[(df_filtered['trade_date'] >= min_date) & (df_filtered['trade_date'] <= max_date)]

        # 拆分训练集和测试集
        train_start_dt, train_end_dt = pd.to_datetime(CONFIG['train_start']), pd.to_datetime(CONFIG['train_end'])
        test_start_dt, test_end_dt = pd.to_datetime(CONFIG['test_start']), pd.to_datetime(CONFIG['test_end'])
        df_train = df_aligned[(df_aligned['trade_date'] >= train_start_dt) & (df_aligned['trade_date'] <= train_end_dt)]
        df_test = df_aligned[(df_aligned['trade_date'] >= test_start_dt) & (df_aligned['trade_date'] <= test_end_dt)]
        print(f"训练集覆盖: {df_train['trade_date'].min().date()} 到 {df_train['trade_date'].max().date()}")
        print(f"测试集覆盖: {df_test['trade_date'].min().date()} 到 {df_test['trade_date'].max().date()}")
    except Exception as e:
        print(f"数据加载或处理失败: {e}")
        raise

    # --- 3. 训练阶段 ---
    print("\n--- 开始训练阶段 ---")
    env_train = PortfolioOptimizationEnv(
        df_train,
        initial_amount=CONFIG['initial_amount'],
        commission_fee_pct=CONFIG['commission_fee_pct'],
        time_window=CONFIG['time_window'],
        features=CONFIG['features']
    )

    agent = PolicyGradient(
        env_train,
        policy=EIIE,
        policy_kwargs={'initial_features': len(CONFIG['features']), 'time_window': CONFIG['time_window']},
        lr=CONFIG['learning_rate'],
        batch_size=CONFIG['batch_size'],
        device=CONFIG['device']
    )

    trained_policy = agent.train(episodes=CONFIG['episodes'])

    # --- 4. 测试阶段 ---
    print("\n--- 开始测试阶段 ---")
    env_test = PortfolioOptimizationEnv(
        df_test,
        initial_amount=CONFIG['initial_amount'],
        commission_fee_pct=CONFIG['commission_fee_pct'],
        time_window=CONFIG['time_window'],
        features=CONFIG['features']
    )

    # 在测试环境中，我们可以创建一个新的agent或使用旧的agent来调用test方法
    # 这里我们复用之前的agent对象，但test方法会创建独立的test_policy
    test_agent = agent

    # 创建测试环境的 PVM 和 buffer
    test_pvm = PVM(env_test.episode_length, env_test.portfolio_size)
    test_buffer = ReplayBuffer(capacity=CONFIG['batch_size'])

    obs, done = env_test.reset(), False
    while not done:
        last_action = test_pvm.retrieve()
        obs_batch = np.expand_dims(obs, axis=0)
        last_action_batch = np.expand_dims(last_action, axis=0)
        # 使用训练好的策略进行决策
        action = trained_policy(obs_batch, last_action_batch)
        test_pvm.add(action)
        obs, reward, done, info = env_test.step(action)

    print("\n--- EIIE 模型执行完毕 ---")
