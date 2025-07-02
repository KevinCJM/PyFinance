import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import random
import collections
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings

# 引入 Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings('ignore')

# 禁用 torch.compile 相关的错误
import torch._dynamo

torch._dynamo.config.suppress_errors = True

# --- 1. 配置中心 (CONFIG) - 专为PPO优化 ---
the_fund_code = '510050.SH'  # 上证50ETF
the_basic_info_file_path_dict = {
    "log": "./Data/wide_log_return_df.parquet",  # 对数收益率
    "high": "./Data/wide_high_df.parquet",  # 最高价
    "low": "./Data/wide_low_df.parquet",  # 最低价
    "vol": "./Data/wide_vol_df.parquet",  # 成交量
    "amount": "./Data/wide_amount_df.parquet",  # 成交额
    "close": "./Data/wide_close_df.parquet",  # 收盘价
    "open": "./Data/wide_open_df.parquet",  # 开盘价
}
the_metrics_info_file_path_dict = [
    "./Data/Metrics/2d.parquet",  # 2天区间指标数据
    "./Data/Metrics/5d.parquet",
    "./Data/Metrics/10d.parquet",
    "./Data/Metrics/15d.parquet",
    "./Data/Metrics/25d.parquet",
    "./Data/Metrics/50d.parquet",
    "./Data/Metrics/75d.parquet",
    "./Data/Metrics/5m.parquet",
    "./Data/Metrics/6m.parquet",
    "./Data/Metrics/rolling_metrics.parquet",  # 滚动技术指标数据
]
CONFIG = {
    "infra": {
        "seed": 42,
        "device": "auto",  # Stable Baselines3 会自动处理设备选择
    },
    "data": {
        "fund_code": '510050.SH',
        # 使用模拟数据，真实路径请自行替换
        "use_dummy_data": True,
        "dummy_days": 4945,  # 与你原始数据长度保持一致
    },
    "preprocessing": {
        "train_split_ratio": 0.8,
        "volatility_window": 21,  # 计算波动率的滚动窗口
    },
    "environment": {
        "initial_capital": 100000,
        "transaction_cost_pct": 0.001,  # 手续费
        "trade_penalty": 0.1,  # 固定的交易行为惩罚
        "hard_stop_loss_pct": -0.15,  # 强制止损阈值
    },
    # PPO 算法的核心超参数
    "ppo_params": {
        "n_steps": 2048,  # 每次更新前，每个环境要运行的步数
        "batch_size": 64,  # PPO更新时的minibatch大小
        "n_epochs": 10,  # 每次收集到数据后，对数据进行优化的轮数
        "gamma": 0.99,  # 折扣因子
        "gae_lambda": 0.95,  # GAE(广义优势估计)的lambda参数
        "clip_range": 0.2,  # PPO的核心：裁剪范围
        "ent_coef": 0.0,  # 熵系数，鼓励探索
        "learning_rate": 0.0003,  # 学习率
        "verbose": 0,  # 日志级别 (0=不打印, 1=打印)
    },
    # 训练总步数
    "training": {
        "total_timesteps": 300000,  # 总共训练的环境交互步数
    },
    # 均线策略基准的配置
    "ma_strategy": {
        "short_window": 42,
        "long_window": 252,
    },
}


def set_seeds(seed):
    """设置所有随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# --- 2. 数据加载与预处理 ---
def get_data(fund_code, basic_info_file_path_dict, metrics_info_file_path_dict):
    print("Loading basic market data...")
    the_final_df = None
    for file_type, file_path in basic_info_file_path_dict.items():
        try:
            df = pd.read_parquet(file_path)
            fund_data = df[[fund_code]].copy()
            fund_data.columns = [file_type]

            if the_final_df is None:
                the_final_df = fund_data
            else:
                the_final_df = the_final_df.join(fund_data, how='outer')
            print(f"Loaded {file_type} data: {fund_data.shape}")
        except Exception as e:
            print(f"Warning: Failed to load {file_type} from {file_path}: {e}")

    print("Loading metrics data...")
    for i, file_path in enumerate(metrics_info_file_path_dict):
        try:
            df = pd.read_parquet(file_path)
            df = df[df['ts_code'] == fund_code].reset_index(drop=True)
            df = df.drop(columns=['ts_code'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index(['date']).sort_index()
            the_final_df = the_final_df.join(df, how='outer')
            print(f"Loaded metrics file {i + 1}/{len(metrics_info_file_path_dict)}: {df.shape}")
        except Exception as e:
            print(f"Warning: Failed to load metrics from {file_path}: {e}")

    print(f"Final combined data shape: {the_final_df.shape}")
    return the_final_df


def preprocess_and_split_data(df, config):
    """
    预处理函数，增加滚动波动率作为新特征。
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    first_valid_indices = [df[col].first_valid_index() for col in df.columns]
    start_date = max(dt for dt in first_valid_indices if dt is not None)
    df = df.loc[start_date:].copy()

    # 预计算用于奖励函数的滚动波动率
    vol_window = config['preprocessing']['volatility_window']
    df['reward_volatility'] = df['log'].rolling(window=vol_window, min_periods=vol_window).std()

    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # 分割数据
    split_index = int(len(df) * config['preprocessing']['train_split_ratio'])
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # 归一化
    feature_cols = df.columns.drop(['close', 'high', 'low', 'reward_volatility'])

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = pd.DataFrame(scaler.transform(train_df[feature_cols]), index=train_df.index, columns=feature_cols)
    train_scaled[['close', 'high', 'low', 'reward_volatility']] = train_df[
        ['close', 'high', 'low', 'reward_volatility']]

    test_scaled = pd.DataFrame(scaler.transform(test_df[feature_cols]), index=test_df.index, columns=feature_cols)
    test_scaled[['close', 'high', 'low', 'reward_volatility']] = test_df[['close', 'high', 'low', 'reward_volatility']]

    train_scaled.dropna(inplace=True)
    test_scaled.dropna(inplace=True)

    return train_scaled, test_scaled


# --- 3. Gym环境 (最终版) ---
class StockTradingEnv(gym.Env):
    def __init__(self, df, config):
        super(StockTradingEnv, self).__init__()
        # 分离特征和用于计算的原始数据
        self.df = df
        self.feature_df = df.drop(columns=['close', 'high', 'low', 'reward_volatility'], errors='ignore')
        self.price_series = df['close']
        self.high_series = df['high']
        self.low_series = df['low']
        self.volatility_series = df['reward_volatility']

        self.config = config
        self.env_config = config['environment']
        self.hard_stop_loss_pct = self.env_config['hard_stop_loss_pct']
        self.initial_capital = self.env_config['initial_capital']
        self.transaction_cost_pct = self.env_config['transaction_cost_pct']
        self.trade_penalty = self.env_config['trade_penalty']

        self.action_space = spaces.Discrete(config['ppo_params'].get('action_dim', 3))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_df.columns) + 2,), dtype=np.float32
        )
        self._precompute_features()
        self.reset()

    def _precompute_features(self):
        self.features_array = self.feature_df.values.astype(np.float32)
        self.prices_array = self.price_series.values.astype(np.float32)
        self.log_returns_array = np.log(self.price_series / self.price_series.shift(1)).values.astype(np.float32)
        self.volatility_array = self.volatility_series.values.astype(np.float32)
        self.high_array = self.high_series.values.astype(np.float32)
        self.low_array = self.low_series.values.astype(np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.history = []
        self.entry_price = 0
        return self._get_observation()

    def _get_observation(self):
        market_obs = self.features_array[self.current_step]
        position_status = 1.0 if self.shares > 0 else 0.0
        if self.shares > 0:
            current_price = self.prices_array[self.current_step]
            unrealized_pnl_pct = (current_price / self.entry_price) - 1.0 if self.entry_price > 0 else 0.0
        else:
            unrealized_pnl_pct = 0.0
        return np.concatenate([market_obs, [position_status, unrealized_pnl_pct]]).astype(np.float32)

    def step(self, action):
        prev_portfolio_value = self.portfolio_value

        unrealized_pnl = 0
        if self.shares > 0 and self.entry_price > 0:
            unrealized_pnl = (self.prices_array[self.current_step] / self.entry_price) - 1
        if self.shares > 0 and unrealized_pnl < self.hard_stop_loss_pct:
            action = 2

        self._take_action(action)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        current_price = self.prices_array[self.current_step if not done else -1]
        self.portfolio_value = self.cash + self.shares * current_price

        reward = 0.0
        if prev_portfolio_value > 0:
            portfolio_log_return = np.log(self.portfolio_value / prev_portfolio_value)
            benchmark_log_return = self.log_returns_array[self.current_step if not done else -1]
            market_volatility = self.volatility_array[self.current_step if not done else -1]

            # 奖励的核心：风险调整后的Alpha
            if not np.isnan(benchmark_log_return):
                excess_return = portfolio_log_return - benchmark_log_return
                if market_volatility > 1e-9:
                    reward = excess_return / market_volatility
                else:
                    reward = excess_return

        if action == 1 or action == 2:
            reward -= self.trade_penalty

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            'date': self.df.index[self.current_step - 1],
            'portfolio_value': self.portfolio_value,
            'action': action,
            'price': self.price_series.iloc[self.current_step - 1],
            'reward': reward
        }
        self.history.append(info)
        return obs, reward, done, info

    def _take_action(self, action):
        current_step_index = self.current_step
        if action == 1 and self.cash > 0:
            buy_price = self.high_array[current_step_index]
            cost = self.cash * self.transaction_cost_pct
            buy_capital = self.cash - cost
            if buy_price > 0:
                self.shares = buy_capital / buy_price
            self.cash = 0
            self.entry_price = buy_price
        elif action == 2 and self.shares > 0:
            sell_price = self.low_array[current_step_index]
            proceeds = self.shares * sell_price
            cost = proceeds * self.transaction_cost_pct
            self.cash = proceeds - cost
            self.shares = 0
            self.entry_price = 0


# --- 4. 均线策略 (保持不变) ---
def run_moving_average_strategy(df, config):
    # ... (此处省略，使用上一版本中的代码)
    pass  # 占位符


# --- 5. PPO训练与评估函数 ---
def train_ppo_agent(config, train_df):
    print("Initializing PPO training...")
    env = DummyVecEnv([lambda: StockTradingEnv(train_df, config)])

    policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
    model = PPO(
        "MlpPolicy", env, policy_kwargs=policy_kwargs,
        device=config['infra']['device'], **config['ppo_params']
    )
    model.learn(total_timesteps=config['training']['total_timesteps'], progress_bar=True)
    print("\n--- PPO Training Finished ---")
    return model


def evaluate_agent(model, eval_df, config):
    print("Starting agent evaluation...")
    env = StockTradingEnv(eval_df, config)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
    print(f"Evaluation completed.")
    return env.history


# --- 6. 性能计算与可视化 ---
def calculate_episode_stats(history, initial_capital):
    if not history:
        return {
            "final_value": initial_capital,
            "total_return": 0.0,
            "annual_volatility": 0.0,
            "total_rewards": 0.0
        }

    history_df = pd.DataFrame(history)
    final_value = history_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1

    daily_returns = history_df['portfolio_value'].pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        annual_volatility = 0.0
    else:
        annual_volatility = daily_returns.std() * np.sqrt(252)

    total_rewards = history_df['reward'].sum()

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annual_volatility": annual_volatility,
        "total_rewards": total_rewards
    }


# --- 7. 增强的可视化函数 ---
def plot_results(history, config, benchmark_df, title, training_losses=None, episode_rewards=None):
    """增强的可视化函数，包含训练指标"""
    if training_losses and episode_rewards:
        # 创建4个子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    else:
        # 创建2个子图
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

    history_df = pd.DataFrame(history).set_index('date')
    initial_capital = config['environment']['initial_capital']

    # --- 图1: 净值与价格对比 ---
    ax1.set_title(title, fontsize=16)

    normalized_portfolio_value = history_df['portfolio_value'] / initial_capital
    ax1.plot(history_df.index, normalized_portfolio_value,
             label='DQN Agent Portfolio (Normalized)', color='blue', linewidth=2, zorder=2)

    normalized_benchmark = benchmark_df['close'] / benchmark_df['close'].iloc[0]
    ax1.plot(benchmark_df.index, normalized_benchmark,
             label='Buy and Hold Benchmark (Normalized)', color='grey',
             linestyle='--', linewidth=2, zorder=1)

    ax1.set_ylabel('Normalized Net Value (Start=1)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    # 第二个Y轴显示原始价格
    ax1_twin = ax1.twinx()
    ax1_twin.plot(benchmark_df.index, benchmark_df['close'],
                  label='ETF Price (Raw)', color='orange', alpha=0.4, zorder=0)
    ax1_twin.set_ylabel('ETF Price', color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # --- 训练指标图 ---
    if training_losses and episode_rewards:
        # 图2: 训练损失
        ax2.set_title('Training Loss Over Episodes')
        if training_losses:
            ax2.plot(training_losses, color='red', alpha=0.7)
            # 添加移动平均线
            if len(training_losses) > 10:
                window = min(10, len(training_losses) // 4)
                ma_losses = pd.Series(training_losses).rolling(window=window).mean()
                ax2.plot(ma_losses, color='darkred', linewidth=2, label=f'MA({window})')
                ax2.legend()
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)

        # 图4: 奖励变化
        ax4.set_title('Episode Rewards Over Time')
        if episode_rewards:
            ax4.plot(episode_rewards, color='green', alpha=0.7)
            # 添加移动平均线
            if len(episode_rewards) > 10:
                window = min(10, len(episode_rewards) // 4)
                ma_rewards = pd.Series(episode_rewards).rolling(window=window).mean()
                ax4.plot(ma_rewards, color='darkgreen', linewidth=2, label=f'MA({window})')
                ax4.legend()
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Total Reward')
        ax4.grid(True, alpha=0.3)

    # --- 持仓状态时间轴 ---
    ax3.set_title('Agent Position Status')

    actions = history_df['action']
    buy_times = actions[actions == 1].index
    sell_times = actions[actions == 2].index

    position_active = False
    last_buy_time = None
    legend_added = False

    for date in history_df.index:
        if date in buy_times:
            position_active = True
            last_buy_time = date
        if date in sell_times:
            if position_active:
                label = 'Long Position' if not legend_added else ""
                ax3.axvspan(last_buy_time, date, color='green', alpha=0.3, label=label)
                legend_added = True
            position_active = False

    if position_active and last_buy_time is not None:
        label = 'Long Position' if not legend_added else ""
        ax3.axvspan(last_buy_time, history_df.index[-1], color='green', alpha=0.3, label=label)

    ax3.set_yticks([])
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')

    handles, labels = ax3.get_legend_handles_labels()
    if handles:
        ax3.legend(handles, labels, loc='upper left')

    plt.tight_layout()
    plt.show()


def print_performance_comparison(train_history, test_history, train_df, test_df, config):
    """打印详细的性能对比"""
    initial_capital = config['environment']['initial_capital']

    # 计算DQN性能
    train_stats = calculate_episode_stats(train_history, initial_capital)
    test_stats = calculate_episode_stats(test_history, initial_capital)

    # 计算基准性能 (买入持有)
    train_benchmark_return = (train_df['close'].iloc[-1] / train_df['close'].iloc[0]) - 1
    test_benchmark_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0]) - 1

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\n{'Strategy':<20} {'Train Return':<15} {'Test Return':<15} {'Train Sharpe':<15} {'Test Sharpe':<15}")
    print("-" * 80)

    # 计算夏普比率
    def calculate_sharpe(history, initial_capital):
        if not history:
            return 0.0
        df = pd.DataFrame(history)
        returns = df['portfolio_value'].pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    train_sharpe = calculate_sharpe(train_history, initial_capital)
    test_sharpe = calculate_sharpe(test_history, initial_capital)

    print(f"{'DQN Agent':<20} {train_stats['total_return']:<15.2%} {test_stats['total_return']:<15.2%} "
          f"{train_sharpe:<15.3f} {test_sharpe:<15.3f}")
    print(f"{'Buy & Hold':<20} {train_benchmark_return:<15.2%} {test_benchmark_return:<15.2%} {'N/A':<15} {'N/A':<15}")

    print(f"\nDQN vs Benchmark Excess Return:")
    print(f"  Training: {(train_stats['total_return'] - train_benchmark_return):.2%}")
    print(f"  Testing:  {(test_stats['total_return'] - test_benchmark_return):.2%}")

    # 额外的统计信息
    print(f"\nDetailed Statistics:")
    print(f"  Training Period: {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Testing Period: {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}")

    if train_history:
        train_df_hist = pd.DataFrame(train_history)
        buy_actions = (train_df_hist['action'] == 1).sum()
        sell_actions = (train_df_hist['action'] == 2).sum()
        print(f"  Training - Buy Actions: {buy_actions}, Sell Actions: {sell_actions}")

    if test_history:
        test_df_hist = pd.DataFrame(test_history)
        buy_actions = (test_df_hist['action'] == 1).sum()
        sell_actions = (test_df_hist['action'] == 2).sum()
        print(f"  Testing - Buy Actions: {buy_actions}, Sell Actions: {sell_actions}")


# --- 7. 主程序 ---
if __name__ == '__main__':
    # 将省略的函数定义粘贴到这里
    # 例如：
    def run_moving_average_strategy(df, config):
        print("Running Moving Average Strategy...")
        short_window = config['ma_strategy']['short_window']
        long_window = config['ma_strategy']['long_window']
        initial_capital = config['environment']['initial_capital']
        cost_pct = config['environment']['transaction_cost_pct']
        signals = df.copy()
        signals['sma_short'] = signals['close'].rolling(window=short_window, min_periods=1).mean()
        signals['sma_long'] = signals['close'].rolling(window=long_window, min_periods=1).mean()
        signals['signal'] = 0.0
        signals['signal'][short_window:] = np.where(
            signals['sma_short'][short_window:] > signals['sma_long'][short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        cash = initial_capital
        shares = 0
        history = []
        for i in range(len(signals)):
            signal = signals['positions'].iloc[i]
            current_price = signals['close'].iloc[i]
            action_taken = 0
            if signal == 1.0 and shares == 0:
                action_taken = 1
                cost = cash * cost_pct
                buy_capital = cash - cost
                shares = buy_capital / current_price
                cash = 0
            elif signal == -1.0 and shares > 0:
                action_taken = 2
                proceeds = shares * current_price
                cost = proceeds * cost_pct
                cash = proceeds - cost
                shares = 0
            portfolio_value = cash + shares * current_price
            history.append({'date': signals.index[i], 'portfolio_value': portfolio_value, 'action': action_taken,
                            'price': current_price, 'reward': 0})
        print("Moving Average Strategy finished.")
        return history


    def calculate_episode_stats(history, initial_capital):
        if not history: return {"final_value": initial_capital, "total_return": 0.0, "annual_volatility": 0.0,
                                "total_rewards": 0.0}
        history_df = pd.DataFrame(history)
        final_value = history_df['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_capital) - 1
        daily_returns = history_df['portfolio_value'].pct_change().dropna()
        if len(daily_returns) < 2 or daily_returns.std() == 0:
            annual_volatility = 0.0
        else:
            annual_volatility = daily_returns.std() * np.sqrt(252)
        total_rewards = history_df['reward'].sum()
        return {"final_value": final_value, "total_return": total_return, "annual_volatility": annual_volatility,
                "total_rewards": total_rewards}


    def plot_results(ppo_history, ma_history, config, benchmark_df, title):
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ppo_history_df = pd.DataFrame(ppo_history).set_index('date')
        ma_history_df = pd.DataFrame(ma_history).set_index('date')
        initial_capital = config['environment']['initial_capital']
        ax1.set_title(title, fontsize=16)
        normalized_ppo_portfolio = ppo_history_df['portfolio_value'] / initial_capital
        ax1.plot(ppo_history_df.index, normalized_ppo_portfolio, label='PPO Agent Portfolio', color='blue', linewidth=2,
                 zorder=3)
        normalized_ma_portfolio = ma_history_df['portfolio_value'] / initial_capital
        ax1.plot(ma_history_df.index, normalized_ma_portfolio, label='Moving Average Strategy', color='purple',
                 linestyle='-.', linewidth=2, zorder=2)
        normalized_benchmark = benchmark_df['close'] / benchmark_df['close'].iloc[0]
        ax1.plot(benchmark_df.index, normalized_benchmark, label='Buy and Hold Benchmark', color='grey', linestyle='--',
                 linewidth=1.5, zorder=1)
        ax1.set_ylabel('Normalized Net Value (Start=1)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)
        plt.show()


    def print_performance_comparison(ppo_train_history, ma_train_history, train_df, ppo_test_history, ma_test_history,
                                     test_df, config):
        initial_capital = config['environment']['initial_capital']
        ppo_train_stats = calculate_episode_stats(ppo_train_history, initial_capital)
        ppo_test_stats = calculate_episode_stats(ppo_test_history, initial_capital)
        ma_train_stats = calculate_episode_stats(ma_train_history, initial_capital)
        ma_test_stats = calculate_episode_stats(ma_test_history, initial_capital)
        train_benchmark_return = (train_df['close'].iloc[-1] / train_df['close'].iloc[0]) - 1
        test_benchmark_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0]) - 1

        print("\n" + "=" * 80);
        print("PERFORMANCE COMPARISON");
        print("=" * 80)
        header = f"{'Strategy':<25} {'Train Return':<15} {'Test Return':<15}"
        print(f"\n{header}");
        print("-" * len(header))
        print(f"{'PPO Agent':<25} {ppo_train_stats['total_return']:<15.2%} {ppo_test_stats['total_return']:<15.2%}")
        print(f"{'Moving Average':<25} {ma_train_stats['total_return']:<15.2%} {ma_test_stats['total_return']:<15.2%}")
        print(f"{'Buy & Hold':<25} {train_benchmark_return:<15.2%} {test_benchmark_return:<15.2%}")
        print("\n" + "=" * 80)


    print("=" * 80)
    print("PPO TRADING SYSTEM using Stable Baselines3")
    print("=" * 80)
    s_t = time.time()

    try:
        # 1. 设置随机种子
        print("\n1. Setting Random Seeds...")
        set_seeds(CONFIG["infra"]["seed"])

        # 2. 加载和预处理数据
        print("\n2. Loading and Preprocessing Data...")
        full_df = get_data(the_fund_code, the_basic_info_file_path_dict, the_metrics_info_file_path_dict)
        train_df, test_df = preprocess_and_split_data(full_df, CONFIG)

        # 3. 训练PPO智能体
        print("\n3. Training PPO Agent...")
        trained_model = train_ppo_agent(CONFIG, train_df)

        # 4. 运行基准策略
        print("\n4. Running Benchmark Strategies...")
        ma_train_history = run_moving_average_strategy(train_df, CONFIG)
        ma_test_history = run_moving_average_strategy(test_df, CONFIG)

        # 5. 评估PPO智能体
        print("\n5. Evaluating PPO Agent...")
        ppo_train_history = evaluate_agent(trained_model, train_df, CONFIG)
        ppo_test_history = evaluate_agent(trained_model, test_df, CONFIG)

        # 6. 性能对比
        print("\n6. Performance Analysis...")
        print_performance_comparison(ppo_train_history, ma_train_history, train_df,
                                     ppo_test_history, ma_test_history, test_df, CONFIG)

        # 7. 可视化结果
        print("\n7. Generating Visualizations...")
        plot_results(ppo_train_history, ma_train_history, CONFIG, train_df,
                     f"PPO Performance (Training Set) - {CONFIG['data']['fund_code']}")
        plot_results(ppo_test_history, ma_test_history, CONFIG, test_df,
                     f"PPO Performance (Test Set) - {CONFIG['data']['fund_code']}")

        # 8. 保存模型
        print("\n8. Saving PPO Model...")
        trained_model.save(f"ppo_model_{CONFIG['data']['fund_code']}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print(f"\nTotal execution time: {time.time() - s_t:.2f} seconds")
