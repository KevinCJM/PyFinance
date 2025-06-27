# -----------------------------------------------------------------------------
# 导入必要的库
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Dirichlet, Bernoulli
import torch.optim as optim
import gym
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import warnings
from typing import List, Dict

# -----------------------------------------------------------------------------
# 全局设置
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# 模块一：数据处理
# ==============================================================================
def load_and_preprocess_data(file_path: str, fund_list: List[str], window_size: int = 60) -> (
        Dict[str, pd.DataFrame], pd.DataFrame):
    """
    加载并预处理ETF日线数据。使用滚动窗口进行标准化，以避免前视偏差。
    """
    print("--- 1. 开始加载和预处理数据 (使用滚动标准化) ---")
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"错误：'{file_path}' 文件未找到。请确保文件路径正确。")
        raise

    etf_df = df[df['ts_code'].isin(fund_list)].reset_index(drop=True)
    etf_df['trade_date'] = pd.to_datetime(etf_df['trade_date'])
    etf_df['log_return'] = etf_df.groupby('ts_code')['close'].transform(lambda x: np.log(x / x.shift(1)))

    min_date = etf_df.groupby('ts_code')['trade_date'].min().reset_index()
    etf_df = etf_df[etf_df['trade_date'] >= min_date['trade_date'].max()].reset_index(drop=True)

    def rolling_scale(pivot_df: pd.DataFrame, window: int) -> pd.DataFrame:
        rolling_mean = pivot_df.rolling(window=window, min_periods=1).mean().shift(1)
        rolling_std = pivot_df.rolling(window=window, min_periods=1).std().shift(1)
        rolling_std.replace(0, 1, inplace=True)
        return (pivot_df - rolling_mean) / rolling_std

    def process_pivot_df_rolling(the_df, value_col, window, fill_val=None):
        pivot = the_df.pivot(index='trade_date', columns='ts_code', values=value_col)
        if value_col not in ['log_return']:
            if fill_val is not None:
                pivot = pivot.fillna(fill_val)
            else:
                pivot = pivot.fillna(method='ffill').fillna(method='bfill')
            pivot = rolling_scale(pivot, window)
        else:
            pivot = pivot.fillna(0)
        return pivot

    feature_cols = ['open', 'close', 'high', 'low', 'vol', 'amount']
    all_data_dict = {col: process_pivot_df_rolling(etf_df, col, window_size) for col in feature_cols}
    log_df = etf_df.pivot(index='trade_date', columns='ts_code', values='log_return').fillna(0)

    first_valid_index = all_data_dict['close'].first_valid_index()
    for key in all_data_dict:
        all_data_dict[key] = all_data_dict[key].loc[first_valid_index:]
    log_df = log_df.loc[first_valid_index:]

    print("数据处理完成。")
    return all_data_dict, log_df


# ==============================================================================
# 模块二：基准模型
# ==============================================================================
def calculate_markowitz_weights(log_returns_df: pd.DataFrame) -> np.ndarray:
    """计算马科维茨最优投资组合权重（最大化夏普比率）"""
    mu = log_returns_df.mean() * 252
    cov = log_returns_df.cov() * 252
    n_portfolios = 10000;
    n_assets = len(mu)
    results = np.zeros((3, n_portfolios));
    weights_record = []
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        p_return = np.sum(mu * weights)
        p_stddev = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        results[0, i], results[1, i], results[2, i] = p_return, p_stddev, p_return / p_stddev if p_stddev > 0 else 0
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    print("马科维茨最优权重 (最大夏普比率):")
    for i, w in enumerate(optimal_weights):
        print(f"  {log_returns_df.columns[i]}: {w:.4f}")
    return optimal_weights


# ==============================================================================
# 模块三：强化学习环境
# ==============================================================================
class AssetAllocationEnv(gym.Env):
    """用于资产配置的Gym环境 (融合决策版)"""

    def __init__(self, data_dict: Dict[str, pd.DataFrame], log_returns: pd.DataFrame, config: Dict):
        super(AssetAllocationEnv, self).__init__()
        self.data_dict, self.log_returns = data_dict, log_returns
        self.n_steps, self.n_assets = log_returns.shape
        self.n_features = len(data_dict)
        self.window_size = config.get('window_size', 30)
        self.transaction_cost_rate = config.get('transaction_cost_rate', 0.001)
        self.risk_penalty_coef = config.get('risk_penalty_coef', 0.5)
        self.reward_window = config.get('reward_window', 22)
        cnn_shape = (self.n_features, self.window_size, self.n_assets)
        weights_shape = (self.n_assets,)
        self.observation_space = spaces.Dict({
            'market_image': spaces.Box(low=-np.inf, high=np.inf, shape=cnn_shape, dtype=np.float32),
            'current_weights': spaces.Box(low=0, high=1, shape=weights_shape, dtype=np.float32)})
        self.action_space = spaces.Dict({
            'weights': spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32),
            'rebalance': spaces.Discrete(2)})
        self.reset()

    def _get_observation(self):
        end_idx = self.current_step + 1
        start_idx = max(0, end_idx - self.window_size)
        market_image_list = [self.data_dict[key].iloc[start_idx:end_idx].values.T for key in self.data_dict]
        market_image = np.array(market_image_list)
        if market_image.shape[2] < self.window_size:
            padding = np.zeros((self.n_features, self.n_assets, self.window_size - market_image.shape[2]))
            market_image = np.concatenate((padding, market_image), axis=2)
        market_image = np.transpose(market_image, (0, 2, 1))
        return {'market_image': market_image.astype(np.float32), 'current_weights': self.weights.astype(np.float32)}

    def reset(self):
        self.current_step, self.portfolio_value = 0, 1.0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value_history, self.weights_history = [self.portfolio_value], [self.weights]
        self.portfolio_log_returns_history = []
        self.rebalance_history = []  # 记录调仓的时间点
        return self._get_observation()

    def step(self, action: Dict[str, np.ndarray]):
        rebalance_decision = action['rebalance']
        target_weights = action['weights']
        old_weights = self.weights
        transaction_cost = 0

        if rebalance_decision == 1:
            turnover = np.sum(np.abs(target_weights - old_weights)) / 2.0
            transaction_cost = turnover * self.transaction_cost_rate
            self.weights = target_weights
            self.rebalance_history.append(self.current_step)

        returns = self.log_returns.iloc[self.current_step].values
        portfolio_log_return = np.dot(old_weights, returns)
        self.portfolio_log_returns_history.append(portfolio_log_return)

        self.portfolio_value *= np.exp(portfolio_log_return)
        self.portfolio_value *= (1 - transaction_cost)

        base_reward = portfolio_log_return
        if len(self.portfolio_log_returns_history) < self.reward_window:
            volatility_penalty = 0
        else:
            rolling_std = np.std(self.portfolio_log_returns_history[-self.reward_window:])
            volatility_penalty = self.risk_penalty_coef * rolling_std
        reward = base_reward - volatility_penalty - transaction_cost

        self.portfolio_value_history.append(self.portfolio_value)
        self.weights_history.append(self.weights)
        self.current_step += 1
        done = self.current_step >= self.n_steps

        next_obs = self._get_observation() if not done else {
            'market_image': np.zeros((self.n_features, self.window_size, self.n_assets), dtype=np.float32),
            'current_weights': np.zeros(self.n_assets, dtype=np.float32)}
        return next_obs, reward, done, {}


# ==============================================================================
# 4. PPO Agent 模块 (融合决策版)
# ==============================================================================
class ActorCNN(nn.Module):
    """演员网络 - 具有权重和调仓两个决策头"""

    def __init__(self, n_features: int, n_assets: int, window_size: int, config: Dict):
        super(ActorCNN, self).__init__()
        cnn_kernel_time_size = config.get('cnn_kernel_time_size', 5)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=32, kernel_size=(cnn_kernel_time_size, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, n_assets)),
            nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, window_size, n_assets)
            cnn_out_dim = self.cnn(dummy_input).shape[1]
        self.mlp_body = nn.Sequential(nn.Linear(cnn_out_dim + n_assets, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.weights_head = nn.Sequential(nn.Linear(128, n_assets), nn.Softplus())
        self.rebalance_head = nn.Linear(128, 1)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        cnn_out = self.cnn(state_dict['market_image'])
        combined_input = torch.cat([cnn_out, state_dict['current_weights']], dim=1)
        body_out = self.mlp_body(combined_input)
        alphas = self.weights_head(body_out) + 1e-6
        weights_dist = Dirichlet(alphas)
        rebalance_logit = self.rebalance_head(body_out)
        rebalance_dist = Bernoulli(logits=rebalance_logit)
        return weights_dist, rebalance_dist


class CriticCNN(nn.Module):
    """评论家网络 - 与Actor对称"""

    def __init__(self, n_features: int, n_assets: int, window_size: int, config: Dict):
        super(CriticCNN, self).__init__()
        cnn_kernel_time_size = config.get('cnn_kernel_time_size', 5)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=32, kernel_size=(cnn_kernel_time_size, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, n_assets)),
            nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, window_size, n_assets)
            cnn_out_dim = self.cnn(dummy_input).shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + n_assets, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        cnn_out = self.cnn(state_dict['market_image'])
        combined_input = torch.cat([cnn_out, state_dict['current_weights']], dim=1)
        return self.mlp(combined_input)


class PPOAgent:
    """PPO智能体，封装了训练和评估逻辑"""

    def __init__(self, n_features, n_assets, window_size, config: Dict):
        self.config = config
        self.actor = ActorCNN(n_features, n_assets, window_size, config)
        self.critic = CriticCNN(n_features, n_assets, window_size, config)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'],
                                          weight_decay=config.get('weight_decay', 0))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'],
                                           weight_decay=config.get('weight_decay', 0))
        self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=0.1,
                                                                 total_iters=config['num_iterations'])
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0,
                                                                  end_factor=0.1, total_iters=config['num_iterations'])

    def _process_state_dict_to_tensors(self, state_dict):
        market_image_t = torch.FloatTensor(state_dict['market_image']).unsqueeze(0)
        current_weights_t = torch.FloatTensor(state_dict['current_weights']).unsqueeze(0)
        return {'market_image': market_image_t, 'current_weights': current_weights_t}

    def train(self, env: gym.Env):
        print("\n--- 4. 开始PPO训练 (融合决策版) ---")
        history = {'total_rewards': [], 'actor_losses': [], 'critic_losses': [], 'entropies': [], 'learning_rates': []}
        n_steps = env.n_steps
        for iteration in range(self.config['num_iterations']):
            states, weights_actions, rebalance_actions, log_probs, rewards, dones, values = [], [], [], [], [], [], []
            state = env.reset()
            for _ in range(n_steps):
                state_dict_t = self._process_state_dict_to_tensors(state)
                with torch.no_grad():
                    weights_dist, rebalance_dist = self.actor(state_dict_t)
                    w_action, r_action = weights_dist.sample(), rebalance_dist.sample()
                    log_prob = weights_dist.log_prob(w_action) + rebalance_dist.log_prob(r_action)
                    value = self.critic(state_dict_t)
                action_to_env = {'weights': w_action.squeeze().cpu().numpy(),
                                 'rebalance': r_action.squeeze().cpu().numpy()}
                next_state, reward, done, _ = env.step(action_to_env)
                states.append(state);
                weights_actions.append(w_action.cpu().numpy().flatten())
                rebalance_actions.append(r_action.cpu().numpy().flatten())
                log_probs.append(log_prob.cpu().numpy())
                rewards.append(float(reward))
                dones.append(done)
                values.append(value.item())
                state = next_state

            history['total_rewards'].append(float(sum(rewards)))
            self.actor_scheduler.step();
            self.critic_scheduler.step()
            current_lr = self.actor_optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            if (iteration + 1) % 10 == 0:
                print(
                    f"  Iteration {iteration + 1}/{self.config['num_iterations']}, Total Reward: {history['total_rewards'][-1]:.4f}, Current LR: {current_lr:.2e}")

            advantages, returns = self._calculate_gae_and_returns(rewards, values, dones, next_state)

            market_images_t = torch.FloatTensor(np.array([s['market_image'] for s in states]))
            current_weights_t = torch.FloatTensor(np.array([s['current_weights'] for s in states]))
            weights_actions_t, rebalance_actions_t = torch.FloatTensor(np.array(weights_actions)), torch.FloatTensor(
                np.array(rebalance_actions))
            log_probs_t, advantages_t, returns_t = torch.FloatTensor(np.array(log_probs).flatten()), torch.FloatTensor(
                advantages), torch.FloatTensor(returns)

            self._optimize_policy(n_steps, market_images_t, current_weights_t, weights_actions_t, rebalance_actions_t,
                                  log_probs_t, advantages_t, returns_t, history)

        print("--- PPO训练完成 ---")
        return env.portfolio_value_history, history

    def _calculate_gae_and_returns(self, rewards, values, dones, next_state):
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_advantage = 0
        with torch.no_grad():
            last_value = self.critic(self._process_state_dict_to_tensors(next_state)).item() if not dones[-1] else 0
        for t in reversed(range(n_steps)):
            next_non_terminal = 1.0 - (dones[t + 1] if t < n_steps - 1 else dones[-1])
            next_value = values[t + 1] if t < n_steps - 1 else last_value
            delta = rewards[t] + self.config['gamma'] * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.config['gamma'] * self.config[
                'gae_lambda'] * next_non_terminal * last_advantage
        returns = advantages + np.array(values)
        return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8), returns

    def _optimize_policy(self, n_steps, market_images_t, current_weights_t, weights_actions_t, rebalance_actions_t,
                         log_probs_t, advantages_t, returns_t, history):
        epoch_actor_losses, epoch_critic_losses, epoch_entropies = [], [], []
        for _ in range(self.config['n_epochs']):
            indices = np.arange(n_steps)
            np.random.shuffle(indices)
            for start in range(0, n_steps, self.config['batch_size']):
                batch_indices = indices[start: start + self.config['batch_size']]
                batch_state_dict = {'market_image': market_images_t[batch_indices],
                                    'current_weights': current_weights_t[batch_indices]}

                weights_dist, rebalance_dist = self.actor(batch_state_dict)
                new_log_prob_w = weights_dist.log_prob(weights_actions_t[batch_indices])
                new_log_prob_r = rebalance_dist.log_prob(rebalance_actions_t[batch_indices])
                new_log_probs = new_log_prob_w + new_log_prob_r.squeeze()
                entropy = weights_dist.entropy().mean() + rebalance_dist.entropy().mean()

                ratio = torch.exp(new_log_probs - log_probs_t[batch_indices])
                term1 = ratio * advantages_t[batch_indices]
                term2 = torch.clamp(ratio, 1 - self.config['epsilon'], 1 + self.config['epsilon']) * advantages_t[
                    batch_indices]
                actor_loss = -torch.min(term1, term2).mean() - self.config['entropy_coef'] * entropy
                self.actor_optimizer.zero_grad();
                actor_loss.backward();
                self.actor_optimizer.step()

                predicted_values = self.critic(batch_state_dict).squeeze()
                critic_loss = nn.MSELoss()(predicted_values, returns_t[batch_indices])
                self.critic_optimizer.zero_grad();
                critic_loss.backward();
                self.critic_optimizer.step()

                epoch_actor_losses.append(actor_loss.item());
                epoch_critic_losses.append(critic_loss.item());
                epoch_entropies.append(entropy.item())
        history['actor_losses'].append(np.mean(epoch_actor_losses));
        history['critic_losses'].append(np.mean(epoch_critic_losses));
        history['entropies'].append(np.mean(epoch_entropies))

    def evaluate(self, env: gym.Env, train_final_value: float):
        print("\n--- 5. 在测试集上评估已训练的PPO模型 ---")
        state = env.reset()
        env.portfolio_value = train_final_value;
        env.portfolio_value_history = [train_final_value]
        self.actor.eval();
        self.critic.eval()
        with torch.no_grad():
            for _ in range(env.n_steps):
                state_dict_t = self._process_state_dict_to_tensors(state)
                weights_dist, rebalance_dist = self.actor(state_dict_t)
                w_action, r_action = weights_dist.sample(), rebalance_dist.sample()  # or dist.mean for deterministic
                action_to_env = {'weights': w_action.squeeze().cpu().numpy(),
                                 'rebalance': r_action.squeeze().cpu().numpy()}
                state, _, done, _ = env.step(action_to_env)
                if done: break
        print("--- PPO评估完成 ---")
        return env.portfolio_value_history, env.weights_history, env.rebalance_history


# ==============================================================================
# 5. 可视化模块
# ==============================================================================
def plot_results(train_results, test_results):
    print("\n--- 6. 生成业绩对比图表 ---")
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle('PPO Performance Analysis', fontsize=16)

    axs[0].plot(train_results['index'], train_results['ppo'], label='PPO (In-Sample)', linewidth=2)
    axs[0].plot(train_results['index'], train_results['markowitz'], label='Markowitz (In-Sample)', linestyle='--')
    axs[0].set_title('Performance on Training Set');
    axs[0].legend()

    ppo_norm = test_results['ppo'] / test_results['ppo'][0]
    markowitz_norm = test_results['markowitz'] / test_results['markowitz'][0]
    axs[1].plot(test_results['index'][1:], ppo_norm[1:], label='PPO (Out-of-Sample, Normalized)', linewidth=2)
    axs[1].plot(test_results['index'][1:], markowitz_norm[1:], label='Markowitz (Out-of-Sample, Normalized)',
                linestyle='--')
    axs[1].set_title('Normalized Performance on Test Set');
    axs[1].legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.stackplot(test_results['index'], np.array(test_results['weights']).T, labels=test_results['asset_labels'],
                  alpha=0.8)
    rebalance_points, rebalance_dates = test_results['rebalance_points'], test_results['index'][
        test_results['rebalance_points']]
    if len(rebalance_dates) > 0:
        plt.scatter(rebalance_dates, np.ones(len(rebalance_dates)), marker='v', color='red', s=50, zorder=10,
                    label=f'Rebalance Points ({len(rebalance_dates)} times)')
    plt.title('PPO Dynamic Asset Allocation on Test Set');
    plt.xlabel('Date');
    plt.ylabel('Asset Weight')
    plt.legend(loc='upper left', ncol=2, fontsize='small');
    plt.margins(0, 0);
    plt.show()


def plot_convergence(history: Dict):
    print("\n--- 7. 生成收敛过程图表 ---")
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('PPO Convergence Metrics', fontsize=16)
    axs[0, 0].plot(history['total_rewards'])
    axs[0, 0].set_title('Total Reward per Iteration')
    axs[0, 1].plot(history['critic_losses'], color='orange')
    axs[0, 1].set_title('Critic Loss')
    axs[1, 0].plot(history['actor_losses'], color='green')
    axs[1, 0].set_title('Actor Loss')
    axs[1, 1].plot(history['entropies'], color='red')
    axs[1, 1].set_title('Policy Entropy')
    axs[2, 0].plot(history['learning_rates'], color='purple')
    axs[2, 0].set_title('Learning Rate Decay')
    axs[2, 1].axis('off')
    for ax in axs.flat:
        if not ax.get_xlabel(): ax.set_xlabel('Iteration')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    CONFIG = {
        'file_path': './Data/etf_daily.parquet',
        'fund_list': [
            # '510050.SH',  # 上证50ETF
            # '159915.SZ',  # 创业板ETF
            '159912.SZ',  # 沪深300ETF
            # '512500.SH',  # 中证500ETF华夏
            '511010.SH',  # 国债ETF
            # '513100.SH',  # 纳指ETF
            # '513030.SH',  # 德国ETF
            # '513080.SH',  # 法国CAC40ETF
            # '513520.SH',  # 日经ETF
            # '518880.SH',  # 黄金ETF
            # '161226.SZ',  # 国投白银LOF
            # '501018.SH',  # 南方原油LOF
            # '159981.SZ',  # 能源化工ETF
            # '159985.SZ',  # 豆粕ETF
            # '159980.SZ',  # 有色ETF
            '511990.SH',  # 华宝添益货币ETF
        ],
        'test_ratio': 0.2,
        'window_size': 30,
        'cnn_kernel_time_size': 5,
        'actor_lr': 3e-5,
        'critic_lr': 1e-4,
        'weight_decay': 1e-5,
        'gamma': 0.99,
        'epsilon': 0.2,
        'gae_lambda': 0.95,
        'n_epochs': 10,
        'batch_size': 64,
        'num_iterations': 20,
        'entropy_coef': 0.01,
        'risk_penalty_coef': 0.1,
        'reward_window': 22,
        'transaction_cost_rate': 0.001
    }

    all_data_dict, log_df = load_and_preprocess_data(CONFIG['file_path'], CONFIG['fund_list'], CONFIG['window_size'])

    n_total_steps = len(log_df)
    n_test_steps = int(n_total_steps * CONFIG['test_ratio'])
    n_train_steps = n_total_steps - n_test_steps
    data_dict_train = {key: df.iloc[:n_train_steps] for key, df in all_data_dict.items()}
    data_dict_test = {key: df.iloc[n_train_steps:].reset_index(drop=True) for key, df in all_data_dict.items()}
    log_df_train = log_df.iloc[:n_train_steps]
    log_df_test = log_df.iloc[n_train_steps:].reset_index(drop=True)

    markowitz_weights = calculate_markowitz_weights(log_df_train)
    markowitz_returns_train = (log_df_train * markowitz_weights).sum(axis=1)
    markowitz_value_train = np.exp(markowitz_returns_train.cumsum())
    markowitz_returns_test = (log_df_test * markowitz_weights).sum(axis=1)
    markowitz_value_test = np.exp(markowitz_returns_test.cumsum())
    markowitz_value_test = markowitz_value_train.iloc[-1] * markowitz_value_test

    env_train = AssetAllocationEnv(data_dict_train, log_df_train, CONFIG)
    agent = PPOAgent(n_features=env_train.n_features, n_assets=env_train.n_assets, window_size=env_train.window_size,
                     config=CONFIG)
    ppo_value_train, train_history = agent.train(env_train)

    env_test = AssetAllocationEnv(data_dict_test, log_df_test, CONFIG)
    ppo_value_test, ppo_weights_test, ppo_rebalance_test = agent.evaluate(env_test, ppo_value_train[-1])

    train_results = {'index': log_df_train.index.insert(0, log_df_train.index[0] - pd.Timedelta(days=1)),
                     'ppo': ppo_value_train, 'markowitz': np.insert(markowitz_value_train.values, 0, 1.0)}
    test_results = {'index': log_df_test.index.insert(0, log_df_test.index[0] - pd.Timedelta(days=1)),
                    'ppo': ppo_value_test,
                    'markowitz': np.insert(markowitz_value_test.values, 0, markowitz_value_train.iloc[-1]),
                    'weights': ppo_weights_test, 'rebalance_points': ppo_rebalance_test, 'asset_labels': log_df.columns}

    plot_results(train_results, test_results)
    plot_convergence(train_history)
