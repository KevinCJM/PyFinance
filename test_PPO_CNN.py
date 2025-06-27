import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import torch.optim as optim
import gym
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import warnings
from typing import List, Dict

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# 1. 数据处理模块
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
# 2. 投资组合模型模块
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
        results[0, i], results[1, i], results[2, i] = p_return, p_stddev, p_return / p_stddev
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    print("马科维茨最优权重 (最大夏普比率):")
    for i, w in enumerate(optimal_weights):
        print(f"  {log_returns_df.columns[i]}: {w:.4f}")
    return optimal_weights


# ==============================================================================
# 3. 强化学习环境模块 (CNN版)
# ==============================================================================
class AssetAllocationEnv(gym.Env):
    """用于资产配置的Gym环境 (为CNN提供图像式状态，并包含风险调整后奖励)"""

    def __init__(self, data_dict: Dict[str, pd.DataFrame], log_returns: pd.DataFrame, config: Dict):
        super(AssetAllocationEnv, self).__init__()
        self.data_dict = data_dict
        self.log_returns = log_returns
        self.n_steps, self.n_assets = log_returns.shape
        self.n_features = len(data_dict)
        self.window_size = config.get('window_size', 30)
        self.risk_penalty_coef = config.get('risk_penalty_coef', 0.5)
        self.reward_window = config.get('reward_window', 22)

        cnn_shape = (self.n_features, self.window_size, self.n_assets)
        weights_shape = (self.n_assets,)
        self.observation_space = spaces.Dict({
            'market_image': spaces.Box(low=-np.inf, high=np.inf, shape=cnn_shape, dtype=np.float32),
            'current_weights': spaces.Box(low=0, high=1, shape=weights_shape, dtype=np.float32)
        })
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.reset()

    def _get_observation(self):
        end_idx = self.current_step + 1
        start_idx = max(0, end_idx - self.window_size)
        market_image_list = [self.data_dict[key].iloc[start_idx:end_idx].values.T for key in self.data_dict]
        market_image = np.array(market_image_list)

        if market_image.shape[2] < self.window_size:
            padding = np.zeros((self.n_features, self.n_assets, self.window_size - market_image.shape[2]))
            market_image = np.concatenate((padding, market_image), axis=2)

        # 调整维度为 (特征, 时间, 资产)
        market_image = np.transpose(market_image, (0, 2, 1))

        return {'market_image': market_image.astype(np.float32), 'current_weights': self.weights.astype(np.float32)}

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1.0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value_history = [self.portfolio_value]
        self.weights_history = [self.weights]
        self.portfolio_log_returns_history = []
        return self._get_observation()

    def step(self, action: np.ndarray):
        returns = self.log_returns.iloc[self.current_step].values
        portfolio_log_return = np.dot(self.weights, returns)
        self.portfolio_log_returns_history.append(portfolio_log_return)
        self.portfolio_value *= np.exp(portfolio_log_return)

        base_reward = portfolio_log_return
        if len(self.portfolio_log_returns_history) < self.reward_window:
            volatility_penalty = 0
        else:
            rolling_std = np.std(self.portfolio_log_returns_history[-self.reward_window:])
            volatility_penalty = self.risk_penalty_coef * rolling_std
        reward = base_reward - volatility_penalty

        self.weights = action
        self.portfolio_value_history.append(self.portfolio_value)
        self.weights_history.append(self.weights)
        self.current_step += 1
        done = self.current_step >= self.n_steps

        next_obs = self._get_observation() if not done else {
            'market_image': np.zeros((self.n_features, self.window_size, self.n_assets), dtype=np.float32),
            'current_weights': np.zeros(self.n_assets, dtype=np.float32)
        }
        return next_obs, reward, done, {}


# ==============================================================================
# 4. PPO Agent 模块 (全局关系CNN)
# ==============================================================================
class ActorCNN(nn.Module):
    def __init__(self, n_features, n_assets, window_size, config: Dict):
        super(ActorCNN, self).__init__()

        # 从config获取卷积核的时间高度
        kernel_time_size = config.get('cnn_kernel_time_size', 1)

        # 核心修改：使用“全局宽度”的卷积核
        self.cnn = nn.Sequential(
            # Conv2d的输入形状: (N, C_in, H_in, W_in) -> (批大小, 特征数, 时间窗口, 资产数)
            nn.Conv2d(
                in_channels=n_features,
                out_channels=32,
                # 卷积核的高度为k, 宽度为n_assets，一次性看穿所有资产
                kernel_size=(5, n_assets)
            ),
            nn.ReLU(),
            # 可以在这里加更多层，但输出的宽度将变为1
            # 例如，下一层的kernel_size就应该是 (k, 1)
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, window_size, n_assets)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + n_assets, 256), nn.ReLU(),
            nn.Dropout(p=0.1),  # <--- 在全连接层后加入Dropout
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(p=0.1),  # <--- 再加入一层Dropout
            nn.Linear(128, n_assets), nn.Softplus()
        )

    def forward(self, state_dict):
        market_image = state_dict['market_image']
        current_weights = state_dict['current_weights']

        cnn_out = self.cnn(market_image)
        combined_input = torch.cat([cnn_out, current_weights], dim=1)

        alphas = self.mlp(combined_input) + 1e-6
        return Dirichlet(alphas)


class CriticCNN(nn.Module):
    # Critic也使用同样强大的结构来保证信息对等
    def __init__(self, n_features, n_assets, window_size, config: Dict):
        super(CriticCNN, self).__init__()
        kernel_time_size = config.get('cnn_kernel_time_size', 1)
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=n_features,
                out_channels=32,
                kernel_size=(5, n_assets)
            ),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, window_size, n_assets)
            cnn_out_dim = self.cnn(dummy_input).shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + n_assets, 256), nn.ReLU(),
            nn.Dropout(p=0.1),  # <--- 在全连接层后加入Dropout
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(p=0.1),  # <--- 再加入一层Dropout
            nn.Linear(128, 1)
        )

    def forward(self, state_dict):
        market_image = state_dict['market_image']
        current_weights = state_dict['current_weights']
        cnn_out = self.cnn(market_image)
        combined_input = torch.cat([cnn_out, current_weights], dim=1)
        return self.mlp(combined_input)


# ==============================================================================
# 4. PPO Agent 模块 (最终版 - 分层解耦CNN)
# ==============================================================================

class ActorCNN_2(nn.Module):
    """
    演员网络 (Actor) - 采用您设计的分层解耦CNN架构
    """

    def __init__(self, n_features: int, n_assets: int, window_size: int, config: Dict):
        super(ActorCNN_2, self).__init__()

        # 从config获取卷积核的时间高度
        kernel_time_size = config.get('cnn_kernel_time_size', 5)  # 您提议的5天

        # 定义分层CNN网络
        self.cnn = nn.Sequential(
            # --- 第1层：时序特征提取 (专家组) ---
            # 对每个资产独立地进行5天的时间序列卷积
            nn.Conv2d(
                in_channels=n_features,
                out_channels=32,
                kernel_size=(kernel_time_size, 1),  # 宽度为1，实现独立分析
                padding='same'  # 使用padding保持时间维度长度不变
            ),
            nn.ReLU(),

            # --- 第2层：截面特征融合 (首席策略师) ---
            # 在一个时间点上，同时观察所有资产的时序特征
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, n_assets)  # 高度为1，宽度为n_assets，实现全局截面分析
            ),
            nn.ReLU(),

            nn.Flatten()
        )

        # 动态计算CNN输出的维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, window_size, n_assets)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        # MLP决策头 (投决会)
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + n_assets, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_assets), nn.Softplus())

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dirichlet:
        market_image = state_dict['market_image']
        current_weights = state_dict['current_weights']

        # 特征提取
        cnn_out = self.cnn(market_image)

        # 信息融合与决策
        combined_input = torch.cat([cnn_out, current_weights], dim=1)
        alphas = self.mlp(combined_input) + 1e-6
        return Dirichlet(alphas)


class CriticCNN_2(nn.Module):
    """
    评论家网络 (Critic) - 采用与Actor对称的强大结构
    """

    def __init__(self, n_features: int, n_assets: int, window_size: int, config: Dict):
        super(CriticCNN_2, self).__init__()
        kernel_time_size = config.get('cnn_kernel_time_size', 5)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=32, kernel_size=(kernel_time_size, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, n_assets)),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, window_size, n_assets)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + n_assets, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        market_image = state_dict['market_image']
        current_weights = state_dict['current_weights']
        cnn_out = self.cnn(market_image)
        combined_input = torch.cat([cnn_out, current_weights], dim=1)
        return self.mlp(combined_input)


class PPOAgent:
    def __init__(self, n_features, n_assets, window_size, config: Dict):
        self.config = config
        self.actor = ActorCNN(n_features, n_assets, window_size, config)
        self.critic = CriticCNN(n_features, n_assets, window_size, config)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=config['actor_lr'],
                                          weight_decay=1e-4
                                          )
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=config['critic_lr'],
                                           weight_decay=1e-4
                                           )
        # self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     self.actor_optimizer,
        #     start_factor=1.0,
        #     end_factor=0.01,
        #     total_iters=config['num_iterations']
        # )
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer,
            gamma=0.999  # 每个step，学习率变为原来的99.9%
        )
        # self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     self.critic_optimizer,
        #     start_factor=1.0,
        #     end_factor=0.01,
        #     total_iters=config['num_iterations']
        # )
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer,
            gamma=0.999  # 每个step，学习率变为原来的99.9%
        )

    def _process_state_dict_to_tensors(self, state_dict):
        market_image_t = torch.FloatTensor(state_dict['market_image']).unsqueeze(0)
        current_weights_t = torch.FloatTensor(state_dict['current_weights']).unsqueeze(0)
        return {'market_image': market_image_t, 'current_weights': current_weights_t}

    def train(self, env: gym.Env):
        print("\n--- 4. 开始PPO训练 ---")
        history = {'total_rewards': [], 'actor_losses': [],
                   'critic_losses': [], 'entropies': [],
                   'learning_rates': [], 'entropy_coef': []}
        n_steps = env.n_steps
        for iteration in range(self.config['num_iterations']):
            market_images, current_weights_list, actions, log_probs, rewards, dones, values = [], [], [], [], [], [], []
            state = env.reset()

            # --- 在迭代开始时计算当前熵系数 ---
            decay_ratio = min(1.0, iteration / self.config['entropy_coef_decay_steps'])
            current_entropy_coef = self.config['entropy_coef_start'] - \
                                   (self.config['entropy_coef_start'] - self.config['entropy_coef_end']) * decay_ratio

            for _ in range(n_steps):
                state_dict_t = self._process_state_dict_to_tensors(state)
                with torch.no_grad():
                    dist = self.actor(state_dict_t)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = self.critic(state_dict_t)
                next_state, reward, done, _ = env.step(action.squeeze().cpu().numpy())
                market_images.append(state['market_image'])
                current_weights_list.append(state['current_weights'])
                actions.append(action.cpu().numpy().flatten())
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

            advantages = np.zeros(n_steps, dtype=np.float32)
            last_advantage = 0
            with torch.no_grad():
                next_state_t = self._process_state_dict_to_tensors(next_state)
                last_value = self.critic(next_state_t).item() if not done else 0
            for t in reversed(range(n_steps)):
                next_non_terminal = 1.0 - (dones[t + 1] if t < n_steps - 1 else done)
                next_value = values[t + 1] if t < n_steps - 1 else last_value
                delta = rewards[t] + self.config['gamma'] * next_value * next_non_terminal - values[t]
                advantages[t] = last_advantage = delta + self.config['gamma'] * self.config[
                    'gae_lambda'] * next_non_terminal * last_advantage
            returns = advantages + np.array(values)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            market_images_t, current_weights_t = torch.FloatTensor(np.array(market_images)), torch.FloatTensor(
                np.array(current_weights_list))
            actions_t, log_probs_t = torch.FloatTensor(np.array(actions)), torch.FloatTensor(
                np.array(log_probs).flatten())
            advantages_t, returns_t = torch.FloatTensor(advantages), torch.FloatTensor(returns)

            epoch_actor_losses, epoch_critic_losses, epoch_entropies = [], [], []
            for _ in range(self.config['n_epochs']):
                indices = np.arange(n_steps)
                np.random.shuffle(indices)
                for start in range(0, n_steps, self.config['batch_size']):
                    batch_indices = indices[start: start + self.config['batch_size']]
                    batch_state_dict = {'market_image': market_images_t[batch_indices],
                                        'current_weights': current_weights_t[batch_indices]}
                    dist = self.actor(batch_state_dict)
                    new_log_probs = dist.log_prob(actions_t[batch_indices])
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_log_probs - log_probs_t[batch_indices])
                    term1 = ratio * advantages_t[batch_indices]
                    term2 = torch.clamp(ratio, 1 - self.config['epsilon'], 1 + self.config['epsilon']) * advantages_t[
                        batch_indices]
                    actor_loss = -torch.min(term1, term2).mean() - current_entropy_coef * entropy
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    epoch_actor_losses.append(actor_loss.item())
                    predicted_values = self.critic(batch_state_dict).squeeze()
                    critic_loss = nn.MSELoss()(predicted_values, returns_t[batch_indices])
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    epoch_critic_losses.append(critic_loss.item())
                    epoch_entropies.append(entropy.item())
            history['actor_losses'].append(np.mean(epoch_actor_losses))
            history['critic_losses'].append(np.mean(epoch_critic_losses))
            history['entropies'].append(np.mean(epoch_entropies))
            history['entropy_coef'].append(np.mean(current_entropy_coef))
        print("--- PPO训练完成 ---")
        return env.portfolio_value_history, history

    def evaluate(self, env: gym.Env, train_final_value: float):
        print("\n--- 5. 在测试集上评估已训练的PPO模型 ---")
        state = env.reset()
        env.portfolio_value = train_final_value
        env.portfolio_value_history = [train_final_value]
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            for _ in range(env.n_steps):
                state_dict_t = self._process_state_dict_to_tensors(state)
                dist = self.actor(state_dict_t)
                action = dist.sample()
                state, _, done, _ = env.step(action.squeeze().cpu().numpy())
                if done:
                    break
        print("--- PPO评估完成 ---")
        return env.portfolio_value_history, env.weights_history


# ==============================================================================
# 5. 可视化模块
# ==============================================================================
def plot_results(train_results, test_results):
    print("\n--- 6. 生成业绩对比图表 ---")
    plt.figure(figsize=(14, 7))
    plt.plot(train_results['index'], train_results['ppo'], label='PPO (In-Sample)', linewidth=2)
    plt.plot(train_results['index'], train_results['markowitz'], label='Markowitz (In-Sample)', linestyle='--')
    plt.title('Performance on Training Set');
    plt.xlabel('Date');
    plt.ylabel('Portfolio Value');
    plt.legend();
    plt.show()

    ppo_norm = test_results['ppo'] / test_results['ppo'][0]
    markowitz_norm = test_results['markowitz'] / test_results['markowitz'][0]
    plt.figure(figsize=(14, 7))
    plt.plot(test_results['index'][1:], ppo_norm[1:], label='PPO (Out-of-Sample, Normalized)', linewidth=2)
    plt.plot(test_results['index'][1:], markowitz_norm[1:], label='Markowitz (Out-of-Sample, Normalized)',
             linestyle='--')
    plt.title('Normalized Performance on Test Set (Starting from 1.0)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.stackplot(test_results['index'], np.array(test_results['weights']).T, labels=test_results['asset_labels'],
                  alpha=0.8)
    plt.title('PPO Dynamic Asset Allocation on Test Set')
    plt.xlabel('Date')
    plt.ylabel('Asset Weight')
    plt.legend(loc='upper left', ncol=2, fontsize='small')
    plt.margins(0, 0)
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

    # 新增 subplot：绘制 entropy_coef 曲线
    axs[2, 1].plot(history['entropy_coef'], color='blue')
    axs[2, 1].set_title('Entropy Coefficient')

    for ax in axs.flat:
        if not ax.get_xlabel():
            ax.set_xlabel('Iteration')

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
        'test_ratio': 0.2,  # 测试数据集比例，用于划分训练集和测试集
        'window_size': 30,  # 窗口大小，表示模型考虑的时间序列长度
        'actor_lr': 3e-5,  # 演员网络学习率，用于更新演员网络参数
        'critic_lr': 1e-4,  # 批评家网络学习率，用于更新批评家网络参数
        'gamma': 0.99,  # 折扣因子，用于计算未来奖励的现值
        'epsilon': 0.2,  # PPO算法中的裁剪范围，限制策略更新的步长
        'gae_lambda': 0.95,  # 通用优势估计中的lambda参数，用于计算优势函数
        'n_epochs': 10,  # 训练轮次，表示策略和价值网络的更新次数
        'batch_size': 64,  # 批次大小，用于训练网络的样本数量
        'num_iterations': 1500,  # 迭代次数，表示整个训练过程中数据集被遍历的次数
        # 'entropy_coef': 0.01,  # 熵系数，用于鼓励策略的探索性行为
        'risk_penalty_coef': 0.6,  # 风险惩罚系数，用于在目标函数中加入对风险的惩罚
        'reward_window': 22,  # 奖励窗口大小，用于跟踪最近的奖励以评估性能
        'cnn_kernel_time_size': 5,  # <--- 新增：第一层CNN的时间核大小
        # --- 新增：熵调度参数 ---
        'entropy_coef_start': 0.01,  # 初始熵系数，鼓励前期探索
        'entropy_coef_end': 0.01,  # 最终熵系数，允许后期充分利用
        'entropy_coef_decay_steps': 1000,  # 在多少次迭代内完成衰减

    }

    all_data_dict, log_df = load_and_preprocess_data(CONFIG['file_path'], CONFIG['fund_list'], CONFIG['window_size'])

    print("\n--- 2. 拆分训练集与测试集 ---")
    n_total_steps = len(log_df)
    n_test_steps = int(n_total_steps * CONFIG['test_ratio'])
    n_train_steps = n_total_steps - n_test_steps
    print(f"总数据量: {n_total_steps} 天, 训练集: {n_train_steps} 天, 测试集: {n_test_steps} 天")
    data_dict_train = {key: df.iloc[:n_train_steps] for key, df in all_data_dict.items()}
    data_dict_test = {key: df.iloc[n_train_steps:] for key, df in all_data_dict.items()}
    log_df_train = log_df.iloc[:n_train_steps]
    log_df_test = log_df.iloc[n_train_steps:]

    print("\n--- 3. 计算马科维茨基准模型 ---")
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
    ppo_value_test, ppo_weights_test = agent.evaluate(env_test, ppo_value_train[-1])

    train_results = {'index': log_df_train.index.insert(0, log_df_train.index[0] - pd.Timedelta(days=1)),
                     'ppo': ppo_value_train, 'markowitz': np.insert(markowitz_value_train.values, 0, 1.0)}
    test_results = {'index': log_df_test.index.insert(0, log_df_test.index[0] - pd.Timedelta(days=1)),
                    'ppo': ppo_value_test,
                    'markowitz': np.insert(markowitz_value_test.values, 0, markowitz_value_train.iloc[-1]),
                    'weights': ppo_weights_test, 'asset_labels': log_df.columns}

    plot_results(train_results, test_results)
    plot_convergence(train_history)
