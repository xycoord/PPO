import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

import numpy as np
from typing import Sequence, OrderedDict
from abc import ABC, abstractmethod

import torch
from torch import nn, optim, distributions
from torch.utils.data import Dataset, DataLoader

import itertools
from tqdm import tqdm

# Video rendering
from pathlib import Path
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

# Tracking
import wandb
wandb.login()

# ====== Pytorch Utils ======

device = None
dtype = torch.float32

_str_to_activation = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'selu': nn.SELU,
    'softplus': nn.Softplus,
    'identity': nn.Identity,
}

class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_layers: int,
                 size: int,
                 activation: nn.Module|str =nn.Tanh,
                 output_activation: nn.Module|str=nn.Identity):
        """Basic MLP network."""
        super().__init__()

        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]

        # Build network
        layers = []
        in_size = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, size))
            layers.append(activation())
            in_size = size
        layers.append(nn.Linear(in_size, output_size))
        layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("Using CPU.")

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def set_seed(seed):
    """Set all the seeds for reproducibility"""
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch on CPU
    torch.cuda.manual_seed(seed)     # PyTorch on GPU (single GPU)
    torch.cuda.manual_seed_all(seed) # PyTorch on all GPUs (multi-GPU)
    torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    torch.backends.cudnn.benchmark = False     # Disable CUDA benchmarking

init_gpu()


# ====== Trajectories & Experiences ======

from dataclasses import dataclass
@dataclass
class Trajectory():
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    truncateds: torch.Tensor
    q_values: torch.Tensor
    advantages: torch.Tensor
    total_reward: float
    length: int

class TrajectoryBuilder():

    def __init__(self, max_ep_len):
        self._observations = []
        self._actions = []
        self._log_probs = []
        self._rewards = []
        self._terminals = []
        self._truncateds = []
        self._done = False
        self._length = 0
        self._max_ep_len = max_ep_len


    def add_step(self, observation, action, log_prob, reward, terminal, truncated):
        assert not self._done

        self._done = terminal or truncated

        self._observations.append(observation)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._terminals.append(self._done)
        self._truncateds.append(truncated)

        self._length += 1

        return self._done


    def get_trajectory(self, final_obs, calculate_q_values, estimate_advantages):

        tensor_kwargs = {'dtype': dtype, 'device': device, 'requires_grad': False}

        final_obs = torch.tensor(final_obs, **tensor_kwargs)
        observations = torch.tensor(self._observations, **tensor_kwargs)
        actions = torch.tensor(self._actions, **tensor_kwargs)
        log_probs = torch.tensor(self._log_probs, **tensor_kwargs)
        terminals = torch.tensor(self._terminals, **tensor_kwargs)
        truncateds = torch.tensor(self._truncateds, **tensor_kwargs)
        rewards = torch.tensor(self._rewards, **tensor_kwargs)

        q_values = calculate_q_values(rewards, truncateds, final_obs).detach()
        advantages = estimate_advantages(observations, rewards, q_values, terminals).detach()
        total_reward = rewards.sum()

        return Trajectory(
            observations,
            actions,
            log_probs,
            rewards,
            terminals,
            truncateds,
            q_values,
            advantages,
            total_reward,
            self._length
        )

@dataclass
class ExperienceData:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    truncateds: torch.Tensor
    q_values: torch.Tensor
    advantages: torch.Tensor

    def to(self, device):
        self.observations = self.observations.to(device)
        self.actions = self.actions.to(device)
        self.log_probs = self.log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.terminals = self.terminals.to(device)
        self.truncateds = self.truncateds.to(device)
        self.q_values = self.q_values.to(device)
        self.advantages = self.advantages.to(device)

    def __len__(self):
        return len(self.observations)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for ExperienceData objects"""
        return ExperienceData(
            observations=torch.stack([item.observations for item in batch]),
            actions=torch.stack([item.actions for item in batch]),
            log_probs=torch.stack([item.log_probs for item in batch]),
            rewards=torch.stack([item.rewards for item in batch]),
            terminals=torch.stack([item.terminals for item in batch]),
            truncateds=torch.stack([item.truncateds for item in batch]),
            q_values=torch.stack([item.q_values for item in batch]),
            advantages=torch.stack([item.advantages for item in batch]),
        )

class ExperienceDataset(Dataset):
    """
    Dataset for experience replay.
    It stores experiences (1 time step) and can be sampled as shuffled batches
    with a data loader.
    """

    def __init__(self, trajectories: list[Trajectory]):

        self._len = sum(map(lambda t: t.length, trajectories))
        obs_dim = trajectories[0].observations.shape[1]
        act_dim = trajectories[0].actions.shape
        discrete = len(act_dim) == 1
        act_dim = () if discrete else act_dim[1]

        tensor_kwargs = {'dtype': dtype, 'device': torch.device('cpu')}

        # Initialize arrays with initial capacity
        self._observations = torch.zeros(combined_shape(self._len, obs_dim), **tensor_kwargs)
        self._actions = torch.zeros(combined_shape(self._len, act_dim), **tensor_kwargs)
        self._log_probs = torch.zeros(self._len, **tensor_kwargs)
        self._rewards = torch.zeros(self._len, **tensor_kwargs)
        self._terminals = torch.zeros(self._len, **tensor_kwargs)
        self._truncateds = torch.zeros(self._len, **tensor_kwargs)
        self._q_values = torch.zeros(self._len, **tensor_kwargs)
        self._advantages = torch.zeros(self._len, **tensor_kwargs)

        # Trajectory metadata
        self._total_rewards = []
        self._episode_lengths = []

        # State tracking
        self._ptr = 0

        for trajectory in trajectories:
            self._store(trajectory)


    def _store(self, trajectory):

        # Prevent overflow
        assert not self._ptr >= self._len, "Dataset full"

        # Store trajectory data
        end_idx = self._ptr + trajectory.length
        self._observations[self._ptr:end_idx] = trajectory.observations
        self._actions[self._ptr:end_idx] = trajectory.actions
        self._log_probs[self._ptr:end_idx] = trajectory.log_probs
        self._rewards[self._ptr:end_idx] = trajectory.rewards
        self._terminals[self._ptr:end_idx] = trajectory.terminals
        self._truncateds[self._ptr:end_idx] = trajectory.truncateds
        self._q_values[self._ptr:end_idx] = trajectory.q_values
        self._advantages[self._ptr:end_idx] = trajectory.advantages

        # Track episode-level information
        self._total_rewards.append(trajectory.total_reward)
        self._episode_lengths.append(trajectory.length)

        # Update position
        self._ptr = end_idx

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return ExperienceData(
            self._observations[idx],
            self._actions[idx],
            self._log_probs[idx],
            self._rewards[idx],
            self._terminals[idx],
            self._truncateds[idx],
            self._q_values[idx],
            self._advantages[idx],
        )

    def normalize_advantages(self):
        self._advantages = (self._advantages - self._advantages.mean()) / (self._advantages.std() + 1e-8)

    def get_all_observations(self):
        return self._observations

    def get_all_q_values(self):
        return self._q_values

    def get_total_rewards(self):
        return torch.tensor(self._total_rewards)

    def get_episode_lengths(self):
        return torch.tensor(self._episode_lengths)

    def get_metrics(self, prefix):
        """Compute metrics for logging."""

        metrics = OrderedDict()

        returns = self.get_total_rewards()
        metrics[f"{prefix}_AverageReturn"] = returns.mean()
        metrics[f"{prefix}_StdReturn"] = returns.std() if len(returns) > 1 else 0.0
        metrics[f"{prefix}_MaxReturn"] = returns.max()
        metrics[f"{prefix}_MinReturn"] = returns.min()

        episode_lengths = self.get_episode_lengths()
        metrics[f"{prefix}_AverageEpLen"] = episode_lengths.float().mean()

        return metrics


# ====== Policy ======

@dataclass
class PolicyConfig:
    ac_dim: int
    ob_dim: int
    discrete: bool
    n_layers: int
    layer_size: int
    learning_rate: float

class MLPPolicy(ABC, nn.Module):
    """
    Base MLP policy, which can take an observation and output a distribution over actions.
    Supports both discrete and continuous actions.
    Maintains its own Adam optimizer.
    Requires the update method to bedefined by subclasses.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()

        net = MLP(
            input_size=config.ob_dim,
            output_size=config.ac_dim,
            n_layers=config.n_layers,
            size=config.layer_size,
        ).to(device)

        if config.discrete:
            self.logits_net = net
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = net
            self.logstd = nn.Parameter(
                torch.zeros(config.ac_dim, dtype=torch.float32, device=device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            config.learning_rate,
        )

        self.config = config

    @torch.no_grad()
    def get_action(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run a forward pass and sample an action from the distribution."""
        distribution = self.forward(from_numpy(observations))
        action = distribution.sample()
        log_prob = to_numpy(distribution.log_prob(action))
        action = to_numpy(action)

        return action, log_prob

    def forward(self, observations: torch.FloatTensor) -> distributions.Distribution:
        """
        Network forward pass
        Returns a distribution over actions.
        """
        if self.config.discrete:
            logits = self.logits_net(observations)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(observations)
            return distributions.MultivariateNormal(mean, torch.diag(torch.exp(self.logstd)))

    @abstractmethod
    def update(self, obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        pass

@dataclass
class PPOUpdaterConfig:
    """All the parameters the policy's update method needs"""
    epochs: int
    batch_size: int
    epsilon: float
    target_kl: float
    entropy_coef: float
    max_grad_norm: float
    learning_rate: float

class MLPPolicyPPO(MLPPolicy):
    """Policy subclass for the PPO-clip algorithm."""

    def update(
        self,
        dataset: ExperienceDataset,
        config: PPOUpdaterConfig
    ) -> dict:
        """
        Implements the PPO-clip actor update.

        Uses epoch-level early stopping to prevent over-training on the current
        dataset of experiences.

        Returns:
            A dictionary of metrics to log.

        Note: loss, entropy and kl are averaged over the final epoch and are
              representative of the final state of the policy.
              grad_norms are taken across all epochs to allow best setting of
              the max_grad_norm hyperparameter.
        """

        # We will sample shuffled batches from the experience dataset
        # to update the policy
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=ExperienceData.collate_fn,
            pin_memory=True
        )

        # Final epoch aggregate metrics
        mean_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0

        # All epoch metrics
        grad_norms = []

        for epoch in range(config.epochs):
            total_loss = 0.0
            total_entropy = 0.0
            total_kl = 0.0
            for batch in dataloader:
                self.optimizer.zero_grad(set_to_none=True)

                batch.to(device)

                loss, approx_kl, entropy = self._compute_loss(batch, config.epsilon, config.entropy_coef)

                # For final epoch metrics
                total_loss += loss.item() * len(batch)
                total_entropy += entropy * len(batch)
                total_kl += approx_kl * len(batch)

                loss.backward()
                # Gradient clipping
                grad_norms.append(torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=config.max_grad_norm).item())

                self.optimizer.step()

            # Limit KL divergence for a set of experiences
            # to ensure we don't overtrain on this set
            mean_kl = total_kl / len(dataset)
            if mean_kl > 1.5 * config.target_kl:
                 break

            mean_loss = total_loss / len(dataset)
            mean_entropy = total_entropy / len(dataset)

        return {
            "Actor Loss": mean_loss,
            "KL Divergence": mean_kl,
            "Actor Entropy": mean_entropy,
            "Actor Grad Norm": np.mean(grad_norms),
            "Actor Grad Norm (max)": np.max(grad_norms)
        }


    def _compute_loss(self, batch: ExperienceData, epsilon: float, entropy_coef: float) -> tuple[torch.Tensor, float, float]:
        """
        Computes the loss for the PPO algorithm.

        Uses a computational trick equivalent to traditional ratio clipping.
        Instead of clipping ratio ∈ [1-ε, 1+ε], we compute:
        - When advantages > 0: minimum(ratio*adv, (1+ε)*adv)
        - When advantages < 0: minimum(ratio*adv, (1-ε)*adv)
        This gives the same conservative estimate as explicit clipping
        more efficiently than torch.clamp.
        """

        # Load data
        observations = batch.observations
        actions = batch.actions
        old_log_probs = batch.log_probs
        advantages = batch.advantages

        # Calculate surrogate
        distribution = self.forward(observations)
        log_probs = distribution.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate = ratio * advantages

        # Calculate bound
        adv_sign = torch.sign(advantages)
        surrogate_bound = (1 + adv_sign * epsilon) * advantages

        entropy = distribution.entropy().mean()

        # Element-wise minimum of original and bound surrogates
        loss_clip = -torch.minimum(surrogate, surrogate_bound).mean()
        loss_entropy = (-entropy_coef) * entropy

        loss = loss_clip + loss_entropy

        approx_kl = (old_log_probs - log_probs).mean().item()

        return loss, approx_kl, entropy.item()


# ====== Critic ======

@dataclass
class CriticConfig:
    ob_dim: int
    n_layers: int
    layer_size: int
    baseline_learning_rate: float
    baseline_gradient_steps: int

class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation.
    Maintains its own Adam optimizer.
    """

    def __init__(self, config: CriticConfig):
        super().__init__()

        self.network = MLP(
            input_size=config.ob_dim,
            output_size=1,
            n_layers=config.n_layers,
            size=config.layer_size,
        ).to(device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            config.baseline_learning_rate,
        )

        self.config = config

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Computes value estimates for given observations.
        Args:
            observations (batch_size, ob_dim)
        Returns:
            Value estimates tensor of shape (batch_size,)
        """
        values = self.network(observations).squeeze(-1)
        return values

    def update(self, trajs_dataset: ExperienceDataset) -> dict:
        observations = trajs_dataset.get_all_observations().to(device)
        q_values = trajs_dataset.get_all_q_values().to(device)
        loss = torch.tensor(0.0, device=device)

        for _ in range(self.config.baseline_gradient_steps):
            self.optimizer.zero_grad(set_to_none=True)

            predicted_values = self.forward(observations)

            loss = torch.nn.functional.mse_loss(predicted_values, q_values)

            loss.backward()
            self.optimizer.step()

        return {
            "Critic Loss": loss.item(),
        }


# ====== Agent ======

@dataclass
class AgentConfig:
    """Parameters the agent itself uses"""
    gamma: float
    gae_lambda: float
    normalize_advantages: bool
    use_reward_to_go: bool

class PPOAgent(nn.Module):
    def __init__(
        self,
        config: AgentConfig,
        policy_config: PolicyConfig,
        critic_config: CriticConfig | None,
        updater_config: PPOUpdaterConfig,
    ):
        super().__init__()

        actor = MLPPolicyPPO(policy_config)
        critic = ValueCritic(critic_config) if critic_config is not None else None

        # Compile for efficiency
        self.actor = torch.compile(actor)
        self.critic = torch.compile(critic) if critic is not None else None

        self.config = config
        self.updater_config = updater_config

    def update(self, dataset: ExperienceDataset) -> dict:
        """
        Updates both actor and critic networks using the provided experience dataset.
        Returns:
            Dictionary of training metrics from both networks.
        """
        # Normalize advantages to reduce variance and stabilize training
        if self.config.normalize_advantages:
            dataset.normalize_advantages()

        info = self.actor.update(dataset, self.updater_config)

        if self.critic is not None:
            critic_info = self.critic.update(dataset)
            info.update(critic_info)

        return info

    def estimate_advantages(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        q_values: torch.Tensor,
        terminals: torch.Tensor,
    ) -> torch.Tensor:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D torch.Tensors.
        """
        if self.critic is None:
            return q_values

        # Critic
        values = self.critic(obs)
        assert values.shape == q_values.shape

        if self.config.gae_lambda is None:
            advantages = q_values - values
            return advantages

        # GAE
        advantages = self._generalised_advantage_estimation(rewards, values, terminals)

        return advantages

    def _generalised_advantage_estimation(self, rewards: torch.Tensor, values: torch.Tensor, terminals: torch.Tensor) -> torch.Tensor:
        """
        Computes GAE advantages using the recursive formula:
        A_t = δ_t + γλ(1-terminal_t)A_{t+1}
        where δ_t = r_t + γV_{t+1} - V_t
        """
        batch_size = rewards.shape[0]
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        zero = torch.zeros(1, device=device)
        values = torch.cat((values, zero))
        deltas = rewards + gamma * values[1:] - values[:-1]
        advantages = torch.zeros(batch_size + 1, device=device)

        # recursively compute advantage estimates starting from timestep T.
        # use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
        # trajectory, and 0 otherwise.
        for i in reversed(range(batch_size)):
            advantages[i] = (
                deltas[i] +
                gamma * gae_lambda * (1 - terminals[i]) * advantages[i+1]
            )

        # remove dummy advantage
        advantages = advantages[:-1]

        return advantages

    def calculate_q_vals(self, rewards: torch.Tensor, truncateds: torch.Tensor, final_obs: torch.Tensor) -> torch.Tensor:
        """Monte Carlo estimation of the Q function."""

        if self.config.use_reward_to_go:
            q_values = self._discounted_reward_to_go(rewards, truncateds, final_obs)
        else:
            q_values = self._discounted_return(rewards)

        return q_values

    def _discounted_return(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Args:
          rewards [r_0, r_1, ..., r_t', ... r_T]
        Returns:
          discounted_return [sum_{t'=0}^T gamma^t' r_{t'}] * T
        """
        discounted_return = sum(self.config.gamma ** t * reward for t, reward in enumerate(rewards))
        return torch.full_like(rewards, discounted_return)

    def _discounted_reward_to_go(self, rewards: torch.Tensor, truncateds: torch.Tensor, final_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
          rewards [r_0, r_1, ..., r_t', ... r_T]
        Returns:
          discounted_rtg s.t.
            discounted_rtg[t'] = sum_{t'=t}^T gamma^(t'-t) * r_{t'}

        Uses dynamic programming algorithm for efficiency.
        """
        T = len(rewards)
        discounted_rtg = torch.zeros_like(rewards)

        # Base case
        if T > 0:
            if self.critic is not None and truncateds[T-1]:
                # Use the critic to estimate future rtg beyond the sampled trajectory
                value = self.critic(final_obs)
                discounted_rtg[T-1] = rewards[T-1] + self.config.gamma * value
            else: # the last reward is just itself
                discounted_rtg[T-1] = rewards[T-1]

        # Work backwards, using previously computed values
        for t in range(T-2, -1, -1):
            discounted_rtg[t] = rewards[t] + self.config.gamma * discounted_rtg[t+1]

        return discounted_rtg


# ====== Trainer ======

@dataclass
class TrainConfig:
    steps: int
    batch_size: int
    max_ep_len: int
    eval_batch_size: int = 400
    eval_steps: int = 10
    log_wandb: bool = False

class PPOTrainer:
    def __init__(self, agent: PPOAgent, envs: gym.vector.SyncVectorEnv, config: TrainConfig):
        self.agent = agent
        self.envs = envs
        self.config = config

    def run(self):
        progress = tqdm(range(self.config.steps))
        display_keys = ["Train_AverageReturn","Train_AverageEpLen","Actor Loss","Critic Loss"]

        total_envsteps = 0
        for step in progress:
            # Sample Trajectories
            experience_dataset = self._sample_trajectories(self.config.batch_size)
            total_envsteps += len(experience_dataset)

            # Agent Update
            train_info: dict = self.agent.update(experience_dataset)

            train_metrics = experience_dataset.get_metrics(prefix="Train")
            train_metrics.update(train_info)
            train_metrics["total_envsteps"] = total_envsteps
            display_metrics = {k: f"{train_metrics[k]:.4f}" for k in display_keys}
            progress.set_postfix(display_metrics)

            if self.config.log_wandb:
                wandb.log(train_metrics, step=step)

            if step % self.config.eval_steps == 0 or step == self.config.steps - 1:
                self._evaluate(step)

        self.envs.close()
        return self.agent

    def _evaluate(self, step: int):
        """Run evaluation episodes and log metrics for a given step."""
        eval_dataset = self._sample_trajectories(self.config.eval_batch_size)
        val_metrics = eval_dataset.get_metrics(prefix="Eval")
        if self.config.log_wandb:
            wandb.log(val_metrics, step=step)

    def _sample_trajectories(self, min_timesteps_per_batch: int) -> ExperienceDataset:
        """Collect rollouts using policy until we have collected at least min_timesteps_per_batch steps."""

        trajectories = []
        total_envsteps = 0

        # Buffers
        traj_builders = [TrajectoryBuilder(self.config.max_ep_len) for _ in range(self.envs.num_envs)]

        # Reset
        observations, info = self.envs.reset()

        while total_envsteps < min_timesteps_per_batch:

            # Sample Actions
            actions, log_probs = self.agent.actor.get_action(observations)

            # Take steps in environments
            next_observations, rewards, terminated, truncated, infos = self.envs.step(actions)

            for b in range(self.envs.num_envs):
                # Record step
                traj_done = traj_builders[b].add_step(observations[b], actions[b], log_probs[b], rewards[b], terminated[b], truncated[b])

                if traj_done:
                    traj = traj_builders[b].get_trajectory(infos["final_obs"][b], self.agent.calculate_q_vals, self.agent.estimate_advantages)
                    trajectories.append(traj)
                    total_envsteps += traj.length

                    if total_envsteps >= min_timesteps_per_batch:
                        break

                    traj_builders[b] = TrajectoryBuilder(self.config.max_ep_len)

            # Update
            observations = next_observations

        return ExperienceDataset(trajectories)


# ====== Sample Single Trajectory (for video) ======

def sample_trajectory(env: gym.Env, agent: PPOAgent, max_length: int) -> Trajectory:
    """Sample a rollout in the environment from a policy."""

    observation, info = env.reset()
    traj_builder = TrajectoryBuilder(max_length)

    rollout_done = False
    while not rollout_done:

        action, log_probs = agent.actor.get_action(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)

        rollout_done = traj_builder.add_step(observation, action, log_probs, reward, terminated, truncated)

        observation = new_observation

    return traj_builder.get_trajectory(observation, agent.calculate_q_vals, agent.estimate_advantages)

# ====== Utils ======

def configs_from_dicts(env_info: dict, agent_args: dict) -> tuple[PolicyConfig, CriticConfig, PPOUpdaterConfig, AgentConfig]:
    agent_config = AgentConfig(
        **{k: v for k, v in agent_args.items() if k in AgentConfig.__annotations__}
    )

    policy_config = PolicyConfig(
        ac_dim=env_info['ac_dim'],
        ob_dim=env_info['ob_dim'],
        discrete=env_info['discrete'],
        **{k: v for k, v in agent_args.items() if k in PolicyConfig.__annotations__}
    )

    critic_config = CriticConfig(
        ob_dim=env_info['ob_dim'],
        **{k: v for k, v in agent_args.items() if k in CriticConfig.__annotations__}
    )

    updater_config = PPOUpdaterConfig(
        **{k: v for k, v in agent_args.items() if k in PPOUpdaterConfig.__annotations__}
    )

    return agent_config, policy_config, critic_config, updater_config


def show_video(path):
    video = io.open(glob.glob(path)[0], 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''
        <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        </video>
    '''.format(encoded.decode('ascii'))))



def main():
    # ====== Set-up Experiment ======
    env_name = "HalfCheetah-v5"
    project = f"{env_name}_ppo"
    name = f"{env_name}_ppo" # Append with notable hyperparameters

    # ====== Hyperparameters ======
    seed = 42
    num_envs = 4 # Set to number of threads
    batch_size = 4000
    eval_batch_size = 4000
    steps = 500
    max_episode_steps=1000

    agent_args = {
        "n_layers": 2,
        "layer_size": 64,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "use_reward_to_go": True,
        "normalize_advantages": True,
        "use_baseline": True,
        "baseline_learning_rate": 1e-3,
        "baseline_gradient_steps": 40,
        "gae_lambda": 0.90,
        "epochs": 40,
        "batch_size": 64,
        "epsilon": 0.2,
        "target_kl": 0.05,
        "entropy_coef": 0.005,
        "max_grad_norm": 4.0,
    }

    # Init environment
    set_seed(seed)
    env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_name, max_episode_steps=max_episode_steps) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    env.reset(seed=seed)
    envs.reset(seed=seed)

    # Collect environment info
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    max_ep_len = env.spec.max_episode_steps

    env_info = {
        "ob_dim": ob_dim,
        "ac_dim": ac_dim,
        "discrete": discrete,
    }

    # Init Agent
    agent = PPOAgent(*configs_from_dicts(env_info, agent_args))

    # ====== Training Configs ======
    wandb_config = {
        "env_name": env_name,
        "batch_size": batch_size,
        "steps": steps,
        "max_ep_len": max_ep_len,
        "seed": seed,
        **agent_args
    }
    train_config = TrainConfig(steps, batch_size, max_ep_len, eval_batch_size=eval_batch_size, log_wandb=True)

    # ====== Train ======
    wandb.init(project=project, name=name, config=wandb_config)
    trained_agent = PPOTrainer(agent, envs, train_config).run()
    wandb.finish()

    # ====== Video ======
    video_dir = Path("./") / name

    # Start virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Render Trajectory
    render_env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda episode_id: True)
    rendered_traj = sample_trajectory(render_env, trained_agent, max_ep_len)
    render_env.close()

    show_video((video_dir / "rl-video-episode-0.mp4").as_posix())

if __name__ == "__main__":
    main()