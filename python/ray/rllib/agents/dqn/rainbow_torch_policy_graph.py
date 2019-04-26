from __future__ import absolute_import

import os

from gym.spaces import Dict
import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib import SampleBatch
from torch.optim import RMSprop
import numpy as np
from threading import Lock

from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.qmix.model import _get_size
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.torch_policy_graph import TorchPolicyGraph
from ray.rllib.utils.annotations import override
from ray.rllib.models.catalog import ModelCatalog

# Importance sampling weights for prioritized replay
PRIO_WEIGHTS = "weights"


class NoisyLinear(nn.Module):
    """
        todo: add reference to Kaixhin's Rainbow etc.
    """

    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', th.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(th.empty(out_features))
        self.bias_sigma = nn.Parameter(th.empty(out_features))
        self.register_buffer('bias_epsilon', th.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = th.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input_):
        if self.training:
            return F.linear(input_,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input_, self.weight_mu, self.bias_mu)


class RainbowModel(nn.Module):
    """
    todo: add reference to Kaixhin's Rainbow etc.
    """

    def __init__(self, config, observation_space, action_space):
        super().__init__()
        self.atoms = config['num_atoms']
        self.num_actions = action_space.n

        self.conv1 = nn.Conv2d(observation_space.shape[2], 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc_h_v = NoisyLinear(3136, config['hiddens'][0])
        self.fc_h_a = NoisyLinear(3136, config['hiddens'][0])
        self.fc_z_v = NoisyLinear(config['hiddens'][0], self.atoms)
        self.fc_z_a = NoisyLinear(config['hiddens'][0], self.num_actions * self.atoms)

    def forward(self, x, log=False):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.num_actions, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


def _adjust_nstep(n_step, gamma, obs, actions, rewards, new_obs, dones):
    """Rewrites the given trajectory fragments to encode n-step rewards.

    reward[i] = (
        reward[i] * gamma**0 +
        reward[i+1] * gamma**1 +
        ... +
        reward[i+n_step-1] * gamma**(n_step-1))

    The ith new_obs is also adjusted to point to the (i+n_step-1)'th new obs.

    At the end of the trajectory, n is truncated to fit in the traj length.
    """

    assert not any(dones[:-1]), "Unexpected done in middle of trajectory"

    traj_length = len(rewards)
    for i in range(traj_length):
        for j in range(1, n_step):
            if i + j < traj_length:
                new_obs[i] = new_obs[i + j]
                dones[i] = dones[i + j]
                rewards[i] += gamma ** j * rewards[i + j]


def _postprocess_dqn(policy_graph, batch):
    # N-step Q adjustments
    if policy_graph.config["n_step"] > 1:
        _adjust_nstep(policy_graph.config["n_step"],
                      policy_graph.config["gamma"], batch[SampleBatch.CUR_OBS],
                      batch[SampleBatch.ACTIONS], batch[SampleBatch.REWARDS],
                      batch[SampleBatch.NEXT_OBS], batch[SampleBatch.DONES])

    if PRIO_WEIGHTS not in batch:
        batch[PRIO_WEIGHTS] = np.ones_like(batch[SampleBatch.REWARDS])

    # Prioritize on the worker side
    if batch.count > 0 and policy_graph.config["worker_side_prioritization"]:
        td_errors = policy_graph.compute_td_error(
            batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS],
            batch[SampleBatch.REWARDS], batch[SampleBatch.NEXT_OBS],
            np.invert(batch[SampleBatch.DONES])).detach().cpu().numpy()
        new_priorities = (
                np.abs(td_errors) + policy_graph.config["prioritized_replay_eps"])
        batch.data[PRIO_WEIGHTS] = new_priorities

    return batch


class DQNPostProcessing(object):
    """Implements n-step learning and param noise adjustments."""

    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        return _postprocess_dqn(self, sample_batch)


class RainbowTorchPolicyGraph(DQNPostProcessing, PolicyGraph):
    def __init__(self, observation_space, action_space, config):
        _validate(config)
        config = dict(DQN_DEFAULT_CONFIG, **config)
        self.config = config
        self.lock = Lock()
        self.observation_space = observation_space
        self.action_space = action_space
        self.atoms = config['num_atoms']
        self.Vmin = config['v_min']
        self.Vmax = config['v_max']
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms)  # Support (range) of z
        self.delta_z = (self.Vmax - self.Vmin) / (self.atoms - 1)
        self.batch_size = config['sample_batch_size']
        self.n = config['n_step']
        self.discount = config['gamma']
        self.priority_exponent = config['prioritized_replay_alpha']
        self.max_gradient_norm = config['grad_norm_clipping']

        self.online_net = RainbowModel(config, observation_space, self.action_space)
        self.online_net.train()

        self.target_net = RainbowModel(config, observation_space, self.action_space)
        self.update_target()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.device = (torch.device("cuda")
                       if bool(os.environ.get("CUDA_VISIBLE_DEVICES", None))
                       else torch.device("cpu"))
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.support = self.support.to(self.device)
        # Setup optimiser
        self.optimiser = torch.optim.Adam(self.online_net.parameters(), lr=config['lr'], eps=config['adam_epsilon'])

    def compute_td_error(self, states, actions, returns, next_states, nonterminals):
        self.batch_size = actions.shape[0]
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        returns = torch.from_numpy(np.array(returns)).float().to(self.device)
        nonterminals = torch.from_numpy(np.array(nonterminals).astype('float32')).to(self.device).unsqueeze(1)
        actions = torch.from_numpy(np.array(actions)).to(self.device)
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(
                1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(
                self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(
                0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        return loss

    @override(PolicyGraph)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        with torch.no_grad():
            return [(self.online_net(
                torch.from_numpy(np.array(obs_batch)).float().to(self.device)).data * self.support).cpu().sum(2).max(1)[1][
                        0]], [], {}

    @override(PolicyGraph)
    def learn_on_batch(self, samples):
        self.optimiser.zero_grad()
        loss = self.compute_td_error(samples[SampleBatch.CUR_OBS], samples[SampleBatch.ACTIONS],
                                     samples[SampleBatch.REWARDS], samples[SampleBatch.NEXT_OBS],
                                     np.invert(samples[SampleBatch.DONES]))
        weights = torch.from_numpy(samples[PRIO_WEIGHTS]).float().to(self.device)
        (weights * loss).mean().backward()
        nn.utils.clip_grad_norm(self.online_net.parameters(), self.max_gradient_norm)
        self.optimiser.step()
        return {'td_error': loss.detach()}

    @override(PolicyGraph)
    def get_weights(self):
        with self.lock:
            return {k: v.cpu() for k, v in self.online_net.state_dict().items()}

    @override(PolicyGraph)
    def set_weights(self, weights):
        with self.lock:
            self.online_net.load_state_dict(weights)

    @override(PolicyGraph)
    def get_initial_state(self):
        return [s.numpy() for s in self.online_net.state_init()]

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon


def _validate(config):
    pass

