# adapt from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, base, action_dim):
        super(Policy, self).__init__()
        self.base = base
        self.dist = DiagGaussian(self.base.output_size, action_dim)

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, f_src=None, f_tgt=None, use_theta_GT_aff=False,
            deterministic=False):
        value, actor_features, theta_aff = self.base(inputs, f_src, f_tgt, use_theta_GT_aff)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, theta_aff

    def get_value(self, inputs, f_src=None, f_tgt=None, use_theta_GT_aff=False,
            deterministic=False):
        value, _, _ = self.base(inputs, f_src, f_tgt, use_theta_GT_aff)
        return value

    def evaluate_actions(self, inputs, f_src, f_tgt, action):
        value, actor_features, theta_aff = self.base(inputs, f_src, f_tgt)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, theta_aff


