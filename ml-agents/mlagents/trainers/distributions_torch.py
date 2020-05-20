import torch
from torch import nn
from torch import distributions
import numpy as np

EPSILON = 1e-7  # Small value to avoid divide by zero


class GaussianDistribution(nn.Module):
    def __init__(self, hidden_size, num_outputs, conditional_sigma=False, **kwargs):
        super(GaussianDistribution, self).__init__(**kwargs)
        self.conditional_sigma = conditional_sigma
        self.mu = nn.Linear(hidden_size, num_outputs)
        nn.init.xavier_uniform_(self.mu.weight, gain=0.01)
        if conditional_sigma:
            self.log_sigma = nn.Linear(hidden_size, num_outputs)
            nn.init.xavier_uniform(self.log_sigma.weight, gain=0.01)
        else:
            self.log_sigma = nn.Parameter(
                torch.zeros(1, num_outputs, requires_grad=True)
            )

    def forward(self, inputs, masks):
        mu = self.mu(inputs)
        if self.conditional_sigma:
            log_sigma = self.log_sigma(inputs)
        else:
            log_sigma = self.log_sigma
        return [distributions.normal.Normal(loc=mu, scale=torch.exp(log_sigma))]


class MultiCategoricalDistribution(nn.Module):
    def __init__(self, hidden_size, act_sizes):
        super(MultiCategoricalDistribution, self).__init__()
        self.act_sizes = act_sizes
        self.branches = self.create_policy_branches(hidden_size)

    def create_policy_branches(self, hidden_size):
        branches = []
        for size in self.act_sizes:
            branch_output_layer = nn.Linear(hidden_size, size)
            nn.init.xavier_uniform_(branch_output_layer.weight, gain=0.01)
            branches.append(branch_output_layer)
        return nn.ModuleList(branches)

    def mask_branch(self, logits, mask):
        raw_probs = torch.nn.functional.softmax(logits, dim=-1) * mask
        normalized_probs = raw_probs / torch.sum(raw_probs, dim=-1).unsqueeze(-1)
        normalized_logits = torch.log(normalized_probs + EPSILON)
        return normalized_logits

    def split_masks(self, masks):
        split_masks = []
        for idx, _ in enumerate(self.act_sizes):
            start = int(np.sum(self.act_sizes[:idx]))
            end = int(np.sum(self.act_sizes[: idx + 1]))
            split_masks.append(masks[:, start:end])
        return split_masks

    def forward(self, inputs, masks):
        # Todo - Support multiple branches in mask code
        branch_distributions = []
        masks = self.split_masks(masks)
        for idx, branch in enumerate(self.branches):
            logits = branch(inputs)
            norm_logits = self.mask_branch(logits, masks[idx])
            distribution = distributions.categorical.Categorical(logits=norm_logits)
            branch_distributions.append(distribution)
        return branch_distributions
