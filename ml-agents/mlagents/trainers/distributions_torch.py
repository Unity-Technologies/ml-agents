import torch
from torch import nn
from torch import distributions

EPSILON = 1e-6  # Small value to avoid divide by zero


class GaussianDistribution(nn.Module):
    def __init__(self, hidden_size, num_outputs, **kwargs):
        super(GaussianDistribution, self).__init__(**kwargs)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.log_sigma_sq = nn.Linear(hidden_size, num_outputs)
        nn.init.xavier_uniform(self.mu.weight, gain=0.01)
        nn.init.xavier_uniform(self.log_sigma_sq.weight, gain=0.01)

    def forward(self, inputs):
        mu = self.mu(inputs)
        log_sig = self.log_sigma_sq(inputs)
        return distributions.normal.Normal(loc=mu, scale=torch.sqrt(torch.exp(log_sig)))


class MultiCategoricalDistribution(nn.Module):
    def __init__(self, hidden_size, act_sizes):
        super(MultiCategoricalDistribution, self).__init__()
        self.branches = self.create_policy_branches(hidden_size, act_sizes)

    def create_policy_branches(self, hidden_size, act_sizes):
        branches = []
        for size in act_sizes:
            branch_output_layer = nn.Linear(hidden_size, size)
            nn.init.xavier_uniform(branch_output_layer.weight, gain=0.01)
            branches.append(branch_output_layer)
        return branches

    def mask_branch(self, logits, mask):
        raw_probs = torch.sigmoid(logits, dim=-1) * mask
        normalized_probs = raw_probs / torch.sum(raw_probs, dim=-1)
        normalized_logits = torch.log(normalized_probs)
        return normalized_logits

    def forward(self, inputs, masks):
        branch_distributions = []
        for idx, branch in enumerate(self.branches):
            logits = branch(inputs)
            norm_logits = self.mask_branch(logits, masks[idx])
            distribution = distributions.categorical.Categorical(logits=norm_logits)
            branch_distributions.append(distribution)
        return branch_distributions
