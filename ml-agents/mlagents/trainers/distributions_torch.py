import torch
from torch import nn
from torch import distributions

EPSILON = 1e-6  # Small value to avoid divide by zero


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

    def forward(self, inputs):
        mu = self.mu(inputs)
        if self.conditional_sigma:
            log_sigma = self.log_sigma(inputs)
        else:
            log_sigma = self.log_sigma
        return [distributions.normal.Normal(loc=mu, scale=torch.exp(log_sigma))]


class MultiCategoricalDistribution(nn.Module):
    def __init__(self, hidden_size, act_sizes):
        super(MultiCategoricalDistribution, self).__init__()
        self.branches = self.create_policy_branches(hidden_size, act_sizes)

    def create_policy_branches(self, hidden_size, act_sizes):
        branches = []
        for size in act_sizes:
            branch_output_layer = nn.Linear(hidden_size, size)
            nn.init.xavier_uniform_(branch_output_layer.weight, gain=0.01)
            branches.append(branch_output_layer)
        return nn.ModuleList(branches)

    def mask_branch(self, logits, mask):
        raw_probs = torch.nn.functional.softmax(logits, dim=-1) * mask
        normalized_probs = raw_probs / torch.sum(raw_probs, dim=-1).unsqueeze(-1)
        normalized_logits = torch.log(normalized_probs)
        return normalized_logits

    def forward(self, inputs, masks):
        # Todo - Support multiple branches in mask code
        branch_distributions = []
        for branch in self.branches:
            logits = branch(inputs)
            norm_logits = self.mask_branch(logits, masks)
            distribution = distributions.categorical.Categorical(logits=norm_logits)
            branch_distributions.append(distribution)
        return branch_distributions
