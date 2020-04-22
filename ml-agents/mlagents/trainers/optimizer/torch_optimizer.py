from typing import Dict, Any, Optional, Tuple, List
import torch
import numpy as np
from mlagents_envs.base_env import DecisionSteps

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.components.bc.module import BCModule
from mlagents.trainers.components.reward_signals.extrinsic.signal import (
    ExtrinsicRewardSignal,
)
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.trajectory import SplitObservations


class TorchOptimizer(Optimizer):  # pylint: disable=W0223
    def __init__(self, policy: TorchPolicy, trainer_params: Dict[str, Any]):
        super(TorchOptimizer, self).__init__()
        self.policy = policy
        self.trainer_params = trainer_params
        self.update_dict: Dict[str, torch.Tensor] = {}
        self.value_heads: Dict[str, torch.Tensor] = {}
        self.memory_in: torch.Tensor = None
        self.memory_out: torch.Tensor = None
        self.m_size: int = 0
        self.global_step = torch.tensor(0)
        self.bc_module: Optional[BCModule] = None
        self.create_reward_signals(trainer_params["reward_signals"])

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        pass

    def create_reward_signals(self, reward_signal_configs):
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        extrinsic_signal = ExtrinsicRewardSignal(
            self.policy, **reward_signal_configs["extrinsic"]
        )
        self.reward_signals = {"extrinsic": extrinsic_signal}
        # Create reward signals
        # for reward_signal, config in reward_signal_configs.items():
        #    self.reward_signals[reward_signal] = create_reward_signal(
        #        self.policy, reward_signal, config
        #    )
        #    self.update_dict.update(self.reward_signals[reward_signal].update_dict)

    def get_value_estimates(
        self, decision_requests: DecisionSteps, idx: int, done: bool
    ) -> Dict[str, float]:
        """
        Generates value estimates for bootstrapping.
        :param decision_requests:
        :param idx: Index in BrainInfo of agent.
        :param done: Whether or not this is the last element of the episode,
        in which case the value estimate will be 0.
        :return: The value estimate dictionary with key being the name of the reward signal
        and the value the corresponding value estimate.
        """
        vec_vis_obs = SplitObservations.from_observations(decision_requests.obs)

        value_estimates, mean_value = self.policy.critic(
            np.expand_dims(vec_vis_obs.vector_observations[idx], 0),
            np.expand_dims(vec_vis_obs.visual_observations[idx], 0),
        )

        value_estimates = {k: float(v) for k, v in value_estimates.items()}

        # If we're done, reassign all of the value estimates that need terminal states.
        if done:
            for k in value_estimates:
                if self.reward_signals[k].use_terminal_states:
                    value_estimates[k] = 0.0

        return value_estimates

    def get_trajectory_value_estimates(
        self, batch: AgentBuffer, next_obs: List[np.ndarray], done: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        vector_obs = torch.Tensor(np.array(batch["vector_obs"]))
        visual_obs = batch["visual_obs"]

        value_estimates, mean_value = self.policy.critic(vector_obs, visual_obs)

        for name, estimate in value_estimates.items():
            value_estimates[name] = estimate.detach()

        return value_estimates, value_estimates
