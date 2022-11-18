from typing import Dict
import numpy as np
from mlagents.torch_utils import torch

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.settings import BehavioralCloningSettings, ScheduleType
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.buffer import AgentBuffer


class BCModule:
    def __init__(
        self,
        policy: TorchPolicy,
        settings: BehavioralCloningSettings,
        policy_learning_rate: float,
        default_batch_size: int,
        default_num_epoch: int,
    ):
        """
        A BC trainer that can be used inline with RL.
        :param policy: The policy of the learning model
        :param settings: The settings for BehavioralCloning including LR strength, batch_size,
        num_epochs, samples_per_update and LR annealing steps.
        :param policy_learning_rate: The initial Learning Rate of the policy. Used to set an appropriate learning rate
            for the pretrainer.
        """
        self.policy = policy
        self._anneal_steps = settings.steps
        self.current_lr = policy_learning_rate * settings.strength

        learning_rate_schedule: ScheduleType = (
            ScheduleType.LINEAR if self._anneal_steps > 0 else ScheduleType.CONSTANT
        )
        self.decay_learning_rate = ModelUtils.DecayedValue(
            learning_rate_schedule, self.current_lr, 1e-10, self._anneal_steps
        )
        params = self.policy.actor.parameters()
        self.optimizer = torch.optim.Adam(params, lr=self.current_lr)
        _, self.demonstration_buffer = demo_to_buffer(
            settings.demo_path, policy.sequence_length, policy.behavior_spec
        )
        self.batch_size = (
            settings.batch_size if settings.batch_size else default_batch_size
        )
        self.num_epoch = settings.num_epoch if settings.num_epoch else default_num_epoch
        self.n_sequences = max(
            min(self.batch_size, self.demonstration_buffer.num_experiences)
            // policy.sequence_length,
            1,
        )

        self.has_updated = False
        self.use_recurrent = self.policy.use_recurrent
        self.samples_per_update = settings.samples_per_update

    def update(self) -> Dict[str, np.ndarray]:
        """
        Updates model using buffer.
        :param max_batches: The maximum number of batches to use per update.
        :return: The loss of the update.
        """
        # Don't continue training if the learning rate has reached 0, to reduce training time.

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        if self.current_lr <= 1e-10:  # Unlike in TF, this never actually reaches 0.
            return {"Losses/Pretraining Loss": 0}

        batch_losses = []
        possible_demo_batches = (
            self.demonstration_buffer.num_experiences // self.n_sequences
        )
        possible_batches = possible_demo_batches

        max_batches = self.samples_per_update // self.n_sequences

        n_epoch = self.num_epoch
        for _ in range(n_epoch):
            self.demonstration_buffer.shuffle(
                sequence_length=self.policy.sequence_length
            )
            if max_batches == 0:
                num_batches = possible_batches
            else:
                num_batches = min(possible_batches, max_batches)
            for i in range(num_batches // self.policy.sequence_length):
                demo_update_buffer = self.demonstration_buffer
                start = i * self.n_sequences * self.policy.sequence_length
                end = (i + 1) * self.n_sequences * self.policy.sequence_length
                mini_batch_demo = demo_update_buffer.make_mini_batch(start, end)
                run_out = self._update_batch(mini_batch_demo, self.n_sequences)
                loss = run_out["loss"]
                batch_losses.append(loss)

        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.current_lr = decay_lr

        self.has_updated = True
        update_stats = {"Losses/Pretraining Loss": np.mean(batch_losses)}
        return update_stats

    def _behavioral_cloning_loss(
        self,
        selected_actions: AgentAction,
        log_probs: ActionLogProbs,
        expert_actions: torch.Tensor,
    ) -> torch.Tensor:
        bc_loss = 0
        if self.policy.behavior_spec.action_spec.continuous_size > 0:
            bc_loss += torch.nn.functional.mse_loss(
                selected_actions.continuous_tensor, expert_actions.continuous_tensor
            )
        if self.policy.behavior_spec.action_spec.discrete_size > 0:
            one_hot_expert_actions = ModelUtils.actions_to_onehot(
                expert_actions.discrete_tensor,
                self.policy.behavior_spec.action_spec.discrete_branches,
            )
            log_prob_branches = ModelUtils.break_into_branches(
                log_probs.all_discrete_tensor,
                self.policy.behavior_spec.action_spec.discrete_branches,
            )
            bc_loss += torch.mean(
                torch.stack(
                    [
                        torch.sum(
                            -torch.nn.functional.log_softmax(log_prob_branch, dim=1)
                            * expert_actions_branch,
                            dim=1,
                        )
                        for log_prob_branch, expert_actions_branch in zip(
                            log_prob_branches, one_hot_expert_actions
                        )
                    ]
                )
            )
        return bc_loss

    def _update_batch(
        self, mini_batch_demo: AgentBuffer, n_sequences: int
    ) -> Dict[str, float]:
        """
        Helper function for update_batch.
        """
        np_obs = ObsUtil.from_buffer(
            mini_batch_demo, len(self.policy.behavior_spec.observation_specs)
        )
        # Convert to tensors
        tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]
        act_masks = None
        expert_actions = AgentAction.from_buffer(mini_batch_demo)
        if self.policy.behavior_spec.action_spec.discrete_size > 0:

            act_masks = ModelUtils.list_to_tensor(
                np.ones(
                    (
                        self.n_sequences * self.policy.sequence_length,
                        sum(self.policy.behavior_spec.action_spec.discrete_branches),
                    ),
                    dtype=np.float32,
                )
            )

        memories = []
        if self.policy.use_recurrent:
            memories = torch.zeros(1, self.n_sequences, self.policy.m_size)

        selected_actions, run_out, _ = self.policy.actor.get_action_and_stats(
            tensor_obs,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        log_probs = run_out["log_probs"]
        bc_loss = self._behavioral_cloning_loss(
            selected_actions, log_probs, expert_actions
        )
        self.optimizer.zero_grad()
        bc_loss.backward()

        self.optimizer.step()
        run_out = {"loss": bc_loss.item()}
        return run_out
