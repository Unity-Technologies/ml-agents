from typing import Dict, Any, List

import numpy as np

from mlagents_envs.base_env import DecisionSteps
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.policy import Policy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import GlobalSteps
from mlagents.torch_utils import default_device
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.behavior_id_utils import get_global_agent_id
from mlagents_envs.timers import timed
from mlagents.torch_utils import torch
from mlagents.trainers.torch_entities.utils import ModelUtils

EPSILON = 1e-7


class ASEPolicy(Policy):
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        network_settings: NetworkSettings,
        actor_cls: type,
        actor_kwargs: Dict[str, Any],
        disc_enc_cls: type,
        disc_enc_kwargs: Dict[str, Any],
    ):
        super().__init__(seed, behavior_spec, network_settings)
        self.global_step = GlobalSteps()

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
            "Losses/Discriminator Loss": "discriminator_loss",
            "Losses/Encoder Loss": "encoder_loss",
        }

        self.actor = actor_cls(
            observation_specs=self.behavior_spec.observation_specs,
            network_settings=network_settings,
            action_spec=behavior_spec.action_spec,
            **actor_kwargs,
        )

        self.discriminator_encoder = disc_enc_cls(
            observation_specs=self.behavior_spec.observation_specs,
            network_settings=network_settings,
            **disc_enc_kwargs,
        )

        # Save the m_size needed for export
        self._export_m_size = self.m_size
        # m_size needed for training is determined by network, not trainer settings
        self.m_size = self.actor.memory_size

        self.actor.to(default_device())
        self.discriminator_encoder.to(default_device())

    def _extract_masks(self, decision_requests: DecisionSteps) -> np.ndarray:
        mask = None
        if self.behavior_spec.action_spec.discrete_size > 0:
            num_discrete_flat = np.sum(self.behavior_spec.action_spec.discrete_branches)
            mask = torch.ones([len(decision_requests), num_discrete_flat])
            if decision_requests.action_mask is not None:
                mask = torch.as_tensor(
                    1 - np.concatenate(decision_requests.action_mask, axis=1)
                )
        return mask

    @timed
    def evaluate(
        self, decision_requests: DecisionSteps, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param global_agent_ids:
        :param decision_requests: DecisionStep object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        obs = decision_requests.obs
        masks = self._extract_masks(decision_requests)
        tensor_obs = [torch.as_tensor(np_ob) for np_ob in obs]

        memories = torch.as_tensor(self.retrieve_memories(global_agent_ids)).unsqueeze(
            0
        )
        with torch.no_grad():
            action, run_out, memories = self.actor.get_action_and_stats(
                tensor_obs, masks=masks, memories=memories
            )
        run_out["action"] = action.to_action_tuple()
        if "log_probs" in run_out:
            run_out["log_probs"] = run_out["log_probs"].to_log_probs_tuple()
        if "entropy" in run_out:
            run_out["entropy"] = ModelUtils.to_numpy(run_out["entropy"])
        if self.use_recurrent:
            run_out["memory_out"] = ModelUtils.to_numpy(memories).squeeze(0)
        return run_out

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        if len(decision_requests) == 0:
            return ActionInfo.empty()

        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]  # For 1-D array, the iterator order is correct.

        run_out = self.evaluate(decision_requests, global_agent_ids)
        self.save_memories(global_agent_ids, run_out.get("memory_out"))
        self.check_nan_action(run_out.get("action"))
        return ActionInfo(
            action=run_out.get("action"),
            env_action=run_out.get("env_action"),
            outputs=run_out,
            agent_ids=list(decision_requests.agent_id),
        )

    def increment_step(self, n_steps):
        pass

    def get_current_step(self):
        return self.global_step.current_step

    def load_weights(self, values: List[np.ndarray]) -> None:
        pass

    def get_weights(self) -> List[np.ndarray]:
        pass

    def init_load_weights(self) -> None:
        pass

    def get_modules(self):
        return {"Policy": self.actor, "global_step": self.global_step}
