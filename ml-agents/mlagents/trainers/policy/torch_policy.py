from typing import Any, Dict
import numpy as np
import torch
from mlagents.trainers.action_info import ActionInfo

from mlagents.trainers.policy import Policy
from mlagents_envs.base_env import DecisionSteps
from mlagents.tf_utils import tf
from mlagents_envs.timers import timed

from mlagents.trainers.policy.policy import UnityPolicyException
from mlagents.trainers.trajectory import SplitObservations
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.models_torch import ActionType, EncoderType, Actor, Critic

EPSILON = 1e-7  # Small value to avoid divide by zero


class TorchPolicy(Policy):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        load: bool,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous action spaces, as well as recurrent networks.
        :param seed: Random seed.
        :param brain: Assigned BrainParameters object.
        :param trainer_params: Defined training parameters.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        :param tanh_squash: Whether to use a tanh function on the continuous output,
        or a clipped output.
        :param reparameterize: Whether we are using the resampling trick to update the policy
        in continuous output.
        """
        super(TorchPolicy, self).__init__(brain, seed)
        self.grads = None
        num_layers = trainer_params["num_layers"]
        self.h_size = trainer_params["hidden_units"]
        self.normalize = trainer_params["normalize"]
        self.seed = seed
        self.brain = brain
        self.global_step = 0

        self.act_size = brain.vector_action_space_size
        self.sequence_length = 1
        if self.use_recurrent:
            self.m_size = trainer_params["memory_size"]
            self.sequence_length = trainer_params["sequence_length"]
            if self.m_size == 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is 0 even "
                    "though the trainer uses recurrent.".format(brain.brain_name)
                )
            elif self.m_size % 2 != 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is {1} "
                    "but it must be divisible by 2.".format(
                        brain.brain_name, self.m_size
                    )
                )

        if num_layers < 1:
            num_layers = 1
        self.num_layers = num_layers
        self.vis_encode_type = EncoderType(
            trainer_params.get("vis_encode_type", "simple")
        )
        self.tanh_squash = tanh_squash
        self.reparameterize = reparameterize
        self.condition_sigma_on_obs = condition_sigma_on_obs

        # Non-exposed parameters; these aren't exposed because they don't have a
        # good explanation and usually shouldn't be touched.
        self.log_std_min = -20
        self.log_std_max = 2

        self.inference_dict: Dict[str, tf.Tensor] = {}
        self.update_dict: Dict[str, tf.Tensor] = {}
        # TF defaults to 32-bit, so we use the same here.
        torch.set_default_tensor_type(torch.DoubleTensor)

        reward_signal_configs = trainer_params["reward_signals"]
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.actor = Actor(
            h_size=int(trainer_params["hidden_units"]),
            act_type=ActionType.CONTINUOUS,
            vector_sizes=[brain.vector_observation_space_size],
            act_size=sum(brain.vector_action_space_size),
            normalize=trainer_params["normalize"],
            num_layers=int(trainer_params["num_layers"]),
            m_size=trainer_params["memory_size"],
            use_lstm=self.use_recurrent,
            visual_sizes=brain.camera_resolutions,
            vis_encode_type=EncoderType(
                trainer_params.get("vis_encode_type", "simple")
            ),
        )

        self.critic = Critic(
            h_size=int(trainer_params["hidden_units"]),
            vector_sizes=[brain.vector_observation_space_size],
            normalize=trainer_params["normalize"],
            num_layers=int(trainer_params["num_layers"]),
            m_size=trainer_params["memory_size"],
            use_lstm=self.use_recurrent,
            visual_sizes=brain.camera_resolutions,
            stream_names=list(reward_signal_configs.keys()),
            vis_encode_type=EncoderType(
                trainer_params.get("vis_encode_type", "simple")
            ),
        )

    def split_decision_step(self, decision_requests):
        vec_vis_obs = SplitObservations.from_observations(decision_requests.obs)
        mask = None
        if not self.use_continuous_act:
            mask = np.ones(
                (len(decision_requests), np.sum(self.brain.vector_action_space_size)),
                dtype=np.float32,
            )
            if decision_requests.action_mask is not None:
                mask = 1 - np.concatenate(decision_requests.action_mask, axis=1)
        return vec_vis_obs.vector_observations, vec_vis_obs.visual_observations, mask

    def update_normalization(self, vector_obs: np.ndarray) -> None:
        """
        If this policy normalizes vector observations, this will update the norm values in the graph.
        :param vector_obs: The vector observations to add to the running estimate of the distribution.
        """
        if self.use_vec_obs and self.normalize:
            self.critic.network_body.normalize(vector_obs)
            self.actor.network_body.normalize(vector_obs)

    def execute_model(self, vec_obs, vis_obs, masks=None):
        action_dists = self.actor(vec_obs, vis_obs, masks)
        actions = []
        log_probs = []
        entropies = []
        for action_dist in action_dists:
            action = action_dist.sample()
            actions.append(action)
            log_probs.append(action_dist.log_prob(action))
            entropies.append(action_dist.entropy())
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        value_heads, mean_value = self.critic(vec_obs, vis_obs)
        return actions, log_probs, entropies, value_heads

    @timed
    def evaluate(self, decision_requests: DecisionSteps) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param decision_requests: DecisionStep object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        vec_obs, vis_obs, masks = self.split_decision_step(decision_requests)
        vec_obs = [vec_obs]  # For consistency with visual observations
        run_out = {}
        action, log_probs, entropy, value_heads = self.execute_model(
            vec_obs, vis_obs, masks
        )
        run_out["action"] = np.array(action.detach())
        run_out["log_probs"] = np.array(log_probs.detach())
        run_out["entropy"] = np.array(entropy.detach())
        run_out["value_heads"] = {
            name: np.array(t.detach()) for name, t in value_heads.items()
        }
        run_out["value"] = np.mean(list(run_out["value_heads"].values()), 0)
        run_out["learning_rate"] = 0.0
        self.actor.network_body.update_normalization(vec_obs)
        self.critic.network_body.update_normalization(vec_obs)
        return run_out

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param worker_id:
        :param decision_requests: A dictionary of brain names and BrainInfo from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(decision_requests) == 0:
            return ActionInfo.empty()
        run_out = self.evaluate(
            decision_requests
        )  # pylint: disable=assignment-from-no-return
        return ActionInfo(
            action=run_out.get("action"),
            value=run_out.get("value"),
            outputs=run_out,
            agent_ids=list(decision_requests.agent_id),
        )

    @property
    def vis_obs_size(self):
        return self.brain.number_visual_observations

    @property
    def vec_obs_size(self):
        return self.brain.vector_observation_space_size

    @property
    def use_vis_obs(self):
        return self.vis_obs_size > 0

    @property
    def use_vec_obs(self):
        return self.vec_obs_size > 0

    @property
    def use_recurrent(self):
        return False

    @property
    def use_continuous_act(self):
        return True

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        step = self.global_step
        return step

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        self.global_step += n_steps
        return self.get_current_step()

    def save_model(self, step):
        pass

    def export_model(self):
        pass
