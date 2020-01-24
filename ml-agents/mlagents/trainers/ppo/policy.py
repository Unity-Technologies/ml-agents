import logging
import numpy as np
from typing import Any, Dict, Optional, List

from mlagents.tf_utils import tf

from mlagents_envs.timers import timed
from mlagents_envs.base_env import BatchedStepResult
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.models import EncoderType, LearningRateSchedule
from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.ppo.optimizer import PPOOptimizer
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.components.bc.module import BCModule

logger = logging.getLogger("mlagents.trainers")


class PPOPolicy(TFPolicy):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        is_training: bool,
        load: bool,
    ):
        """
        Policy for Proximal Policy Optimization Networks.
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        super().__init__(seed, brain, trainer_params)

        reward_signal_configs = trainer_params["reward_signals"]
        self.inference_dict: Dict[str, tf.Tensor] = {}
        self.update_dict: Dict[str, tf.Tensor] = {}
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.create_model(
            brain, trainer_params, reward_signal_configs, is_training, load, seed
        )

        with self.graph.as_default():
            self.bc_module: Optional[BCModule] = None
            # Create pretrainer if needed
            if "behavioral_cloning" in trainer_params:
                BCModule.check_config(trainer_params["behavioral_cloning"])
                self.bc_module = BCModule(
                    self,
                    policy_learning_rate=trainer_params["learning_rate"],
                    default_batch_size=trainer_params["batch_size"],
                    default_num_epoch=3,
                    **trainer_params["behavioral_cloning"],
                )

        if load:
            self._load_graph()
        else:
            self._initialize_graph()

    def create_model(
        self, brain, trainer_params, reward_signal_configs, is_training, load, seed
    ):
        """
        Create PPO model
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param reward_signal_configs: Reward signal config
        :param seed: Random seed.
        """
        with self.graph.as_default():
            self.model = PPOModel(
                brain=brain,
                lr=float(trainer_params["learning_rate"]),
                lr_schedule=LearningRateSchedule(
                    trainer_params.get("learning_rate_schedule", "linear")
                ),
                h_size=int(trainer_params["hidden_units"]),
                epsilon=float(trainer_params["epsilon"]),
                beta=float(trainer_params["beta"]),
                max_step=float(trainer_params["max_steps"]),
                normalize=trainer_params["normalize"],
                use_recurrent=trainer_params["use_recurrent"],
                num_layers=int(trainer_params["num_layers"]),
                m_size=self.m_size,
                seed=seed,
                stream_names=list(reward_signal_configs.keys()),
                vis_encode_type=EncoderType(
                    trainer_params.get("vis_encode_type", "simple")
                ),
            )
            self.optimizer = PPOOptimizer(
                brain=brain,
                policy=self.model,
                sess=self.sess,
                reward_signal_configs=reward_signal_configs,
                lr=float(trainer_params["learning_rate"]),
                lr_schedule=LearningRateSchedule(
                    trainer_params.get("learning_rate_schedule", "linear")
                ),
                h_size=int(trainer_params["hidden_units"]),
                epsilon=float(trainer_params["epsilon"]),
                beta=float(trainer_params["beta"]),
                max_step=float(trainer_params["max_steps"]),
                normalize=False,
                use_recurrent=trainer_params["use_recurrent"],
                num_layers=int(trainer_params["num_layers"]),
                m_size=self.m_size,
                seed=seed,
                vis_encode_type=EncoderType(
                    trainer_params.get("vis_encode_type", "simple")
                ),
            )
            self.optimizer.create_ppo_optimizer()

        self.inference_dict.update(
            {
                "action": self.model.output,
                "log_probs": self.model.all_log_probs,
                "entropy": self.model.entropy,
                "learning_rate": self.optimizer.learning_rate,
            }
        )
        if self.use_continuous_act:
            self.inference_dict["pre_action"] = self.model.output_pre
        if self.use_recurrent:
            self.inference_dict["policy_memory_out"] = self.model.memory_out
            self.inference_dict["optimizer_memory_out"] = self.optimizer.memory_out

        self.total_policy_loss = self.optimizer.abs_policy_loss
        self.update_dict.update(
            {
                "value_loss": self.optimizer.value_loss,
                "policy_loss": self.total_policy_loss,
                "update_batch": self.optimizer.update_batch,
            }
        )

    @timed
    def evaluate(
        self, batched_step_result: BatchedStepResult, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param batched_step_result: BatchedStepResult object containing inputs.
        :param global_agent_ids: The global (with worker ID) agent ids of the data in the batched_step_result.
        :return: Outputs from network as defined by self.inference_dict.
        """
        feed_dict = {
            self.model.batch_size: batched_step_result.n_agents(),
            self.model.sequence_length: 1,
        }
        epsilon = None
        if self.use_recurrent:
            if not self.use_continuous_act:
                feed_dict[self.model.prev_action] = self.retrieve_previous_action(
                    global_agent_ids
                )
            feed_dict[self.model.memory_in] = self.retrieve_memories(global_agent_ids)
        if self.use_continuous_act:
            epsilon = np.random.normal(
                size=(batched_step_result.n_agents(), self.model.act_size[0])
            )
            feed_dict[self.model.epsilon] = epsilon
        feed_dict = self.fill_eval_dict(feed_dict, batched_step_result)
        run_out = self._execute_model(feed_dict, self.inference_dict)
        return run_out
