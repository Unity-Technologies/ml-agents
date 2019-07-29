# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Behavioral Cloning)
# Contains an implementation of Behavioral Cloning Algorithm

import logging
import numpy as np

from mlagents.envs.env_manager import AgentStep
from mlagents.trainers.bc.trainer import BCTrainer

logger = logging.getLogger("mlagents.trainers")


class OnlineBCTrainer(BCTrainer):
    """The OnlineBCTrainer is an implementation of Online Behavioral Cloning."""

    def __init__(self, brain, trainer_parameters, training, load, seed, run_id):
        """
        Responsible for collecting experiences and training PPO model.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The identifier of the current run
        """
        super(OnlineBCTrainer, self).__init__(
            brain, trainer_parameters, training, load, seed, run_id
        )

        self.param_keys = [
            "brain_to_imitate",
            "batch_size",
            "time_horizon",
            "summary_freq",
            "max_steps",
            "batches_per_epoch",
            "use_recurrent",
            "hidden_units",
            "learning_rate",
            "num_layers",
            "sequence_length",
            "memory_size",
            "model_path",
        ]

        self.check_param_keys()
        self.brain_to_imitate = trainer_parameters["brain_to_imitate"]
        self.batches_per_epoch = trainer_parameters["batches_per_epoch"]
        self.n_sequences = max(
            int(trainer_parameters["batch_size"] / self.policy.sequence_length), 1
        )

    def __str__(self):
        return """Hyperparameters for the Imitation Trainer of brain {0}: \n{1}""".format(
            self.brain_name,
            "\n".join(
                [
                    "\t{0}:\t{1}".format(x, self.trainer_parameters[x])
                    for x in self.param_keys
                ]
            ),
        )

    def add_experiences(self, agent_step: AgentStep) -> None:
        """
        Adds experiences to each agent's experience history.
        :param agent_step: Agent step to be added to the training buffer.
        """
        if agent_step.current_agent_info.brain_name != self.brain_to_imitate:
            return

        # Used to collect teacher experience into training buffer
        info_teacher = agent_step.previous_agent_info
        next_info_teacher = agent_step.current_agent_info
        agent_id = info_teacher.id

        if info_teacher.text_observations != "":
            info_teacher_record, info_teacher_reset = info_teacher.text_observation.lower().split(
                ","
            )
            next_info_teacher_record, next_info_teacher_reset = next_info_teacher.text_observation.lower().split(
                ","
            )
            if next_info_teacher_reset == "true":
                self.demonstration_buffer.reset_update_buffer()
        else:
            info_teacher_record, next_info_teacher_record = "true", "true"
        if info_teacher_record == "true" and next_info_teacher_record == "true":
            if not info_teacher.local_done:
                for i in range(self.policy.vis_obs_size):
                    self.demonstration_buffer[agent_id]["visual_obs%d" % i].append(
                        info_teacher.visual_observations[i]
                    )
                if self.policy.use_vec_obs:
                    self.demonstration_buffer[agent_id]["vector_obs"].append(
                        info_teacher.vector_observations
                    )
                if self.policy.use_recurrent:
                    if info_teacher.memories.shape[1] == 0:
                        info_teacher.memories = np.zeros((1, self.policy.m_size))
                    self.demonstration_buffer[agent_id]["memory"].append(
                        info_teacher.memories
                    )
                self.demonstration_buffer[agent_id]["actions"].append(
                    next_info_teacher.previous_vector_actions
                )

        super(OnlineBCTrainer, self).add_experiences(agent_step)

    def process_experiences(self, agent_step: AgentStep) -> None:
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param agent_step: Agent step to be processed.
        """

        if agent_step.current_agent_info.brain_name != self.brain_to_imitate:
            return
        info_teacher = agent_step.current_agent_info
        teacher_action_list = len(self.demonstration_buffer[info_teacher.id]["actions"])
        horizon_reached = teacher_action_list > self.trainer_parameters["time_horizon"]
        teacher_filled = len(self.demonstration_buffer[info_teacher.id]["actions"]) > 0
        if (info_teacher.local_done or horizon_reached) and teacher_filled:
            agent_id = info_teacher.id
            self.demonstration_buffer.append_update_buffer(
                agent_id, batch_size=None, training_length=self.policy.sequence_length
            )
            self.demonstration_buffer[agent_id].reset_agent()

        super(OnlineBCTrainer, self).process_experiences(agent_step)
