# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Behavioral Cloning)
# Contains an implementation of Behavioral Cloning Algorithm

import logging
import numpy as np

from mlagents.envs import AllBrainInfo
from mlagents.trainers.bc.trainer import BCTrainer

logger = logging.getLogger("mlagents.trainers")


class OnlineBCTrainer(BCTrainer):
    """The OnlineBCTrainer is an implementation of Online Behavioral Cloning."""

    def __init__(self, brain, trainer_parameters, training, load, seed, run_id):
        """
        Responsible for collecting experiences and training PPO model.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The The identifier of the current run
        """
        super(OnlineBCTrainer, self).__init__(brain, trainer_parameters, training, load, seed,
                                              run_id)

        self.param_keys = ['brain_to_imitate', 'batch_size', 'time_horizon',
                           'summary_freq', 'max_steps',
                           'batches_per_epoch', 'use_recurrent',
                           'hidden_units', 'learning_rate', 'num_layers',
                           'sequence_length', 'memory_size', 'model_path']

        self.check_param_keys()
        self.brain_to_imitate = trainer_parameters['brain_to_imitate']
        self.batches_per_epoch = trainer_parameters['batches_per_epoch']
        self.n_sequences = max(int(trainer_parameters['batch_size'] / self.policy.sequence_length),
                               1)

    def __str__(self):
        return '''Hyperparameters for the Imitation Trainer of brain {0}: \n{1}'''.format(
            self.brain_name, '\n'.join(
                ['\t{0}:\t{1}'.format(x, self.trainer_parameters[x]) for x in self.param_keys]))

    def add_experiences(self, curr_info: AllBrainInfo, next_info: AllBrainInfo,
                        take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param curr_info: Current AllBrainInfo (Dictionary of all current brains and corresponding BrainInfo).
        :param next_info: Next AllBrainInfo (Dictionary of all current brains and corresponding BrainInfo).
        :param take_action_outputs: The outputs of the take action method.
        """

        # Used to collect teacher experience into training buffer
        info_teacher = curr_info[self.brain_to_imitate]
        next_info_teacher = next_info[self.brain_to_imitate]
        for agent_id in info_teacher.agents:
            self.demonstration_buffer[agent_id].last_brain_info = info_teacher

        for agent_id in next_info_teacher.agents:
            stored_info_teacher = self.demonstration_buffer[agent_id].last_brain_info
            if stored_info_teacher is None:
                continue
            else:
                idx = stored_info_teacher.agents.index(agent_id)
                next_idx = next_info_teacher.agents.index(agent_id)
                if stored_info_teacher.text_observations[idx] != "":
                    info_teacher_record, info_teacher_reset = \
                        stored_info_teacher.text_observations[idx].lower().split(",")
                    next_info_teacher_record, next_info_teacher_reset = \
                    next_info_teacher.text_observations[idx]. \
                        lower().split(",")
                    if next_info_teacher_reset == "true":
                        self.demonstration_buffer.reset_update_buffer()
                else:
                    info_teacher_record, next_info_teacher_record = "true", "true"
                if info_teacher_record == "true" and next_info_teacher_record == "true":
                    if not stored_info_teacher.local_done[idx]:
                        for i in range(self.policy.vis_obs_size):
                            self.demonstration_buffer[agent_id]['visual_obs%d' % i] \
                                .append(stored_info_teacher.visual_observations[i][idx])
                        if self.policy.use_vec_obs:
                            self.demonstration_buffer[agent_id]['vector_obs'] \
                                .append(stored_info_teacher.vector_observations[idx])
                        if self.policy.use_recurrent:
                            if stored_info_teacher.memories.shape[1] == 0:
                                stored_info_teacher.memories = np.zeros(
                                    (len(stored_info_teacher.agents),
                                     self.policy.m_size))
                            self.demonstration_buffer[agent_id]['memory'].append(
                                stored_info_teacher.memories[idx])
                        self.demonstration_buffer[agent_id]['actions'].append(
                            next_info_teacher.previous_vector_actions[next_idx])

        super(OnlineBCTrainer, self).add_experiences(curr_info, next_info, take_action_outputs)

    def process_experiences(self, current_info: AllBrainInfo, next_info: AllBrainInfo):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current AllBrainInfo
        :param next_info: Next AllBrainInfo
        """
        info_teacher = next_info[self.brain_to_imitate]
        for l in range(len(info_teacher.agents)):
            teacher_action_list = len(self.demonstration_buffer[info_teacher.agents[l]]['actions'])
            horizon_reached = teacher_action_list > self.trainer_parameters['time_horizon']
            teacher_filled = len(self.demonstration_buffer[info_teacher.agents[l]]['actions']) > 0
            if (info_teacher.local_done[l] or horizon_reached) and teacher_filled:
                agent_id = info_teacher.agents[l]
                self.demonstration_buffer.append_update_buffer(
                    agent_id, batch_size=None, training_length=self.policy.sequence_length)
                self.demonstration_buffer[agent_id].reset_agent()

        super(OnlineBCTrainer, self).process_experiences(current_info, next_info)
