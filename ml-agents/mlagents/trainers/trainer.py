# # Unity ML-Agents Toolkit
import logging
from typing import Dict, List, Deque, Any
import os
import tensorflow as tf
import numpy as np
from collections import deque, defaultdict

from mlagents.envs import UnityException, AllBrainInfo, ActionInfoOutputs, BrainInfo
from mlagents.envs.timers import set_gauge
from mlagents.trainers import TrainerMetrics
from mlagents.trainers.buffer import Buffer
from mlagents.trainers.tf_policy import Policy
from mlagents.envs import BrainParameters

LOGGER = logging.getLogger("mlagents.trainers")


class UnityTrainerException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class Trainer(object):
    """This class is the base class for the mlagents.envs.trainers"""

    def __init__(
        self,
        brain: BrainParameters,
        trainer_parameters: dict,
        training: bool,
        run_id: int,
        reward_buff_cap: int = 1,
    ):
        """
        Responsible for collecting experiences and training a neural network model.
        :BrainParameters brain: Brain to be trained.
        :dict trainer_parameters: The parameters for the trainer (dictionary).
        :bool training: Whether the trainer is set for training.
        :int run_id: The identifier of the current run
        :int reward_buff_cap:
        """
        self.param_keys: List[str] = []
        self.brain_name = brain.brain_name
        self.run_id = run_id
        self.trainer_parameters = trainer_parameters
        self.summary_path = trainer_parameters["summary_path"]
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.cumulative_returns_since_policy_update: List[float] = []
        self.is_training = training
        self.stats: Dict[str, List] = defaultdict(list)
        self.trainer_metrics = TrainerMetrics(
            path=self.summary_path + ".csv", brain_name=self.brain_name
        )
        self.summary_writer = tf.summary.FileWriter(self.summary_path)
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy: Policy = None

    def check_param_keys(self):
        for k in self.param_keys:
            if k not in self.trainer_parameters:
                raise UnityTrainerException(
                    "The hyper-parameter {0} could not be found for the {1} trainer of "
                    "brain {2}.".format(k, self.__class__, self.brain_name)
                )

    def dict_to_str(self, param_dict: Dict[str, Any], num_tabs: int) -> str:
        """
        Takes a parameter dictionary and converts it to a human-readable string.
        Recurses if there are multiple levels of dict. Used to print out hyperaparameters.
        param: param_dict: A Dictionary of key, value parameters.
        return: A string version of this dictionary.
        """
        if not isinstance(param_dict, dict):
            return str(param_dict)
        else:
            append_newline = "\n" if num_tabs > 0 else ""
            return append_newline + "\n".join(
                [
                    "\t"
                    + "  " * num_tabs
                    + "{0}:\t{1}".format(
                        x, self.dict_to_str(param_dict[x], num_tabs + 1)
                    )
                    for x in param_dict
                ]
            )

    def __str__(self) -> str:
        return """Hyperparameters for the {0} of brain {1}: \n{2}""".format(
            self.__class__.__name__,
            self.brain_name,
            self.dict_to_str(self.trainer_parameters, 0),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self) -> float:
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return float(self.trainer_parameters["max_steps"])

    @property
    def get_step(self) -> int:
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.step

    @property
    def reward_buffer(self) -> Deque[float]:
        """
        Returns the reward buffer. The reward buffer contains the cumulative
        rewards of the most recent episodes completed by agents using this
        trainer.
        :return: the reward buffer.
        """
        return self._reward_buffer

    def increment_step(self, n_steps: int) -> None:
        """
        Increment the step count of the trainer

        :param n_steps: number of steps to increment the step count by
        """
        self.step = self.policy.increment_step(n_steps)

    def add_experiences(
        self,
        curr_all_info: AllBrainInfo,
        next_all_info: AllBrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param next_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        """
        raise UnityTrainerException(
            "The process_experiences method was not implemented."
        )

    def process_experiences(
        self, current_info: AllBrainInfo, next_info: AllBrainInfo
    ) -> None:
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Dictionary of all current-step brains and corresponding BrainInfo.
        :param next_info: Dictionary of all next-step brains and corresponding BrainInfo.
        """
        raise UnityTrainerException(
            "The process_experiences method was not implemented."
        )

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        raise UnityTrainerException("The end_episode method was not implemented.")

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        raise UnityTrainerException("The is_ready_update method was not implemented.")

    def update_policy(self):
        """
        Uses demonstration_buffer to update model.
        """
        raise UnityTrainerException("The update_model method was not implemented.")

    def save_model(self) -> None:
        """
        Saves the model
        """
        self.policy.save_model(self.get_step)

    def export_model(self) -> None:
        """
        Exports the model
        """
        self.policy.export_model()

    def write_training_metrics(self) -> None:
        """
        Write training metrics to a CSV  file
        :return:
        """
        self.trainer_metrics.write_training_metrics()

    def write_summary(
        self, global_step: int, delta_train_start: float, lesson_num: int = 0
    ) -> None:
        """
        Saves training statistics to Tensorboard.
        :param delta_train_start:  Time elapsed since training started.
        :param lesson_num: Current lesson number in curriculum.
        :param global_step: The number of steps the simulation has been going for
        """
        if (
            global_step % self.trainer_parameters["summary_freq"] == 0
            and global_step != 0
        ):
            is_training = (
                "Training."
                if self.is_training and self.get_step <= self.get_max_steps
                else "Not Training."
            )
            step = min(self.get_step, self.get_max_steps)
            if len(self.stats["Environment/Cumulative Reward"]) > 0:
                mean_reward = np.mean(self.stats["Environment/Cumulative Reward"])
                LOGGER.info(
                    " {}: {}: Step: {}. "
                    "Time Elapsed: {:0.3f} s "
                    "Mean "
                    "Reward: {:0.3f}"
                    ". Std of Reward: {:0.3f}. {}".format(
                        self.run_id,
                        self.brain_name,
                        step,
                        delta_train_start,
                        mean_reward,
                        np.std(self.stats["Environment/Cumulative Reward"]),
                        is_training,
                    )
                )
                set_gauge(f"{self.brain_name}.mean_reward", mean_reward)
            else:
                LOGGER.info(
                    " {}: {}: Step: {}. No episode was completed since last summary. {}".format(
                        self.run_id, self.brain_name, step, is_training
                    )
                )
            summary = tf.Summary()
            for key in self.stats:
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    summary.value.add(tag="{}".format(key), simple_value=stat_mean)
                    self.stats[key] = []
            summary.value.add(tag="Environment/Lesson", simple_value=lesson_num)
            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

    def write_tensorboard_text(self, key: str, input_dict: Dict[str, Any]) -> None:
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            with tf.Session() as sess:
                s_op = tf.summary.text(
                    key,
                    tf.convert_to_tensor(
                        ([[str(x), str(input_dict[x])] for x in input_dict])
                    ),
                )
                s = sess.run(s_op)
                self.summary_writer.add_summary(s, self.get_step)
        except Exception:
            LOGGER.info(
                "Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above."
            )
            pass


class RLTrainer(Trainer):
    """This class is the base class for the trainers that use Reward Signals"""

    def __init__(self, *args, **kwargs):
        super(RLTrainer, self).__init__(*args, **kwargs)
        self.step = 0
        # Make sure we have at least one reward_signal
        if not self.trainer_parameters["reward_signals"]:
            raise UnityTrainerException(
                "No reward signals were defined. At least one must be used with {}.".format(
                    self.__class__.__name__
                )
            )
        # collected_rewards is a dictionary from name of reward signal to a dictionary of agent_id to cumulative reward
        # used for reporting only. We always want to report the environment reward to Tensorboard, regardless
        # of what reward signals are actually present.
        self.collected_rewards = {"environment": {}}
        self.training_buffer = Buffer()
        self.episode_steps = {}

    def construct_curr_info(self, next_info: BrainInfo) -> BrainInfo:
        """
        Constructs a BrainInfo which contains the most recent previous experiences for all agents
        which correspond to the agents in a provided next_info.
        :BrainInfo next_info: A t+1 BrainInfo.
        :return: curr_info: Reconstructed BrainInfo to match agents of next_info.
        """
        visual_observations: List[List[Any]] = [
            []
        ]  # TODO add types to brain.py methods
        vector_observations = []
        text_observations = []
        memories = []
        rewards = []
        local_dones = []
        max_reacheds = []
        agents = []
        prev_vector_actions = []
        prev_text_actions = []
        action_masks = []
        for agent_id in next_info.agents:
            agent_brain_info = self.training_buffer[agent_id].last_brain_info
            if agent_brain_info is None:
                agent_brain_info = next_info
            agent_index = agent_brain_info.agents.index(agent_id)
            for i in range(len(next_info.visual_observations)):
                visual_observations[i].append(
                    agent_brain_info.visual_observations[i][agent_index]
                )
            vector_observations.append(
                agent_brain_info.vector_observations[agent_index]
            )
            text_observations.append(agent_brain_info.text_observations[agent_index])
            if self.policy.use_recurrent:
                if len(agent_brain_info.memories) > 0:
                    memories.append(agent_brain_info.memories[agent_index])
                else:
                    memories.append(self.policy.make_empty_memory(1))
            rewards.append(agent_brain_info.rewards[agent_index])
            local_dones.append(agent_brain_info.local_done[agent_index])
            max_reacheds.append(agent_brain_info.max_reached[agent_index])
            agents.append(agent_brain_info.agents[agent_index])
            prev_vector_actions.append(
                agent_brain_info.previous_vector_actions[agent_index]
            )
            prev_text_actions.append(
                agent_brain_info.previous_text_actions[agent_index]
            )
            action_masks.append(agent_brain_info.action_masks[agent_index])
        if self.policy.use_recurrent:
            memories = np.vstack(memories)
        curr_info = BrainInfo(
            visual_observations,
            vector_observations,
            text_observations,
            memories,
            rewards,
            agents,
            local_dones,
            prev_vector_actions,
            prev_text_actions,
            max_reacheds,
            action_masks,
        )
        return curr_info

    def add_experiences(
        self,
        curr_all_info: AllBrainInfo,
        next_all_info: AllBrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param next_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        """
        self.trainer_metrics.start_experience_collection_timer()
        if take_action_outputs:
            self.stats["Policy/Entropy"].append(take_action_outputs["entropy"].mean())
            self.stats["Policy/Learning Rate"].append(
                take_action_outputs["learning_rate"]
            )
            for name, signal in self.policy.reward_signals.items():
                self.stats[signal.value_name].append(
                    np.mean(take_action_outputs["value"][name])
                )

        curr_info = curr_all_info[self.brain_name]
        next_info = next_all_info[self.brain_name]

        for agent_id in curr_info.agents:
            self.training_buffer[agent_id].last_brain_info = curr_info
            self.training_buffer[
                agent_id
            ].last_take_action_outputs = take_action_outputs

        if curr_info.agents != next_info.agents:
            curr_to_use = self.construct_curr_info(next_info)
        else:
            curr_to_use = curr_info

        tmp_rewards_dict = {}
        for name, signal in self.policy.reward_signals.items():
            tmp_rewards_dict[name] = signal.evaluate(curr_to_use, next_info)

        for agent_id in next_info.agents:
            stored_info = self.training_buffer[agent_id].last_brain_info
            stored_take_action_outputs = self.training_buffer[
                agent_id
            ].last_take_action_outputs
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    for i, _ in enumerate(stored_info.visual_observations):
                        self.training_buffer[agent_id]["visual_obs%d" % i].append(
                            stored_info.visual_observations[i][idx]
                        )
                        self.training_buffer[agent_id]["next_visual_obs%d" % i].append(
                            next_info.visual_observations[i][next_idx]
                        )
                    if self.policy.use_vec_obs:
                        self.training_buffer[agent_id]["vector_obs"].append(
                            stored_info.vector_observations[idx]
                        )
                        self.training_buffer[agent_id]["next_vector_in"].append(
                            next_info.vector_observations[next_idx]
                        )
                    if self.policy.use_recurrent:
                        if stored_info.memories.shape[1] == 0:
                            stored_info.memories = np.zeros(
                                (len(stored_info.agents), self.policy.m_size)
                            )
                        self.training_buffer[agent_id]["memory"].append(
                            stored_info.memories[idx]
                        )

                    self.training_buffer[agent_id]["masks"].append(1.0)
                    self.training_buffer[agent_id]["done"].append(
                        next_info.local_done[next_idx]
                    )
                    # Add the outputs of the last eval
                    self.add_policy_outputs(stored_take_action_outputs, agent_id, idx)
                    # Store action masks if neccessary
                    if not self.policy.use_continuous_act:
                        self.training_buffer[agent_id]["action_mask"].append(
                            stored_info.action_masks[idx], padding_value=1
                        )
                    self.training_buffer[agent_id]["prev_action"].append(
                        stored_info.previous_vector_actions[idx]
                    )
                    value = take_action_outputs["value"]
                    # Add the value outputs if needed
                    self.add_rewards_outputs(
                        value, tmp_rewards_dict, agent_id, idx, next_idx
                    )

                    for name, rewards in self.collected_rewards.items():
                        if agent_id not in rewards:
                            rewards[agent_id] = 0
                        if name == "environment":
                            # Report the reward from the environment
                            rewards[agent_id] += np.array(next_info.rewards)[next_idx]
                        else:
                            # Report the reward signals
                            rewards[agent_id] += tmp_rewards_dict[name].scaled_reward[
                                next_idx
                            ]
                if not next_info.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1
        self.trainer_metrics.end_experience_collection_timer()

    def add_policy_outputs(
        self, take_action_outputs: ActionInfoOutputs, agent_id: str, agent_idx: int
    ) -> None:
        """
        Takes the output of the last action and store it into the training buffer.
        We break this out from add_experiences since it is very highly dependent
        on the type of trainer.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        :param agent_id: the Agent we're adding to.
        :param agent_idx: the index of the Agent agent_id
        """
        raise UnityTrainerException(
            "The process_experiences method was not implemented."
        )

    def add_rewards_outputs(
        self,
        value: Dict[str, Any],
        rewards_dict: Dict[str, float],
        agent_id: str,
        agent_idx: int,
        agent_next_idx: int,
    ) -> None:
        """
        Takes the value and evaluated rewards output of the last action and store it
        into the training buffer. We break this out from add_experiences since it is very
        highly dependent on the type of trainer.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        :param rewards_dict: Dict of rewards after evaluation
        :param agent_id: the Agent we're adding to.
        :param agent_idx: the index of the Agent agent_id in the current brain info
        :param agent_next_idx: the index of the Agent agent_id in the next brain info
        """
        raise UnityTrainerException(
            "The process_experiences method was not implemented."
        )
