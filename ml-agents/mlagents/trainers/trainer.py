# # Unity ML-Agents Toolkit
import logging

import tensorflow as tf
import numpy as np

from mlagents.envs import UnityException, AllBrainInfo

logger = logging.getLogger("mlagents.trainers")


class UnityTrainerException(UnityException):
    """
    Related to errors with the Trainer.
    """
    pass


class Trainer(object):
    """This class is the base class for the mlagents.trainers"""

    def __init__(self, brain, trainer_parameters, training, run_id):
        """
        Responsible for collecting experiences and training a neural network model.
        :BrainParameters brain: Brain to be trained.
        :dict trainer_parameters: The parameters for the trainer (dictionary).
        :bool training: Whether the trainer is set for training.
        :int run_id: The identifier of the current run
        """
        self.param_keys = []
        self.brain_name = brain.brain_name
        self.run_id = run_id
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.stats = {}
        self.summary_writer = None
        self.policy = None

    def __str__(self):
        return '''{} Trainer'''.format(self.__class__)

    def check_param_keys(self):
        for k in self.param_keys:
            if k not in self.trainer_parameters:
                raise UnityTrainerException(
                    "The hyper-parameter {0} could not be found for the {1} trainer of "
                    "brain {2}.".format(k, self.__class__, self.brain_name))

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        raise UnityTrainerException("The parameters property was not implemented.")

    @property
    def graph_scope(self):
        """
        Returns the graph scope of the trainer.
        """
        raise UnityTrainerException("The graph_scope property was not implemented.")

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        raise UnityTrainerException("The get_max_steps property was not implemented.")

    @property
    def get_step(self):
        """
        Returns the number of training steps the trainer has performed
        :return: the step count of the trainer
        """
        raise UnityTrainerException("The get_step property was not implemented.")

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        raise UnityTrainerException("The get_last_reward property was not implemented.")

    def increment_step_and_update_last_reward(self):
        """
        Increment the step count of the trainer and updates the last reward
        """
        raise UnityTrainerException(
            "The increment_step_and_update_last_reward method was not implemented.")

    def take_action(self, all_brain_info: AllBrainInfo):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param all_brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: a tuple containing action, memories, values and an object
        to be passed to add experiences
        """
        raise UnityTrainerException("The take_action method was not implemented.")

    def add_experiences(self, curr_info: AllBrainInfo, next_info: AllBrainInfo,
                        take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param curr_info: Current AllBrainInfo.
        :param next_info: Next AllBrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """
        raise UnityTrainerException("The add_experiences method was not implemented.")

    def process_experiences(self, current_info: AllBrainInfo, next_info: AllBrainInfo):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Dictionary of all current-step brains and corresponding BrainInfo.
        :param next_info: Dictionary of all next-step brains and corresponding BrainInfo.
        """
        raise UnityTrainerException("The process_experiences method was not implemented.")

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

    def save_model(self):
        """
        Saves the model
        """
        self.policy.save_model(self.get_step)

    def export_model(self):
        """
        Exports the model
        """
        self.policy.export_model()

    def write_summary(self, global_step, lesson_num=0):
        """
        Saves training statistics to Tensorboard.
        :param lesson_num: Current lesson number in curriculum.
        :param global_step: The number of steps the simulation has been going for
        """
        if global_step % self.trainer_parameters['summary_freq'] == 0 and global_step != 0:
            is_training = "Training." if self.is_training and self.get_step <= self.get_max_steps else "Not Training."
            if len(self.stats['Environment/Cumulative Reward']) > 0:
                mean_reward = np.mean(self.stats['Environment/Cumulative Reward'])
                logger.info(" {}: {}: Step: {}. Mean Reward: {:0.3f}. Std of Reward: {:0.3f}. {}"
                            .format(self.run_id, self.brain_name,
                                    min(self.get_step, self.get_max_steps),
                                    mean_reward, np.std(self.stats['Environment/Cumulative Reward']),
                                    is_training))
            else:
                logger.info(" {}: {}: Step: {}. No episode was completed since last summary. {}"
                            .format(self.run_id, self.brain_name, self.get_step, is_training))
            summary = tf.Summary()
            for key in self.stats:
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    summary.value.add(tag='{}'.format(key), simple_value=stat_mean)
                    self.stats[key] = []
            summary.value.add(tag='Environment/Lesson', simple_value=lesson_num)
            self.summary_writer.add_summary(summary, self.get_step)
            self.summary_writer.flush()

    def write_tensorboard_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            with tf.Session() as sess:
                s_op = tf.summary.text(key, tf.convert_to_tensor(
                    ([[str(x), str(input_dict[x])] for x in input_dict])))
                s = sess.run(s_op)
                self.summary_writer.add_summary(s, self.get_step)
        except:
            logger.info(
                "Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")
            pass
