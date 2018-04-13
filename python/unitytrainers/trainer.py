# # Unity ML Agents
import logging

import tensorflow as tf

from unityagents import UnityException, AllBrainInfo

logger = logging.getLogger("unityagents")


class UnityTrainerException(UnityException):
    """
    Related to errors with the Trainer.
    """
    pass


class Trainer(object):
    """This class is the abstract class for the unitytrainers"""

    def __init__(self, sess, env, brain_name, trainer_parameters, training):
        """
        Responsible for collecting experiences and training a neural network model.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        self.brain_name = brain_name
        self.brain = env.brains[self.brain_name]
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.sess = sess

    def __str__(self):
        return '''Empty Trainer'''

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
        Returns the number of steps the trainer has performed
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

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        raise UnityTrainerException("The increment_step method was not implemented.")

    def update_last_reward(self):
        """
        Updates the last reward
        """
        raise UnityTrainerException("The update_last_reward method was not implemented.")

    def take_action(self, all_brain_info: AllBrainInfo):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param all_brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: a tuple containing action, memories, values and an object
        to be passed to add experiences
        """
        raise UnityTrainerException("The take_action method was not implemented.")

    def add_experiences(self, curr_info: AllBrainInfo, next_info: AllBrainInfo, take_action_outputs):
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

    def update_model(self):
        """
        Uses training_buffer to update model.
        """
        raise UnityTrainerException("The update_model method was not implemented.")

    def write_summary(self, lesson_number):
        """
        Saves training statistics to Tensorboard.
        :param lesson_number: The lesson the trainer is at.
        """
        raise UnityTrainerException("The write_summary method was not implemented.")

    def write_tensorboard_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            s_op = tf.summary.text(key,
                                   tf.convert_to_tensor(([[str(x), str(input_dict[x])] for x in input_dict]))
                                   )
            s = self.sess.run(s_op)
            self.summary_writer.add_summary(s, self.get_step)
        except:
            logger.info("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")
            pass
