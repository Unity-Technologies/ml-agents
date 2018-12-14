# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import glob
import logging
import shutil

import yaml
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from mlagents.envs.environment import UnityEnvironment
from mlagents.envs.exception import UnityEnvironmentException

from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.bc.offline_trainer import OfflineBCTrainer
from mlagents.trainers.bc.online_trainer import OnlineBCTrainer
from mlagents.trainers.meta_curriculum import MetaCurriculum
from mlagents.trainers.exception import MetaCurriculumError


class TrainerController(object):
    def __init__(self, env_path, run_id, save_freq, curriculum_folder,
                 fast_simulation, load, train, worker_id, keep_checkpoints,
                 lesson, seed, docker_target_name,
                 trainer_config_path, no_graphics):
        """
        :param env_path: Location to the environment executable to be loaded.
        :param run_id: The sub-directory name for model and summary statistics
        :param save_freq: Frequency at which to save model
        :param curriculum_folder: Folder containing JSON curriculums for the
               environment.
        :param fast_simulation: Whether to run the game at training speed.
        :param load: Whether to load the model or randomly initialize.
        :param train: Whether to train model, or only run inference.
        :param worker_id: Number to add to communication port (5005).
               Used for multi-environment
        :param keep_checkpoints: How many model checkpoints to keep.
        :param lesson: Start learning from this lesson.
        :param seed: Random seed used for training.
        :param docker_target_name: Name of docker volume that will contain all
               data.
        :param trainer_config_path: Fully qualified path to location of trainer
               configuration file.
        :param no_graphics: Whether to run the Unity simulator in no-graphics
                            mode.
        """
        if env_path is not None:
            # Strip out executable extensions if passed
            env_path = (env_path.strip()
                        .replace('.app', '')
                        .replace('.exe', '')
                        .replace('.x86_64', '')
                        .replace('.x86', ''))

        # Recognize and use docker volume if one is passed as an argument
        if not docker_target_name:
            self.docker_training = False
            self.trainer_config_path = trainer_config_path
            self.model_path = './models/{run_id}'.format(run_id=run_id)
            self.curriculum_folder = curriculum_folder
            self.summaries_dir = './summaries'
        else:
            self.docker_training = True
            self.trainer_config_path = \
                '/{docker_target_name}/{trainer_config_path}'.format(
                    docker_target_name=docker_target_name,
                    trainer_config_path = trainer_config_path)
            self.model_path = '/{docker_target_name}/models/{run_id}'.format(
                docker_target_name=docker_target_name,
                run_id=run_id)
            if env_path is not None:
                """
                Comments for future maintenance:
                    Some OS/VM instances (e.g. COS GCP Image) mount filesystems 
                    with COS flag which prevents execution of the Unity scene, 
                    to get around this, we will copy the executable into the 
                    container.
                """
                # Navigate in docker path and find env_path and copy it.
                env_path = self._prepare_for_docker_run(docker_target_name,
                                                        env_path)
            if curriculum_folder is not None:
                self.curriculum_folder = \
                    '/{docker_target_name}/{curriculum_folder}'.format(
                        docker_target_name=docker_target_name,
                        curriculum_folder=curriculum_folder)

            self.summaries_dir = '/{docker_target_name}/summaries'.format(
                docker_target_name=docker_target_name)

        self.logger = logging.getLogger('mlagents.envs')
        self.run_id = run_id
        self.save_freq = save_freq
        self.lesson = lesson
        self.fast_simulation = fast_simulation
        self.load_model = load
        self.train_model = train
        self.worker_id = worker_id
        self.keep_checkpoints = keep_checkpoints
        self.trainers = {}
        self.seed = seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.env = UnityEnvironment(file_name=env_path,
                                    worker_id=self.worker_id,
                                    seed=self.seed,
                                    docker_training=self.docker_training,
                                    no_graphics=no_graphics)
        if env_path is None:
            self.env_name = 'editor_' + self.env.academy_name
        else:
            # Extract out name of environment
            self.env_name = os.path.basename(os.path.normpath(env_path))

        if curriculum_folder is None:
            self.meta_curriculum = None
        else:
            self.meta_curriculum = MetaCurriculum(self.curriculum_folder,
                                                  self.env._resetParameters)

        if self.meta_curriculum:
            for brain_name in self.meta_curriculum.brains_to_curriculums.keys():
                if brain_name not in self.env.external_brain_names:
                    raise MetaCurriculumError('One of the curriculums '
                                              'defined in ' +
                                              self.curriculum_folder + ' '
                                              'does not have a corresponding '
                                              'Brain. Check that the '
                                              'curriculum file has the same '
                                              'name as the Brain '
                                              'whose curriculum it defines.')

    def _prepare_for_docker_run(self, docker_target_name, env_path):
        for f in glob.glob('/{docker_target_name}/*'.format(
                docker_target_name=docker_target_name)):
            if env_path in f:
                try:
                    b = os.path.basename(f)
                    if os.path.isdir(f):
                        shutil.copytree(f,
                                        '/ml-agents/{b}'.format(b=b))
                    else:
                        src_f = '/{docker_target_name}/{b}'.format(
                            docker_target_name=docker_target_name, b=b)
                        dst_f = '/ml-agents/{b}'.format(b=b)
                        shutil.copyfile(src_f, dst_f)
                        os.chmod(dst_f, 0o775)  # Make executable
                except Exception as e:
                    self.logger.info(e)
        env_path = '/ml-agents/{env_name}'.format(env_name=env_path)
        return env_path

    def _get_measure_vals(self):
        if self.meta_curriculum:
            brain_names_to_measure_vals = {}
            for brain_name, curriculum \
                in self.meta_curriculum.brains_to_curriculums.items():
                if curriculum.measure == 'progress':
                    measure_val = (self.trainers[brain_name].get_step /
                        self.trainers[brain_name].get_max_steps)
                    brain_names_to_measure_vals[brain_name] = measure_val
                elif curriculum.measure == 'reward':
                    measure_val = np.mean(self.trainers[brain_name]
                                          .reward_buffer)
                    brain_names_to_measure_vals[brain_name] = measure_val
            return brain_names_to_measure_vals
        else:
            return None

    def _save_model(self,steps=0):
        """
        Saves current model to checkpoint folder.
        :param steps: Current number of steps in training process.
        :param saver: Tensorflow saver for session.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.info('Saved Model')

    def _export_graph(self):
        """
        Exports latest saved models to .bytes format for Unity embedding.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].export_model()

    def _initialize_trainers(self, trainer_config):
        """
        Initialization of the trainers
        :param trainer_config: The configurations of the trainers
        """
        trainer_parameters_dict = {}
        for brain_name in self.env.external_brain_names:
            trainer_parameters = trainer_config['default'].copy()
            trainer_parameters['summary_path'] = '{basedir}/{name}'.format(
                basedir=self.summaries_dir,
                name=str(self.run_id) + '_' + brain_name)
            trainer_parameters['model_path'] = '{basedir}/{name}'.format(
                basedir=self.model_path,
                name=brain_name)
            trainer_parameters['keep_checkpoints'] = self.keep_checkpoints
            if brain_name in trainer_config:
                _brain_key = brain_name
                while not isinstance(trainer_config[_brain_key], dict):
                    _brain_key = trainer_config[_brain_key]
                for k in trainer_config[_brain_key]:
                    trainer_parameters[k] = trainer_config[_brain_key][k]
            trainer_parameters_dict[brain_name] = trainer_parameters.copy()
        for brain_name in self.env.external_brain_names:
            if trainer_parameters_dict[brain_name]['trainer'] == 'offline_bc':
                self.trainers[brain_name] = OfflineBCTrainer(
                    self.env.brains[brain_name],
                    trainer_parameters_dict[brain_name], self.train_model,
                    self.load_model, self.seed, self.run_id)
            elif trainer_parameters_dict[brain_name]['trainer'] == 'online_bc':
                self.trainers[brain_name] = OnlineBCTrainer(
                    self.env.brains[brain_name],
                    trainer_parameters_dict[brain_name], self.train_model,
                    self.load_model, self.seed, self.run_id)
            elif trainer_parameters_dict[brain_name]['trainer'] == 'ppo':
                self.trainers[brain_name] = PPOTrainer(
                    self.env.brains[brain_name],
                    self.meta_curriculum
                        .brains_to_curriculums[brain_name]
                        .min_lesson_length if self.meta_curriculum else 0,
                    trainer_parameters_dict[brain_name],
                    self.train_model, self.load_model, self.seed, self.run_id)
            else:
                raise UnityEnvironmentException('The trainer config contains '
                                                'an unknown trainer type for '
                                                'brain {}'
                                                .format(brain_name))

    def _load_config(self):
        try:
            with open(self.trainer_config_path) as data_file:
                trainer_config = yaml.load(data_file)
                return trainer_config
        except IOError:
            raise UnityEnvironmentException('Parameter file could not be found '
                                            'at {}.'
                                            .format(self.trainer_config_path))
        except UnicodeDecodeError:
            raise UnityEnvironmentException('There was an error decoding '
                                            'Trainer Config from this path : {}'
                                            .format(self.trainer_config_path))

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise UnityEnvironmentException('The folder {} containing the '
                                            'generated model could not be '
                                            'accessed. Please make sure the '
                                            'permissions are set correctly.'
                                            .format(model_path))

    def _reset_env(self):
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        if self.meta_curriculum is not None:
            return self.env.reset(config=self.meta_curriculum.get_config(),
                                  train_mode=self.fast_simulation)
        else:
            return self.env.reset(train_mode=self.fast_simulation)

    def start_learning(self):
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        if self.meta_curriculum is not None:
            self.meta_curriculum.set_all_curriculums_to_lesson_num(self.lesson)
        trainer_config = self._load_config()
        self._create_model_path(self.model_path)

        tf.reset_default_graph()

        # Prevent a single session from taking all GPU memory.
        self._initialize_trainers(trainer_config)
        for _, t in self.trainers.items():
            self.logger.info(t)
        global_step = 0  # This is only for saving the model
        curr_info = self._reset_env()
        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensorboard_text('Hyperparameters',
                                               trainer.parameters)
        try:
            while any([t.get_step <= t.get_max_steps \
                       for k, t in self.trainers.items()]) \
                  or not self.train_model:
                if self.meta_curriculum:
                    # Get the sizes of the reward buffers.
                    reward_buff_sizes = {k:len(t.reward_buffer) \
                                        for (k,t) in self.trainers.items()}
                    # Attempt to increment the lessons of the brains who
                    # were ready.
                    lessons_incremented = \
                        self.meta_curriculum.increment_lessons(
                            self._get_measure_vals(),
                            reward_buff_sizes=reward_buff_sizes)

                # If any lessons were incremented or the environment is
                # ready to be reset
                if (self.meta_curriculum
                        and any(lessons_incremented.values())):
                    curr_info = self._reset_env()
                    for brain_name, trainer in self.trainers.items():
                        trainer.end_episode()
                    for brain_name, changed in lessons_incremented.items():
                        if changed:
                            self.trainers[brain_name].reward_buffer.clear()
                elif self.env.global_done:
                    curr_info = self._reset_env()
                    for brain_name, trainer in self.trainers.items():
                        trainer.end_episode()

                # Decide and take an action
                take_action_vector, \
                take_action_memories, \
                take_action_text, \
                take_action_value, \
                take_action_outputs \
                    = {}, {}, {}, {}, {}
                for brain_name, trainer in self.trainers.items():
                    (take_action_vector[brain_name],
                     take_action_memories[brain_name],
                     take_action_text[brain_name],
                     take_action_value[brain_name],
                     take_action_outputs[brain_name]) = \
                        trainer.take_action(curr_info)
                new_info = self.env.step(vector_action=take_action_vector,
                                         memory=take_action_memories,
                                         text_action=take_action_text,
                                         value=take_action_value)
                for brain_name, trainer in self.trainers.items():
                    trainer.add_experiences(curr_info, new_info,
                                            take_action_outputs[brain_name])
                    trainer.process_experiences(curr_info, new_info)
                    if trainer.is_ready_update() and self.train_model \
                            and trainer.get_step <= trainer.get_max_steps:
                        # Perform gradient descent with experience buffer
                        trainer.update_policy()
                    # Write training statistics to Tensorboard.
                    if self.meta_curriculum is not None:
                        trainer.write_summary(
                            global_step,
                            lesson_num=self.meta_curriculum
                                .brains_to_curriculums[brain_name]
                                .lesson_num)
                    else:
                        trainer.write_summary(global_step)
                    if self.train_model \
                            and trainer.get_step <= trainer.get_max_steps:
                        trainer.increment_step_and_update_last_reward()
                global_step += 1
                if global_step % self.save_freq == 0 and global_step != 0 \
                        and self.train_model:
                    # Save Tensorflow model
                    self._save_model(steps=global_step)
                curr_info = new_info
            # Final save Tensorflow model
            if global_step != 0 and self.train_model:
                self._save_model(steps=global_step)
        except KeyboardInterrupt:
            print('--------------------------Now saving model--------------'
                  '-----------')
            if self.train_model:
                self.logger.info('Learning was interrupted. Please wait '
                                 'while the graph is generated.')
                self._save_model(steps=global_step)
            pass
        self.env.close()
        if self.train_model:
            self._export_graph()
