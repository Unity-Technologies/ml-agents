# # Unity ML Agents
# ## ML-Agent Learning
# Launches unitytrainers for each External Brains in a Unity Environment

import logging
import numpy as np
import os
import re
import tensorflow as tf
import yaml

from tensorflow.python.tools import freeze_graph
from unitytrainers.ppo.trainer import PPOTrainer
from unitytrainers.bc.trainer import BehavioralCloningTrainer
from unityagents import UnityEnvironment, UnityEnvironmentException


class TrainerController(object):
    def __init__(self, env_path, run_id, save_freq, curriculum_file, fast_simulation, load, train,
                 worker_id, keep_checkpoints, lesson, seed, docker_target_name, trainer_config_path):
        """

        :param env_path: Location to the environment executable to be loaded.
        :param run_id: The sub-directory name for model and summary statistics
        :param save_freq: Frequency at which to save model
        :param curriculum_file: Curriculum json file for environment
        :param fast_simulation: Whether to run the game at training speed
        :param load: Whether to load the model or randomly initialize
        :param train: Whether to train model, or only run inference
        :param worker_id: Number to add to communication port (5005). Used for multi-environment
        :param keep_checkpoints: How many model checkpoints to keep
        :param lesson: Start learning from this lesson
        :param seed: Random seed used for training.
        :param docker_target_name: Name of docker volume that will contain all data.
        :param trainer_config_path: Fully qualified path to location of trainer configuration file
        """
        self.trainer_config_path = trainer_config_path
        env_path = (env_path.strip()
                    .replace('.app', '')
                    .replace('.exe', '')
                    .replace('.x86_64', '')
                    .replace('.x86', ''))  # Strip out executable extensions if passed
        # Recognize and use docker volume if one is passed as an argument
        if docker_target_name == '':
            self.docker_training = False
            self.model_path = './models/{run_id}'.format(run_id=run_id)
            self.curriculum_file = curriculum_file
            self.summaries_dir = './summaries'
        else:
            self.docker_training = True
            self.model_path = '/{docker_target_name}/models/{run_id}'.format(
                docker_target_name=docker_target_name,
                run_id=run_id)
            env_path = '/{docker_target_name}/{env_name}'.format(docker_target_name=docker_target_name,
                                                                 env_name=env_path)
            if curriculum_file is None:
                self.curriculum_file = None
            else:
                self.curriculum_file = '/{docker_target_name}/{curriculum_file}'.format(
                    docker_target_name=docker_target_name,
                    curriculum_file=curriculum_file)
            self.summaries_dir = '/{docker_target_name}/summaries'.format(docker_target_name=docker_target_name)
        self.logger = logging.getLogger("unityagents")
        self.run_id = run_id
        self.save_freq = save_freq
        self.lesson = lesson
        self.fast_simulation = fast_simulation
        self.load_model = load
        self.train_model = train
        self.worker_id = worker_id
        self.keep_checkpoints = keep_checkpoints
        self.trainers = {}
        if seed == -1:
            seed = np.random.randint(0, 999999)
        self.seed = seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.env = UnityEnvironment(file_name=env_path, worker_id=self.worker_id,
                                    curriculum=self.curriculum_file, seed=self.seed,
                                    docker_training=self.docker_training)
        self.env_name = os.path.basename(os.path.normpath(env_path))  # Extract out name of environment

    def _get_progress(self):
        if self.curriculum_file is not None:
            progress = 0
            if self.env.curriculum.measure_type == "progress":
                for brain_name in self.env.external_brain_names:
                    progress += self.trainers[brain_name].get_step / self.trainers[brain_name].get_max_steps
                return progress / len(self.env.external_brain_names)
            elif self.env.curriculum.measure_type == "reward":
                for brain_name in self.env.external_brain_names:
                    progress += self.trainers[brain_name].get_last_reward
                return progress
            else:
                return None
        else:
            return None

    def _process_graph(self):
        nodes = []
        scopes = []
        for brain_name in self.trainers.keys():
            if self.trainers[brain_name].graph_scope is not None:
                scope = self.trainers[brain_name].graph_scope + '/'
                if scope == '/':
                    scope = ''
                scopes += [scope]
                if self.trainers[brain_name].parameters["trainer"] == "imitation":
                    nodes += [scope + x for x in ["action"]]
                elif not self.trainers[brain_name].parameters["use_recurrent"]:
                    nodes += [scope + x for x in ["action", "value_estimate", "action_probs"]]
                else:
                    node_list = ["action", "value_estimate", "action_probs", "recurrent_out", "memory_size"]
                    nodes += [scope + x for x in node_list]
        if len(scopes) > 1:
            self.logger.info("List of available scopes :")
            for scope in scopes:
                self.logger.info("\t" + scope)
        self.logger.info("List of nodes to export :")
        for n in nodes:
            self.logger.info("\t" + n)
        return nodes

    def _save_model(self, sess, saver, steps=0):
        """
        Saves current model to checkpoint folder.
        :param sess: Current Tensorflow session.
        :param steps: Current number of steps in training process.
        :param saver: Tensorflow saver for session.
        """
        last_checkpoint = self.model_path + '/model-' + str(steps) + '.cptk'
        saver.save(sess, last_checkpoint)
        tf.train.write_graph(sess.graph_def, self.model_path, 'raw_graph_def.pb', as_text=False)
        self.logger.info("Saved Model")

    def _export_graph(self):
        """
        Exports latest saved model to .bytes format for Unity embedding.
        """
        target_nodes = ','.join(self._process_graph())
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        freeze_graph.freeze_graph(input_graph=self.model_path + '/raw_graph_def.pb',
                                  input_binary=True,
                                  input_checkpoint=ckpt.model_checkpoint_path,
                                  output_node_names=target_nodes,
                                  output_graph=self.model_path + '/' + self.env_name + "_" + self.run_id + '.bytes',
                                  clear_devices=True, initializer_nodes="", input_saver="",
                                  restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")

    def _initialize_trainers(self, trainer_config, sess):
        trainer_parameters_dict = {}
        self.trainers = {}
        for brain_name in self.env.external_brain_names:
            trainer_parameters = trainer_config['default'].copy()
            if len(self.env.external_brain_names) > 1:
                graph_scope = re.sub('[^0-9a-zA-Z]+', '-', brain_name)
                trainer_parameters['graph_scope'] = graph_scope
                trainer_parameters['summary_path'] = '{basedir}/{name}'.format(
                    basedir=self.summaries_dir,
                    name=str(self.run_id) + '_' + graph_scope)
            else:
                trainer_parameters['graph_scope'] = ''
                trainer_parameters['summary_path'] = '{basedir}/{name}'.format(
                    basedir=self.summaries_dir,
                    name=str(self.run_id))
            if brain_name in trainer_config:
                _brain_key = brain_name
                while not isinstance(trainer_config[_brain_key], dict):
                    _brain_key = trainer_config[_brain_key]
                for k in trainer_config[_brain_key]:
                    trainer_parameters[k] = trainer_config[_brain_key][k]
            trainer_parameters_dict[brain_name] = trainer_parameters.copy()
        for brain_name in self.env.external_brain_names:
            if trainer_parameters_dict[brain_name]['trainer'] == "imitation":
                self.trainers[brain_name] = BehavioralCloningTrainer(sess, self.env, brain_name,
                                                                     trainer_parameters_dict[brain_name],
                                                                     self.train_model, self.seed)
            elif trainer_parameters_dict[brain_name]['trainer'] == "ppo":
                self.trainers[brain_name] = PPOTrainer(sess, self.env, brain_name, trainer_parameters_dict[brain_name],
                                                       self.train_model, self.seed)
            else:
                raise UnityEnvironmentException("The trainer config contains an unknown trainer type for brain {}"
                                                .format(brain_name))

    def _load_config(self):
        try:
            with open(self.trainer_config_path) as data_file:
                trainer_config = yaml.load(data_file)
                return trainer_config
        except IOError:
            raise UnityEnvironmentException("""Parameter file could not be found here {}.
                                            Will use default Hyper parameters"""
                                            .format(self.trainer_config_path))
        except UnicodeDecodeError:
            raise UnityEnvironmentException("There was an error decoding Trainer Config from this path : {}"
                                            .format(self.trainer_config_path))

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise UnityEnvironmentException("The folder {} containing the generated model could not be accessed."
                                            " Please make sure the permissions are set correctly."
                                            .format(model_path))

    def start_learning(self):
        self.env.curriculum.set_lesson_number(self.lesson)
        trainer_config = self._load_config()
        self._create_model_path(self.model_path)

        tf.reset_default_graph()

        with tf.Session() as sess:
            self._initialize_trainers(trainer_config, sess)
            for k, t in self.trainers.items():
                self.logger.info(t)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=self.keep_checkpoints)
            # Instantiate model parameters
            if self.load_model:
                self.logger.info('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                if ckpt is None:
                    self.logger.info('The model {0} could not be found. Make sure you specified the right '
                                     '--run-id'.format(self.model_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(init)
            global_step = 0  # This is only for saving the model
            self.env.curriculum.increment_lesson(self._get_progress())
            curr_info = self.env.reset(train_mode=self.fast_simulation)
            if self.train_model:
                for brain_name, trainer in self.trainers.items():
                    trainer.write_tensorboard_text('Hyperparameters', trainer.parameters)
            try:
                while any([t.get_step <= t.get_max_steps for k, t in self.trainers.items()]) or not self.train_model:
                    if self.env.global_done:
                        self.env.curriculum.increment_lesson(self._get_progress())
                        curr_info = self.env.reset(train_mode=self.fast_simulation)
                        for brain_name, trainer in self.trainers.items():
                            trainer.end_episode()
                    # Decide and take an action
                    take_action_vector, take_action_memories, take_action_text, take_action_outputs = {}, {}, {}, {}
                    for brain_name, trainer in self.trainers.items():
                        (take_action_vector[brain_name],
                         take_action_memories[brain_name],
                         take_action_text[brain_name],
                         take_action_outputs[brain_name]) = trainer.take_action(curr_info)
                    new_info = self.env.step(vector_action=take_action_vector, memory=take_action_memories,
                                             text_action=take_action_text)

                    for brain_name, trainer in self.trainers.items():
                        trainer.add_experiences(curr_info, new_info, take_action_outputs[brain_name])
                        trainer.process_experiences(curr_info, new_info)
                        if trainer.is_ready_update() and self.train_model and trainer.get_step <= trainer.get_max_steps:
                            # Perform gradient descent with experience buffer
                            trainer.update_model()
                        # Write training statistics to Tensorboard.
                        trainer.write_summary(self.env.curriculum.lesson_number)
                        if self.train_model and trainer.get_step <= trainer.get_max_steps:
                            trainer.increment_step()
                            trainer.update_last_reward()
                    if self.train_model and trainer.get_step <= trainer.get_max_steps:
                        global_step += 1
                    if global_step % self.save_freq == 0 and global_step != 0 and self.train_model:
                        # Save Tensorflow model
                        self._save_model(sess, steps=global_step, saver=saver)
                    curr_info = new_info
                # Final save Tensorflow model
                if global_step != 0 and self.train_model:
                    self._save_model(sess,  steps=global_step, saver=saver)
            except KeyboardInterrupt:
                if self.train_model:
                    self.logger.info("Learning was interrupted. Please wait while the graph is generated.")
                    self._save_model(sess, steps=global_step, saver=saver)
                pass
        self.env.close()
        if self.train_model:
            self._export_graph()
