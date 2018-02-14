# # Unity ML Agents
# ## ML-Agent Learning
# Launches unitytrainers for each External Brains in a Unity Environment

import logging
import numpy as np
import os
import re
import tensorflow as tf
import yaml

from datetime import datetime
from tensorflow.python.tools import freeze_graph
from unitytrainers.ppo.ppo_trainer import PPOTrainer
from unitytrainers.bc.bc_trainer import BehavioralCloningTrainer
from unityagents import UnityEnvironment, UnityEnvironmentException


class TrainerController(object):
    def __init__(self, env_name, run_id, save_freq, curriculum_file, fast_simulation, load, train,
                 worker_id, keep_checkpoints, lesson, seed):
        self.model_path = './models/{}'.format(run_id)
        self.logger = logging.getLogger("unityagents")
        self.run_id = run_id
        self.save_freq = save_freq
        self.curriculum_file = curriculum_file
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
        self.env = UnityEnvironment(file_name=env_name, worker_id=self.worker_id,
                                    curriculum=self.curriculum_file, seed=self.seed)
        self.env_name = (env_name.strip().replace('.app', '').replace('.exe', '').replace('.x86_64', '')
                         .replace('.x86', ''))
        self.env_name = os.path.basename(os.path.normpath(self.env_name))

    def _get_progress(self):
        if self.curriculum_file is not None:
            if self.env.curriculum.measure_type == "progress":
                progress = 0
                for brain_name in self.env.external_brain_names:
                    progress += self.trainers[brain_name].get_step / self.trainers[brain_name].get_max_steps
                return progress / len(self.env.external_brain_names)
            elif self.env.curriculum.measure_type == "reward":
                progress = 0
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
                    nodes += [scope + x for x in ["action", "value_estimate", "action_probs", "recurrent_out"]]
        if len(scopes) > 1:
            self.logger.info("List of available scopes :")
            for scope in scopes:
                self.logger.info("\t" + scope)
        self.logger.info("List of nodes to export :")
        for n in nodes:
            self.logger.info("\t" + n)
        return nodes

    def _save_model(self, sess, saver, model_path="./", steps=0):
        """
        Saves current model to checkpoint folder.
        :param sess: Current Tensorflow session.
        :param model_path: Designated model path.
        :param steps: Current number of steps in training process.
        :param saver: Tensorflow saver for session.
        """
        last_checkpoint = model_path + '/model-' + str(steps) + '.cptk'
        saver.save(sess, last_checkpoint)
        tf.train.write_graph(sess.graph_def, model_path, 'raw_graph_def.pb', as_text=False)
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
        for brain_name in self.env.external_brain_names:
            trainer_parameters = trainer_config['default'].copy()
            if len(self.env.external_brain_names) > 1:
                graph_scope = re.sub('[^0-9a-zA-Z]+', '-', brain_name)
                trainer_parameters['graph_scope'] = graph_scope
                trainer_parameters['summary_path'] = './summaries/{}'.format(
                    str(self.run_id)) + '_' + graph_scope
            else:
                trainer_parameters['graph_scope'] = ''
                trainer_parameters['summary_path'] = './summaries/{}'.format(self.run_id)
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

    def start_learning(self):
        self.env.curriculum.set_lesson_number(self.lesson)
        self.logger.info(str(self.env))

        tf.reset_default_graph()

        try:
            with open("trainer_config.yaml") as data_file:
                trainer_config = yaml.load(data_file)
        except IOError:
            raise UnityEnvironmentException("The file {} could not be found. Will use default Hyperparameters"
                                            .format("trainer_config.yaml"))
        except UnicodeDecodeError:
            raise UnityEnvironmentException("There was an error decoding {}".format("trainer_config.yaml"))

        try:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        except Exception:
            raise UnityEnvironmentException("The folder {} containing the generated model could not be accessed."
                                            " Please make sure the permissions are set correctly."
                                            .format(self.model_path))

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
            info = self.env.reset(train_mode=self.fast_simulation)
            if self.train_model:
                for brain_name, trainer in self.trainers.items():
                    trainer.write_tensorboard_text('Hyperparameters', trainer.parameters)
            try:
                while any([t.get_step <= t.get_max_steps for k, t in self.trainers.items()]) or not self.train_model:
                    if self.env.global_done:
                        self.env.curriculum.increment_lesson(self._get_progress())
                        info = self.env.reset(train_mode=self.fast_simulation)
                        for brain_name, trainer in self.trainers.items():
                            trainer.end_episode()
                    # Decide and take an action
                    take_action_actions, take_action_memories, take_action_values, take_action_outputs = {}, {}, {}, {}
                    for brain_name, trainer in self.trainers.items():
                        (take_action_actions[brain_name],
                         take_action_memories[brain_name],
                         take_action_values[brain_name],
                         take_action_outputs[brain_name]) = trainer.take_action(info)
                    new_info = self.env.step(action=take_action_actions, memory=take_action_memories,
                                             value=take_action_values)
                    for brain_name, trainer in self.trainers.items():
                        trainer.add_experiences(info, new_info, take_action_outputs[brain_name])
                    info = new_info
                    for brain_name, trainer in self.trainers.items():
                        trainer.process_experiences(info)
                        if trainer.is_ready_update() and self.train_model and trainer.get_step <= trainer.get_max_steps:
                            # Perform gradient descent with experience buffer
                            trainer.update_model()
                        # Write training statistics to tensorboard.
                        trainer.write_summary(self.env.curriculum.lesson_number)
                        if self.train_model and trainer.get_step <= trainer.get_max_steps:
                            trainer.increment_step()
                            trainer.update_last_reward()
                    if self.train_model and trainer.get_step <= trainer.get_max_steps:
                        global_step += 1
                    if global_step % self.save_freq == 0 and global_step != 0 and self.train_model:
                        # Save Tensorflow model
                        self._save_model(sess, model_path=self.model_path, steps=global_step, saver=saver)

                # Final save Tensorflow model
                if global_step != 0 and self.train_model:
                    self._save_model(sess, model_path=self.model_path, steps=global_step, saver=saver)
            except KeyboardInterrupt:
                if self.train_model:
                    self.logger.info("Learning was interrupted. Please wait while the graph is generated.")
                    self._save_model(sess, model_path=self.model_path, steps=global_step, saver=saver)
                pass
        self.env.close()
        if self.train_model:
            self._export_graph()
