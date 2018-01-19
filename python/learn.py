# # Unity ML Agents
# ## ML-Agent Learning (PPO)
# Launches trainers for each External Brains in a Unity Environemnt

import logging
import os
import re
import yaml

from docopt import docopt

from trainers.ghost_trainer import GhostTrainer
from trainers.ppo_models import *
from trainers.ppo_trainer import PPOTrainer
from trainers.imitation_trainer import ImitationTrainer
from unityagents import UnityEnvironment, UnityEnvironmentException


def get_progress():
    if curriculum_file is not None:
        if env.curriculum.measure_type == "progress":
            progress = 0
            for brain_name in env.external_brain_names:
                progress += trainers[brain_name].get_step / trainers[brain_name].get_max_steps
            return progress / len(env.external_brain_names)
        elif env.curriculum.measure_type == "reward":
            progress = 0
            for brain_name in env.external_brain_names:
                progress += trainers[brain_name].get_last_reward
            return progress
        else:
            return None
    else:
        return None


if __name__ == '__main__':
    logger = logging.getLogger("unityagents")
    _USAGE = '''
    Usage:
      ppo (<env>) [options] 

    Options:
      --help                     Show this message.
      --curriculum=<file>        Curriculum json file for environment [default: None].
      --slow                     Whether to run the game at training speed [default: False].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: False].
      --run-path=<path>          The sub-directory name for model and summary statistics [default: ppo]. 
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --train                    Whether to train model, or only run inference [default: False].
      --worker-id=<n>            Number to add to communication port (5005). Used for multi-environment [default: 0].
    '''

    options = docopt(_USAGE)
    logger.info(options)

    # General parameters
    model_path = './models/{}'.format(str(options['--run-path']))

    load_model = options['--load']
    train_model = options['--train']
    save_freq = int(options['--save-freq'])
    env_name = options['<env>']
    keep_checkpoints = int(options['--keep-checkpoints'])
    worker_id = int(options['--worker-id'])
    curriculum_file = str(options['--curriculum'])
    if curriculum_file == "None":
        curriculum_file = None
    lesson = int(options['--lesson'])
    fast_simulation = not bool(options['--slow'])

    env = UnityEnvironment(file_name=env_name, worker_id=worker_id, curriculum=curriculum_file)
    env.curriculum.set_lesson_number(lesson)
    logger.info(str(env))

    tf.reset_default_graph()

    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    except:
        raise UnityEnvironmentException("The folder {} containing the generated model could not be accessed."
                                        " Please make sure the permissions are set correctly.".format(model_path))

    try:
        with open("trainer_configurations.yaml") as data_file:
            trainer_configurations = yaml.load(data_file)
    except IOError:
        raise UnityEnvironmentException("The file {} could not be found. Will use default Hyperparameters"
                                        .format("trainer_configurations.yaml"))
    except UnicodeDecodeError:
        raise UnityEnvironmentException("There was an error decoding {}".format("trainer_configurations.yaml"))

    with tf.Session() as sess:
        trainers = {}
        trainer_parameters_dict = {}
        for brain_name in env.external_brain_names:
            trainer_parameters = trainer_configurations['default'].copy()
            if len(env.external_brain_names) > 1:
                graph_scope = re.sub('[^0-9a-zA-Z]+', '-', brain_name)
                trainer_parameters['graph_scope'] = graph_scope
                trainer_parameters['summary_path'] = './summaries/{}'.format(
                    str(options['--run-path'])) + '_' + graph_scope
            else:
                trainer_parameters['graph_scope'] = ''
                trainer_parameters['summary_path'] = './summaries/{}'.format(str(options['--run-path']))
            if brain_name in trainer_configurations:
                _brain_key = brain_name
                while not isinstance(trainer_configurations[_brain_key], dict):
                    _brain_key = trainer_configurations[_brain_key]
                for k in trainer_configurations[_brain_key]:
                    trainer_parameters[k] = trainer_configurations[_brain_key][k]
            trainer_parameters_dict[brain_name] = trainer_parameters.copy()
        for brain_name in env.external_brain_names:
            if 'is_ghost' not in trainer_parameters_dict[brain_name]:
                trainer_parameters_dict[brain_name]['is_ghost'] = False
            if 'is_imitation' not in trainer_parameters_dict[brain_name]:
                trainer_parameters_dict[brain_name]['is_imitation'] = False
            if trainer_parameters_dict[brain_name]['is_ghost']:
                if trainer_parameters_dict[brain_name]['brain_to_copy'] not in env.external_brain_names:
                    raise UnityEnvironmentException("The external brain {0} could not be found in the environment "
                                                    "even though the ghost trainer of brain {1} is trying to ghost it."
                                                    .format(trainer_parameters_dict[brain_name]['brain_to_copy'],
                                                            brain_name))
                trainer_parameters_dict[brain_name]['original_brain_parameters'] = trainer_parameters_dict[
                    trainer_parameters_dict[brain_name]['brain_to_copy']]
                trainers[brain_name] = GhostTrainer(sess, env, brain_name, trainer_parameters_dict[brain_name],
                                                    train_model)
            elif trainer_parameters_dict[brain_name]['is_imitation']:
                trainers[brain_name] = ImitationTrainer(sess, env, brain_name, trainer_parameters_dict[brain_name],
                                                        train_model)
            else:
                trainers[brain_name] = PPOTrainer(sess, env, brain_name, trainer_parameters_dict[brain_name],
                                                  train_model)

        for k, t in trainers.items():
            logger.info(t)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=keep_checkpoints)
        # Instantiate model parameters
        if load_model:
            logger.info('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                logger.info('The model {0} could not be found. Make sure you specified the right '
                            '--run-path'.format(model_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        global_step = 0  # This is only for saving the model
        env.curriculum.increment_lesson(get_progress())
        info = env.reset(train_mode=fast_simulation)
        if train_model:
            for brain_name, trainer in trainers.items():
                trainer.write_tensorboard_text('Hyperparameters', trainer.parameters)
        try:
            while any([t.get_step <= t.get_max_steps for k, t in trainers.items()]) or not train_model:
                if env.global_done:
                    env.curriculum.increment_lesson(get_progress())
                    info = env.reset(train_mode=fast_simulation)
                    for brain_name, trainer in trainers.items():
                        trainer.end_episode()
                # Decide and take an action
                take_action_actions, take_action_memories, take_action_values, take_action_outputs = {}, {}, {}, {}
                for brain_name, trainer in trainers.items():
                    (take_action_actions[brain_name],
                     take_action_memories[brain_name],
                     take_action_values[brain_name],
                     take_action_outputs[brain_name]) = trainer.take_action(info)
                new_info = env.step(action=take_action_actions, memory=take_action_memories, value=take_action_values)
                for brain_name, trainer in trainers.items():
                    trainer.add_experiences(info, new_info, take_action_outputs[brain_name])
                info = new_info
                for brain_name, trainer in trainers.items():
                    trainer.process_experiences(info)
                    if trainer.is_ready_update() and train_model and trainer.get_step <= trainer.get_max_steps:
                        # Perform gradient descent with experience buffer
                        trainer.update_model()
                    # Write training statistics to tensorboard.
                    trainer.write_summary(env.curriculum.lesson_number)
                    if train_model and trainer.get_step <= trainer.get_max_steps:
                        trainer.increment_step()
                        trainer.update_last_reward()
                if train_model and trainer.get_step <= trainer.get_max_steps:
                    global_step += 1
                if global_step % save_freq == 0 and global_step != 0 and train_model:
                    # Save Tensorflow model
                    save_model(sess, model_path=model_path, steps=global_step, saver=saver)

            # Final save Tensorflow model
            if global_step != 0 and train_model:
                save_model(sess, model_path=model_path, steps=global_step, saver=saver)
        except KeyboardInterrupt:
            if train_model:
                logger.info("Learning was interrupted. Please wait while the graph is generated.")
                save_model(sess, model_path=model_path, steps=global_step, saver=saver)
            pass
    env.close()
    if train_model:
        graph_name = (env_name.strip()
                      .replace('.app', '').replace('.exe', '').replace('.x86_64', '').replace('.x86', ''))
        graph_name = os.path.basename(os.path.normpath(graph_name))
        nodes = []
        scopes = []
        for brain_name in trainers.keys():
            if trainers[brain_name].graph_scope is not None:
                scope = trainers[brain_name].graph_scope + '/'
                if scope == '/':
                    scope = ''
                scopes += [scope]
                if trainers[brain_name].parameters["is_imitation"]:
                    nodes += [scope + x for x in ["action"]]
                elif not trainers[brain_name].parameters["use_recurrent"]:
                    nodes += [scope + x for x in ["action", "value_estimate", "action_probs"]]
                else:
                    nodes += [scope + x for x in ["action", "value_estimate", "action_probs", "recurrent_out"]]
        export_graph(model_path, graph_name, target_nodes=','.join(nodes))
        if len(scopes) > 1:
            logger.info("List of available scopes :")
            for scope in scopes:
                logger.info("\t" + scope)
        logger.info("List of nodes exported :")
        for n in nodes:
            logger.info("\t" + n)
