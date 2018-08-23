# # Unity ML-Agents Toolkit

import logging

import os
import multiprocessing
import numpy as np
from docopt import docopt

from unitytrainers.trainer_controller import TrainerController
from unitytrainers.exception import TrainerError


def run_training(sub_id, run_seed, run_options):
    """
    Launches training session.
    :param sub_id: Unique id for training session.
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    # Docker Parameters
    if run_options['--docker-target-name'] == 'Empty':
        docker_target_name = ''
    else:
        docker_target_name = run_options['--docker-target-name']

    # General parameters
    env_path = run_options['<env>']
    run_id = run_options['--run-id']
    load_model = run_options['--load']
    train_model = run_options['--train']
    save_freq = int(run_options['--save-freq'])
    keep_checkpoints = int(run_options['--keep-checkpoints'])
    worker_id = int(run_options['--worker-id'])
    curriculum_file = str(run_options['--curriculum'])
    if curriculum_file == "None":
        curriculum_file = None
    lesson = int(run_options['--lesson'])
    fast_simulation = not bool(run_options['--slow'])
    no_graphics = run_options['--no-graphics']

    # Constants
    # Assumption that this yaml is present in same dir as this file
    base_path = os.path.dirname(__file__)
    trainer_config_path = os.path.abspath(os.path.join(base_path, "trainer_config.yaml"))

    # Create controller and begin training.
    tc = TrainerController(env_path, run_id + "-" + str(sub_id), save_freq, curriculum_file,
                           fast_simulation, load_model, train_model, worker_id + sub_id,
                           keep_checkpoints, lesson, run_seed, docker_target_name,
                           trainer_config_path, no_graphics)
    tc.start_learning()


if __name__ == '__main__':
    print('''
    
                    ▄▄▄▓▓▓▓
               ╓▓▓▓▓▓▓█▓▓▓▓▓
          ,▄▄▄m▀▀▀'  ,▓▓▓▀▓▓▄                           ▓▓▓  ▓▓▌
        ▄▓▓▓▀'      ▄▓▓▀  ▓▓▓      ▄▄     ▄▄ ,▄▄ ▄▄▄▄   ,▄▄ ▄▓▓▌▄ ▄▄▄    ,▄▄
      ▄▓▓▓▀        ▄▓▓▀   ▐▓▓▌     ▓▓▌   ▐▓▓ ▐▓▓▓▀▀▀▓▓▌ ▓▓▓ ▀▓▓▌▀ ^▓▓▌  ╒▓▓▌
    ▄▓▓▓▓▓▄▄▄▄▄▄▄▄▓▓▓      ▓▀      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌   ▐▓▓▄ ▓▓▌
    ▀▓▓▓▓▀▀▀▀▀▀▀▀▀▀▓▓▄     ▓▓      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌    ▐▓▓▐▓▓
      ^█▓▓▓        ▀▓▓▄   ▐▓▓▌     ▓▓▓▓▄▓▓▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▓▄    ▓▓▓▓`
        '▀▓▓▓▄      ^▓▓▓  ▓▓▓       └▀▀▀▀ ▀▀ ^▀▀    `▀▀ `▀▀   '▀▀    ▐▓▓▌
           ▀▀▀▀▓▄▄▄   ▓▓▓▓▓▓,                                      ▓▓▓▓▀
               `▀█▓▓▓▓▓▓▓▓▓▌
                    ¬`▀▀▀█▓
                    
''')

    logger = logging.getLogger("unityagents")
    _USAGE = '''
    Usage:
      learn (<env>) [options]
      learn [options]
      learn --help

    Options:
      --curriculum=<file>        Curriculum json file for environment [default: None].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: False].
      --run-id=<path>            The directory name for model and summary statistics [default: ppo].
      --num-runs=<n>             Number of concurrent training sessions [default: 1]. 
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --seed=<n>                 Random seed used for training [default: -1].
      --slow                     Whether to run the game at training speed [default: False].
      --train                    Whether to train model, or only run inference [default: False].
      --worker-id=<n>            Number to add to communication port (5005) [default: 0].
      --docker-target-name=<dt>  Docker volume to store training-specific files [default: Empty].
      --no-graphics              Whether to run the environment in no-graphics mode [default: False].
    '''

    options = docopt(_USAGE)
    logger.info(options)
    num_runs = int(options['--num-runs'])
    seed = int(options['--seed'])

    if options['<env>'] is None and num_runs > 1:
        raise TrainerError("It is not possible to launch more than one concurrent training session "
                           "when training from the editor")

    jobs = []
    for i in range(num_runs):
        if seed == -1:
            seed = np.random.randint(0, 9999)
        p = multiprocessing.Process(target=run_training, args=(i, seed, options))
        jobs.append(p)
        p.start()
