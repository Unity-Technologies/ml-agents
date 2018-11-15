# # Unity ML-Agents Toolkit

import logging

from multiprocessing import Process, Queue
import numpy as np
from docopt import docopt

from .trainer_controller import TrainerController
from .exception import TrainerError


def run_training(sub_id, run_seed, run_options, process_queue):
    """
    Launches training session.
    :param process_queue: Queue used to send signal back to main.
    :param sub_id: Unique id for training session.
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    # Docker Parameters
    docker_target_name = (run_options['--docker-target-name']
        if run_options['--docker-target-name'] != 'None' else None)

    # General parameters
    env_path = (run_options['--env']
        if run_options['--env'] != 'None' else None)
    run_id = run_options['--run-id']
    load_model = run_options['--load']
    train_model = run_options['--train']
    save_freq = int(run_options['--save-freq'])
    keep_checkpoints = int(run_options['--keep-checkpoints'])
    worker_id = int(run_options['--worker-id'])
    curriculum_file = (run_options['--curriculum']
        if run_options['--curriculum'] != 'None' else None)
    lesson = int(run_options['--lesson'])
    fast_simulation = not bool(run_options['--slow'])
    no_graphics = run_options['--no-graphics']
    trainer_config_path = run_options['<trainer-config-path>']

    # Create controller and launch environment.
    tc = TrainerController(env_path, run_id + '-' + str(sub_id),
                           save_freq, curriculum_file, fast_simulation,
                           load_model, train_model, worker_id + sub_id,
                           keep_checkpoints, lesson, run_seed,
                           docker_target_name, trainer_config_path, no_graphics)

    # Signal that environment has been launched.
    process_queue.put(True)

    # Begin training
    tc.start_learning()


def main():
    try:
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
    except:
        print('\n\n\tUnity Technologies\n')

    logger = logging.getLogger('mlagents.trainers')
    _USAGE = '''
    Usage:
      mlagents-learn <trainer-config-path> [options]
      mlagents-learn --help

    Options:
      --env=<file>               Name of the Unity executable [default: None].
      --curriculum=<directory>   Curriculum json directory for environment [default: None].
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
      --docker-target-name=<dt>  Docker volume to store training-specific files [default: None].
      --no-graphics              Whether to run the environment in no-graphics mode [default: False].
    '''

    options = docopt(_USAGE)
    logger.info(options)
    num_runs = int(options['--num-runs'])
    seed = int(options['--seed'])

    if options['--env'] == 'None' and num_runs > 1:
        raise TrainerError('It is not possible to launch more than one concurrent training session '
                           'when training from the editor.')

    jobs = []
    run_seed = seed
    for i in range(num_runs):
        if seed == -1:
            run_seed = np.random.randint(0, 10000)
        process_queue = Queue()
        p = Process(target=run_training, args=(i, run_seed, options, process_queue))
        jobs.append(p)
        p.start()
        # Wait for signal that environment has successfully launched
        while process_queue.get() is not True:
            continue
