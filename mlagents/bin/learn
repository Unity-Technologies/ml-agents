# # Unity ML-Agents Toolkit
# ## ML-Agent Learning

import logging

import os
import multiprocessing
import numpy as np
from docopt import docopt


from unitytrainers.trainer_controller import TrainerController
from unitytrainers.exception import TrainerError

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
      --run-id=<path>            The sub-directory name for model and summary statistics [default: ppo].
      --num-runs=<n>             Number of concurrent training sessions [default: 1]. 
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --seed=<n>                 Random seed used for training [default: -1].
      --slow                     Whether to run the game at training speed [default: False].
      --train                    Whether to train model, or only run inference [default: False].
      --worker-id=<n>            Number to add to communication port (5005). Used for multi-environment [default: 0].
      --docker-target-name=<dt>  Docker Volume to store curriculum, executable and model files [default: Empty].
      --no-graphics              Whether to run the Unity simulator in no-graphics mode [default: False].
    '''

    options = docopt(_USAGE)
    logger.info(options)
    # Docker Parameters
    if options['--docker-target-name'] == 'Empty':
        docker_target_name = ''
    else:
        docker_target_name = options['--docker-target-name']

    # General parameters
    run_id = options['--run-id']
    num_runs = int(options['--num-runs'])
    seed = int(options['--seed'])
    load_model = options['--load']
    train_model = options['--train']
    save_freq = int(options['--save-freq'])
    env_path = options['<env>']
    keep_checkpoints = int(options['--keep-checkpoints'])
    worker_id = int(options['--worker-id'])
    curriculum_file = str(options['--curriculum'])
    if curriculum_file == "None":
        curriculum_file = None
    lesson = int(options['--lesson'])
    fast_simulation = not bool(options['--slow'])
    no_graphics = options['--no-graphics']

    # Constants
    # Assumption that this yaml is present in same dir as this file
    base_path = os.path.dirname(__file__)
    TRAINER_CONFIG_PATH = os.path.abspath(os.path.join(base_path, "trainer_config.yaml"))


    def run_training(sub_id, use_seed):
        tc = TrainerController(env_path, run_id + "-" + str(sub_id), save_freq, curriculum_file, fast_simulation,
                               load_model, train_model, worker_id + sub_id, keep_checkpoints, lesson, use_seed,
                               docker_target_name, TRAINER_CONFIG_PATH, no_graphics)
        tc.start_learning()


    if env_path is None and num_runs > 1:
        raise TrainerError("It is not possible to launch more than one concurrent training session "
                           "when training from the editor")

    jobs = []
    for i in range(num_runs):
        if seed == -1:
            use_seed = np.random.randint(0, 9999)
        else:
            use_seed = seed
        p = multiprocessing.Process(target=run_training, args=(i, use_seed))
        jobs.append(p)
        p.start()
