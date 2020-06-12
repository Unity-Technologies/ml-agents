from mlagents.trainers.tests.test_simple_rl import (
    _check_environment_trains,
    PPO_CONFIG,
    generate_config,
)
from mlagents.trainers.tests.simple_test_envs import SimpleEnvironment
from mlagents.trainers.ppo.trainer import TIMINGS
import matplotlib.pyplot as plt
import numpy as np

BRAIN_NAME = "1D"

if __name__ == "__main__":
    env = SimpleEnvironment([BRAIN_NAME], use_discrete=False)
    config = generate_config(
        PPO_CONFIG,
        override_vals={"batch_size": 256, "max_steps": 20000, "buffer_size": 1024},
    )
    try:
        _check_environment_trains(env, config)
    except Exception:
        pass
    print(f"Mean update time {np.mean(TIMINGS)}")
    plt.plot(TIMINGS)
    plt.ylim((0, 0.006))
    plt.title("PyTorch w/ 3DBall Running, batch size 256, 32 hidden units, 1 layer")
    plt.ylabel("Update Time (s)")
    plt.ylabel("Update #")
    plt.show()
