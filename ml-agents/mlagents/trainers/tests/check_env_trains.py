import math
import tempfile
import numpy as np
from typing import Dict
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.simple_env_manager import SimpleEnvManager
from mlagents.trainers.stats import StatsReporter, StatsWriter, StatsSummary
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)


class DebugWriter(StatsWriter):
    """
    Print to stdout so stats can be viewed in pytest
    """

    def __init__(self):
        self._last_reward_summary: Dict[str, float] = {}

    def get_last_rewards(self):
        return self._last_reward_summary

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        for val, stats_summary in values.items():
            if (
                val == "Environment/Cumulative Reward"
                or val == "Environment/Group Cumulative Reward"
            ):

                print(step, val, stats_summary.aggregated_value)
                self._last_reward_summary[category] = stats_summary.aggregated_value


# The reward processor is passed as an argument to _check_environment_trains.
# It is applied to the list of all final rewards for each brain individually.
# This is so that we can process all final rewards in different ways for different algorithms.
# Custom reward processors should be built within the test function and passed to _check_environment_trains
# Default is average over the last 5 final rewards
def default_reward_processor(rewards, last_n_rewards=5):
    rewards_to_use = rewards[-last_n_rewards:]
    # For debugging tests
    print(f"Last {last_n_rewards} rewards:", rewards_to_use)
    return np.array(rewards[-last_n_rewards:], dtype=np.float32).mean()


def check_environment_trains(
    env,
    trainer_config,
    reward_processor=default_reward_processor,
    env_parameter_manager=None,
    success_threshold=0.9,
    env_manager=None,
    training_seed=None,
):
    if env_parameter_manager is None:
        env_parameter_manager = EnvironmentParameterManager()
    # Create controller and begin training.
    with tempfile.TemporaryDirectory() as dir:
        run_id = "id"
        seed = 1337 if training_seed is None else training_seed
        StatsReporter.writers.clear()  # Clear StatsReporters so we don't write to file
        debug_writer = DebugWriter()
        StatsReporter.add_writer(debug_writer)
        if env_manager is None:
            env_manager = SimpleEnvManager(env, EnvironmentParametersChannel())
        trainer_factory = TrainerFactory(
            trainer_config=trainer_config,
            output_path=dir,
            train_model=True,
            load_model=False,
            seed=seed,
            param_manager=env_parameter_manager,
            multi_gpu=False,
        )

        tc = TrainerController(
            trainer_factory=trainer_factory,
            output_path=dir,
            run_id=run_id,
            param_manager=env_parameter_manager,
            train=True,
            training_seed=seed,
        )

        # Begin training
        tc.start_learning(env_manager)
        if (
            success_threshold is not None
        ):  # For tests where we are just checking setup and not reward
            processed_rewards = [
                reward_processor(rewards) for rewards in env.final_rewards.values()
            ]
            assert all(not math.isnan(reward) for reward in processed_rewards)
            assert all(reward > success_threshold for reward in processed_rewards)
