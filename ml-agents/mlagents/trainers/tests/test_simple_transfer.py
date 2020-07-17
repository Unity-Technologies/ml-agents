import math
import tempfile
import pytest
import numpy as np
import attr
from typing import Dict

from mlagents.trainers.tests.transfer_test_envs import SimpleTransferEnvironment
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.trainer_util import TrainerFactory
from mlagents.trainers.simple_env_manager import SimpleEnvManager
from mlagents.trainers.demo_loader import write_demo
from mlagents.trainers.stats import (
    StatsReporter,
    StatsWriter,
    StatsSummary,
    TensorboardWriter,
    CSVWriter,
)
from mlagents.trainers.settings import (
    TrainerSettings,
    PPOSettings,
    PPOTransferSettings,
    SACSettings,
    NetworkSettings,
    SelfPlaySettings,
    BehavioralCloningSettings,
    GAILSettings,
    TrainerType,
    RewardSignalType,
)
from mlagents.trainers.models import EncoderType, ScheduleType
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.space_type_pb2 import discrete, continuous

BRAIN_NAME = "Simple"


PPO_CONFIG = TrainerSettings(
    trainer_type=TrainerType.PPO,
    hyperparameters=PPOSettings(
        learning_rate=5.0e-3,
        learning_rate_schedule=ScheduleType.CONSTANT,
        batch_size=16,
        buffer_size=64,
    ),
    network_settings=NetworkSettings(num_layers=2, hidden_units=32),
    summary_freq=500,
    max_steps=3000,
    threaded=False,
)

SAC_CONFIG = TrainerSettings(
    trainer_type=TrainerType.SAC,
    hyperparameters=SACSettings(
        learning_rate=5.0e-3,
        learning_rate_schedule=ScheduleType.CONSTANT,
        batch_size=8,
        buffer_init_steps=100,
        buffer_size=5000,
        tau=0.01,
        init_entcoef=0.01,
    ),
    network_settings=NetworkSettings(num_layers=2, hidden_units=16, normalize=True),
    summary_freq=100,
    max_steps=1000,
    threaded=False,
)

Transfer_CONFIG = TrainerSettings(
    trainer_type=TrainerType.PPO_Transfer,
    hyperparameters=PPOTransferSettings(
        learning_rate=5.0e-3,
        learning_rate_schedule=ScheduleType.CONSTANT,
        batch_size=16,
        buffer_size=64,
        feature_size=4,
        reuse_encoder=False,
        in_epoch_alter=True,
        # in_batch_alter=True,
        use_op_buffer=True,
        # policy_layers=0,
        # value_layers=0,
        # conv_thres=1e-4,
        # predict_return=True
        # separate_policy_train=True,
        # separate_value_train=True
        # separate_value_net=True,
    ),
    network_settings=NetworkSettings(num_layers=1, hidden_units=32),
    summary_freq=500,
    max_steps=3000,
    threaded=False,
)


# The reward processor is passed as an argument to _check_environment_trains.
# It is applied to the list pf all final rewards for each brain individually.
# This is so that we can process all final rewards in different ways for different algorithms.
# Custom reward processors shuld be built within the test function and passed to _check_environment_trains
# Default is average over the last 5 final rewards
def default_reward_processor(rewards, last_n_rewards=5):
    rewards_to_use = rewards[-last_n_rewards:]
    # For debugging tests
    print("Last {} rewards:".format(last_n_rewards), rewards_to_use)
    return np.array(rewards[-last_n_rewards:], dtype=np.float32).mean()


class DebugWriter(StatsWriter):
    """
    Print to stdout so stats can be viewed in pytest
    """

    def __init__(self):
        self._last_reward_summary: Dict[str, float] = {}
        self.stats = {}

    def get_last_rewards(self):
        return self._last_reward_summary

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        for val, stats_summary in values.items():
            if val == "Environment/Cumulative Reward":
                print(step, val, stats_summary.mean)
                self.stats[step] = stats_summary.mean
                self._last_reward_summary[category] = stats_summary.mean

    def write2file(self, filename):
        with open(filename, "w") as reward_file:
            for step in self.stats.keys():
                reward_file.write(str(step) + ":" + str(self.stats[step]) + "\n")


def _check_environment_trains(
    env,
    trainer_config,
    reward_processor=default_reward_processor,
    meta_curriculum=None,
    success_threshold=0.9,
    env_manager=None,
    run_id="id",
    seed=1337,
):
    # Create controller and begin training.
    model_dir = "./transfer_results/" + run_id
    StatsReporter.writers.clear()  # Clear StatsReporters so we don't write to file
    debug_writer = DebugWriter()
    StatsReporter.add_writer(debug_writer)

    csv_writer = CSVWriter(
        model_dir,
        required_fields=["Environment/Cumulative Reward", "Environment/Episode Length"],
    )
    tb_writer = TensorboardWriter(model_dir, clear_past_data=True)
    StatsReporter.add_writer(tb_writer)
    StatsReporter.add_writer(csv_writer)

    if env_manager is None:
        env_manager = SimpleEnvManager(env, EnvironmentParametersChannel())
    trainer_factory = TrainerFactory(
        trainer_config=trainer_config,
        output_path=model_dir,
        train_model=True,
        load_model=False,
        seed=seed,
        meta_curriculum=meta_curriculum,
        multi_gpu=False,
    )

    tc = TrainerController(
        trainer_factory=trainer_factory,
        output_path=model_dir,
        run_id=run_id,
        meta_curriculum=meta_curriculum,
        train=True,
        training_seed=seed,
    )

    # Begin training
    tc.start_learning(env_manager)
    # debug_writer.write2file(model_dir+"/reward.txt")

    # if (
    #     success_threshold is not None
    # ):  # For tests where we are just checking setup and not reward
    #     processed_rewards = [
    #         reward_processor(rewards) for rewards in env.final_rewards.values()
    #     ]
    #     assert all(not math.isnan(reward) for reward in processed_rewards)
    #     assert all(reward > success_threshold for reward in processed_rewards)


def test_2d_model(
    config=Transfer_CONFIG, obs_spec_type="rich1", run_id="model_rich1", seed=0
):
    env = SimpleTransferEnvironment(
        [BRAIN_NAME],
        use_discrete=False,
        action_size=2,
        step_size=0.1,
        num_vector=2,
        obs_spec_type=obs_spec_type,
        goal_type="hard",
    )
    new_hyperparams = attr.evolve(
        config.hyperparameters,
        batch_size=1200,
        buffer_size=12000,
        learning_rate=5.0e-3,
        use_bisim=True,
        predict_return=True,
        reuse_encoder=True,
        separate_value_train=True,
        separate_policy_train=False,
        use_var_predict=True,
        with_prior=False,
        use_op_buffer=False,
        in_epoch_alter=False,
        in_batch_alter=True,
        policy_layers=0,
        value_layers=2,
        forward_layers=2,
        encoder_layers=2,
        feature_size=16,
        # use_inverse_model=True
    )
    config = attr.evolve(
        config, hyperparameters=new_hyperparams, max_steps=20000, summary_freq=5000
    )
    _check_environment_trains(
        env, {BRAIN_NAME: config}, run_id=run_id + "_s" + str(seed), seed=seed
    )


def test_2d_transfer(
    config=Transfer_CONFIG,
    obs_spec_type="rich1",
    transfer_from="./transfer_results/model_rich2_f4_pv-l0_rew_bisim-op_s0/Simple",
    run_id="transfer_f4_rich1_from-rich2-retrain-pv_rew_bisim-op",
    seed=1337,
):
    env = SimpleTransferEnvironment(
        [BRAIN_NAME],
        use_discrete=False,
        action_size=2,
        step_size=0.1,
        num_vector=2,
        obs_spec_type=obs_spec_type,
        goal_type="hard",
    )
    new_hyperparams = attr.evolve(
        config.hyperparameters,
        batch_size=1200,
        buffer_size=12000,
        use_transfer=True,
        transfer_path=transfer_from,  # separate_policy_train=True, separate_value_train=True,
        use_op_buffer=False,
        in_epoch_alter=False,
        in_batch_alter=True,
        learning_rate=5.0e-3,
        train_policy=True,
        train_value=True,
        train_model=False,
        separate_value_train=True,
        separate_policy_train=False,
        feature_size=16,
        use_var_predict=True,
        with_prior=False,
        policy_layers=0,
        load_policy=False,
        load_value=False,
        predict_return=True,
        forward_layers=2,
        value_layers=2,
        encoder_layers=2,
        use_bisim=True,
    )
    config = attr.evolve(
        config, hyperparameters=new_hyperparams, max_steps=20000, summary_freq=5000
    )
    _check_environment_trains(
        env, {BRAIN_NAME: config}, run_id=run_id + "_s" + str(seed), seed=seed
    )


if __name__ == "__main__":
    for seed in range(5):
        for obs in ["normal", "rich1", "rich2"]:
            test_2d_model(seed=seed, obs_spec_type=obs, run_id="model_" + obs)

        # test_2d_model(config=SAC_CONFIG, run_id="sac_rich2_hard", seed=0)
        for obs in ["normal", "rich2"]:
            test_2d_transfer(
                seed=seed,
                obs_spec_type="rich1",
                transfer_from="./transfer_results/model_" + obs + "_s" + str(seed) + "/Simple",
                run_id=obs + "transfer_to_rich1",
            )

        for obs in ["normal", "rich1"]:
            test_2d_transfer(
                seed=seed,
                obs_spec_type="rich2",
                transfer_from="./transfer_results/model_" + obs + "_s" + str(seed) + "/Simple",
                run_id=obs + "transfer_to_rich2",
            )


# for obs in ["normal"]:
#     test_2d_transfer(seed=0, obs_spec_type="rich1",
#     transfer_from="./transfer_results/model_"+ obs +"_f4_pv-l0_rew_bisim-nop_newalter_noreuse-soft0.1_s0/Simple",
#     run_id="transfer_rich1_retrain-all_f4_pv-l0_rew_bisim-nop_noreuse-soft0.1_from_" + obs)
# for i in range(5):
#     test_2d_model(seed=i)
