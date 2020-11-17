import pytest
import attr


from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
    MemoryEnvironment,
    RecordEnvironment,
)

from mlagents.trainers.demo_loader import write_demo

from mlagents.trainers.settings import (
    NetworkSettings,
    SelfPlaySettings,
    BehavioralCloningSettings,
    GAILSettings,
    RewardSignalType,
    EncoderType,
    FrameworkType,
)

from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.space_type_pb2 import discrete, continuous

from mlagents.trainers.tests.dummy_config import ppo_dummy_config, sac_dummy_config
from mlagents.trainers.tests.check_env_trains import (
    check_environment_trains,
    default_reward_processor,
)

BRAIN_NAME = "1D"

PPO_TORCH_CONFIG = attr.evolve(ppo_dummy_config(), framework=FrameworkType.PYTORCH)
SAC_TORCH_CONFIG = attr.evolve(sac_dummy_config(), framework=FrameworkType.PYTORCH)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ppo(use_discrete):
    env = SimpleEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    config = attr.evolve(PPO_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("use_discrete", [True, False])
def test_2d_ppo(use_discrete):
    env = SimpleEnvironment(
        [BRAIN_NAME], use_discrete=use_discrete, action_size=2, step_size=0.8
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, batch_size=64, buffer_size=640
    )
    config = attr.evolve(
        PPO_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=10000
    )
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("use_discrete", [True, False])
@pytest.mark.parametrize("num_visual", [1, 2])
def test_visual_ppo(num_visual, use_discrete):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        use_discrete=use_discrete,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.2,
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, learning_rate=3.0e-4
    )
    config = attr.evolve(PPO_TORCH_CONFIG, hyperparameters=new_hyperparams)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("num_visual", [1, 2])
@pytest.mark.parametrize("vis_encode_type", ["resnet", "nature_cnn", "match3"])
def test_visual_advanced_ppo(vis_encode_type, num_visual):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        use_discrete=True,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.5,
        vis_obs_size=(5, 5, 5) if vis_encode_type == "match3" else (36, 36, 3),
    )
    new_networksettings = attr.evolve(
        SAC_TORCH_CONFIG.network_settings, vis_encode_type=EncoderType(vis_encode_type)
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, learning_rate=3.0e-4
    )
    config = attr.evolve(
        PPO_TORCH_CONFIG,
        hyperparameters=new_hyperparams,
        network_settings=new_networksettings,
        max_steps=900,
        summary_freq=100,
    )
    # The number of steps is pretty small for these encoders
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.5)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_recurrent_ppo(use_discrete):
    env = MemoryEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    new_network_settings = attr.evolve(
        PPO_TORCH_CONFIG.network_settings,
        memory=NetworkSettings.MemorySettings(memory_size=16),
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters,
        learning_rate=1.0e-3,
        batch_size=64,
        buffer_size=128,
    )
    config = attr.evolve(
        PPO_TORCH_CONFIG,
        hyperparameters=new_hyperparams,
        network_settings=new_network_settings,
        max_steps=6000,
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.9)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_sac(use_discrete):
    env = SimpleEnvironment([BRAIN_NAME], use_discrete=use_discrete)
    config = attr.evolve(SAC_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("use_discrete", [True, False])
def test_2d_sac(use_discrete):
    env = SimpleEnvironment(
        [BRAIN_NAME], use_discrete=use_discrete, action_size=2, step_size=0.8
    )
    new_hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters, buffer_init_steps=2000
    )
    config = attr.evolve(
        SAC_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=10000
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.8)


@pytest.mark.parametrize("use_discrete", [True, False])
@pytest.mark.parametrize("num_visual", [1, 2])
def test_visual_sac(num_visual, use_discrete):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        use_discrete=use_discrete,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.2,
    )
    new_hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters, batch_size=16, learning_rate=3e-4
    )
    config = attr.evolve(SAC_TORCH_CONFIG, hyperparameters=new_hyperparams)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("num_visual", [1, 2])
@pytest.mark.parametrize("vis_encode_type", ["resnet", "nature_cnn", "match3"])
def test_visual_advanced_sac(vis_encode_type, num_visual):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        use_discrete=True,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.5,
        vis_obs_size=(5, 5, 5) if vis_encode_type == "match3" else (36, 36, 3),
    )
    new_networksettings = attr.evolve(
        SAC_TORCH_CONFIG.network_settings, vis_encode_type=EncoderType(vis_encode_type)
    )
    new_hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters,
        batch_size=16,
        learning_rate=3e-4,
        buffer_init_steps=0,
    )
    config = attr.evolve(
        SAC_TORCH_CONFIG,
        hyperparameters=new_hyperparams,
        network_settings=new_networksettings,
        max_steps=100,
    )
    # The number of steps is pretty small for these encoders
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.5)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_recurrent_sac(use_discrete):
    step_size = 0.2 if use_discrete else 0.5
    env = MemoryEnvironment(
        [BRAIN_NAME], use_discrete=use_discrete, step_size=step_size
    )
    new_networksettings = attr.evolve(
        SAC_TORCH_CONFIG.network_settings,
        memory=NetworkSettings.MemorySettings(memory_size=16, sequence_length=16),
    )
    new_hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters,
        batch_size=256,
        learning_rate=1e-3,
        buffer_init_steps=1000,
        steps_per_update=2,
    )
    config = attr.evolve(
        SAC_TORCH_CONFIG,
        hyperparameters=new_hyperparams,
        network_settings=new_networksettings,
        max_steps=2000,
    )
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ghost(use_discrete):
    env = SimpleEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], use_discrete=use_discrete
    )
    self_play_settings = SelfPlaySettings(
        play_against_latest_model_ratio=1.0, save_steps=2000, swap_steps=2000
    )
    config = attr.evolve(PPO_TORCH_CONFIG, self_play=self_play_settings, max_steps=2500)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_ghost_fails(use_discrete):
    env = SimpleEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], use_discrete=use_discrete
    )
    # This config should fail because the ghosted policy is never swapped with a competent policy.
    # Swap occurs after max step is reached.
    self_play_settings = SelfPlaySettings(
        play_against_latest_model_ratio=1.0, save_steps=2000, swap_steps=4000
    )
    config = attr.evolve(PPO_TORCH_CONFIG, self_play=self_play_settings, max_steps=2500)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=None)
    processed_rewards = [
        default_reward_processor(rewards) for rewards in env.final_rewards.values()
    ]
    success_threshold = 0.9
    assert any(reward > success_threshold for reward in processed_rewards) and any(
        reward < success_threshold for reward in processed_rewards
    )


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_asymm_ghost(use_discrete):
    # Make opponent for asymmetric case
    brain_name_opp = BRAIN_NAME + "Opp"
    env = SimpleEnvironment(
        [BRAIN_NAME + "?team=0", brain_name_opp + "?team=1"], use_discrete=use_discrete
    )
    self_play_settings = SelfPlaySettings(
        play_against_latest_model_ratio=1.0,
        save_steps=10000,
        swap_steps=10000,
        team_change=400,
    )
    config = attr.evolve(PPO_TORCH_CONFIG, self_play=self_play_settings, max_steps=4000)
    check_environment_trains(env, {BRAIN_NAME: config, brain_name_opp: config})


@pytest.mark.parametrize("use_discrete", [True, False])
def test_simple_asymm_ghost_fails(use_discrete):
    # Make opponent for asymmetric case
    brain_name_opp = BRAIN_NAME + "Opp"
    env = SimpleEnvironment(
        [BRAIN_NAME + "?team=0", brain_name_opp + "?team=1"], use_discrete=use_discrete
    )
    # This config should fail because the team that us not learning when both have reached
    # max step should be executing the initial, untrained poliy.
    self_play_settings = SelfPlaySettings(
        play_against_latest_model_ratio=0.0,
        save_steps=5000,
        swap_steps=5000,
        team_change=2000,
    )
    config = attr.evolve(PPO_TORCH_CONFIG, self_play=self_play_settings, max_steps=3000)
    check_environment_trains(
        env, {BRAIN_NAME: config, brain_name_opp: config}, success_threshold=None
    )
    processed_rewards = [
        default_reward_processor(rewards) for rewards in env.final_rewards.values()
    ]
    success_threshold = 0.9
    assert any(reward > success_threshold for reward in processed_rewards) and any(
        reward < success_threshold for reward in processed_rewards
    )


@pytest.fixture(scope="session")
def simple_record(tmpdir_factory):
    def record_demo(use_discrete, num_visual=0, num_vector=1):
        env = RecordEnvironment(
            [BRAIN_NAME],
            use_discrete=use_discrete,
            num_visual=num_visual,
            num_vector=num_vector,
            n_demos=100,
        )
        # If we want to use true demos, we can solve the env in the usual way
        # Otherwise, we can just call solve to execute the optimal policy
        env.solve()
        agent_info_protos = env.demonstration_protos[BRAIN_NAME]
        meta_data_proto = DemonstrationMetaProto()
        brain_param_proto = BrainParametersProto(
            vector_action_size=[2] if use_discrete else [1],
            vector_action_descriptions=[""],
            vector_action_space_type=discrete if use_discrete else continuous,
            brain_name=BRAIN_NAME,
            is_training=True,
        )
        action_type = "Discrete" if use_discrete else "Continuous"
        demo_path_name = "1DTest" + action_type + ".demo"
        demo_path = str(tmpdir_factory.mktemp("tmp_demo").join(demo_path_name))
        write_demo(demo_path, meta_data_proto, brain_param_proto, agent_info_protos)
        return demo_path

    return record_demo


@pytest.mark.parametrize("use_discrete", [True, False])
@pytest.mark.parametrize("trainer_config", [PPO_TORCH_CONFIG, SAC_TORCH_CONFIG])
def test_gail(simple_record, use_discrete, trainer_config):
    demo_path = simple_record(use_discrete)
    env = SimpleEnvironment([BRAIN_NAME], use_discrete=use_discrete, step_size=0.2)
    bc_settings = BehavioralCloningSettings(demo_path=demo_path, steps=1000)
    reward_signals = {
        RewardSignalType.GAIL: GAILSettings(encoding_size=32, demo_path=demo_path)
    }
    config = attr.evolve(
        trainer_config,
        reward_signals=reward_signals,
        behavioral_cloning=bc_settings,
        max_steps=500,
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.9)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_gail_visual_ppo(simple_record, use_discrete):
    demo_path = simple_record(use_discrete, num_visual=1, num_vector=0)
    env = SimpleEnvironment(
        [BRAIN_NAME],
        num_visual=1,
        num_vector=0,
        use_discrete=use_discrete,
        step_size=0.2,
    )
    bc_settings = BehavioralCloningSettings(demo_path=demo_path, steps=1500)
    reward_signals = {
        RewardSignalType.GAIL: GAILSettings(encoding_size=32, demo_path=demo_path)
    }
    hyperparams = attr.evolve(PPO_TORCH_CONFIG.hyperparameters, learning_rate=3e-4)
    config = attr.evolve(
        PPO_TORCH_CONFIG,
        reward_signals=reward_signals,
        hyperparameters=hyperparams,
        behavioral_cloning=bc_settings,
        max_steps=1000,
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.9)


@pytest.mark.parametrize("use_discrete", [True, False])
def test_gail_visual_sac(simple_record, use_discrete):
    demo_path = simple_record(use_discrete, num_visual=1, num_vector=0)
    env = SimpleEnvironment(
        [BRAIN_NAME],
        num_visual=1,
        num_vector=0,
        use_discrete=use_discrete,
        step_size=0.2,
    )
    bc_settings = BehavioralCloningSettings(demo_path=demo_path, steps=1000)
    reward_signals = {
        RewardSignalType.GAIL: GAILSettings(encoding_size=32, demo_path=demo_path)
    }
    hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters, learning_rate=3e-4, batch_size=16
    )
    config = attr.evolve(
        SAC_TORCH_CONFIG,
        reward_signals=reward_signals,
        hyperparameters=hyperparams,
        behavioral_cloning=bc_settings,
        max_steps=500,
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.9)
