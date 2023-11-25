import pytest
import attr


from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
    MultiAgentEnvironment,
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
    ConditioningType,
)

from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import (
    BrainParametersProto,
    ActionSpecProto,
)

from mlagents.trainers.tests.dummy_config import (
    ppo_dummy_config,
    sac_dummy_config,
    poca_dummy_config,
)
from mlagents.trainers.tests.check_env_trains import (
    check_environment_trains,
    default_reward_processor,
)

BRAIN_NAME = "1D"

PPO_TORCH_CONFIG = ppo_dummy_config()
SAC_TORCH_CONFIG = sac_dummy_config()
POCA_TORCH_CONFIG = poca_dummy_config()

# tests in this file won't be tested on GPU machine
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_poca(action_sizes):
    env = MultiAgentEnvironment([BRAIN_NAME], action_sizes=action_sizes, num_agents=2)
    config = attr.evolve(POCA_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("num_visual", [1, 2])
def test_visual_poca(num_visual):
    env = MultiAgentEnvironment(
        [BRAIN_NAME], action_sizes=(0, 1), num_agents=2, num_visual=num_visual
    )
    new_hyperparams = attr.evolve(
        POCA_TORCH_CONFIG.hyperparameters, learning_rate=3.0e-4
    )
    config = attr.evolve(POCA_TORCH_CONFIG, hyperparameters=new_hyperparams)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("conditioning_type", [ConditioningType.HYPER])
@pytest.mark.parametrize("num_var_len", [1, 2])
@pytest.mark.parametrize("num_vector", [0, 1])
@pytest.mark.parametrize("num_vis", [0, 1])
def test_var_len_obs_and_goal_poca(num_vis, num_vector, num_var_len, conditioning_type):
    env = MultiAgentEnvironment(
        [BRAIN_NAME],
        action_sizes=(0, 1),
        num_visual=num_vis,
        num_vector=num_vector,
        num_var_len=num_var_len,
        step_size=0.2,
        num_agents=2,
        goal_indices=[0],
    )
    new_network = attr.evolve(
        POCA_TORCH_CONFIG.network_settings, goal_conditioning_type=conditioning_type
    )
    new_hyperparams = attr.evolve(
        POCA_TORCH_CONFIG.hyperparameters, learning_rate=3.0e-4
    )
    config = attr.evolve(
        POCA_TORCH_CONFIG,
        hyperparameters=new_hyperparams,
        network_settings=new_network,
        max_steps=5000,
    )
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("is_multiagent", [True, False])
def test_recurrent_poca(action_sizes, is_multiagent):
    if is_multiagent:
        # This is not a recurrent environment, just check if LSTM doesn't crash
        env = MultiAgentEnvironment(
            [BRAIN_NAME], action_sizes=action_sizes, num_agents=2
        )
    else:
        # Actually test LSTM here
        env = MemoryEnvironment([BRAIN_NAME], action_sizes=action_sizes)
    new_network_settings = attr.evolve(
        POCA_TORCH_CONFIG.network_settings,
        memory=NetworkSettings.MemorySettings(memory_size=16),
    )
    new_hyperparams = attr.evolve(
        POCA_TORCH_CONFIG.hyperparameters,
        learning_rate=1.0e-3,
        batch_size=64,
        buffer_size=128,
    )
    config = attr.evolve(
        POCA_TORCH_CONFIG,
        hyperparameters=new_hyperparams,
        network_settings=new_network_settings,
        max_steps=500 if is_multiagent else 6000,
    )
    check_environment_trains(
        env, {BRAIN_NAME: config}, success_threshold=None if is_multiagent else 0.9
    )


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_ppo(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes)
    config = attr.evolve(PPO_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("action_sizes", [(0, 2), (2, 0)])
def test_2d_ppo(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes, step_size=0.8)
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, batch_size=64, buffer_size=640
    )
    config = attr.evolve(
        PPO_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=10000
    )
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("num_visual", [1, 2])
@pytest.mark.parametrize("shared_critic", [True, False])
def test_visual_ppo(shared_critic, num_visual, action_sizes):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        action_sizes=action_sizes,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.2,
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters,
        learning_rate=3.0e-4,
        shared_critic=shared_critic,
    )
    config = attr.evolve(PPO_TORCH_CONFIG, hyperparameters=new_hyperparams)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("conditioning_type", [ConditioningType.HYPER])
@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("num_var_len", [1, 2])
@pytest.mark.parametrize("num_vector", [0, 1])
@pytest.mark.parametrize("num_vis", [0, 1])
def test_var_len_obs_and_goal_ppo(
    num_vis, num_vector, num_var_len, action_sizes, conditioning_type
):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        action_sizes=action_sizes,
        num_visual=num_vis,
        num_vector=num_vector,
        num_var_len=num_var_len,
        step_size=0.2,
        goal_indices=[0],
    )
    new_network = attr.evolve(
        POCA_TORCH_CONFIG.network_settings, goal_conditioning_type=conditioning_type
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, learning_rate=3.0e-4
    )
    config = attr.evolve(
        PPO_TORCH_CONFIG, hyperparameters=new_hyperparams, network_settings=new_network
    )
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("num_visual", [1, 2])
@pytest.mark.parametrize("vis_encode_type", ["resnet", "nature_cnn", "match3"])
def test_visual_advanced_ppo(vis_encode_type, num_visual):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        action_sizes=(0, 1),
        num_visual=num_visual,
        num_vector=0,
        step_size=0.5,
        vis_obs_size=(5, 5, 5) if vis_encode_type == "match3" else (3, 36, 36),
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


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_recurrent_ppo(action_sizes):
    env = MemoryEnvironment([BRAIN_NAME], action_sizes=action_sizes)
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


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_sac(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes)
    config = attr.evolve(SAC_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("action_sizes", [(0, 2), (2, 0)])
def test_2d_sac(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes, step_size=0.8)
    new_hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters, buffer_init_steps=2000
    )
    config = attr.evolve(
        SAC_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=3000
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.8)


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("num_visual", [1, 2])
def test_visual_sac(num_visual, action_sizes):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        action_sizes=action_sizes,
        num_visual=num_visual,
        num_vector=0,
        step_size=0.2,
    )
    new_hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters, batch_size=16, learning_rate=3e-4
    )
    config = attr.evolve(SAC_TORCH_CONFIG, hyperparameters=new_hyperparams)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("num_var_len", [1, 2])
def test_var_len_obs_sac(num_var_len, action_sizes):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        action_sizes=action_sizes,
        num_visual=0,
        num_var_len=num_var_len,
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
        action_sizes=(0, 1),
        num_visual=num_visual,
        num_vector=0,
        step_size=0.5,
        vis_obs_size=(5, 5, 5) if vis_encode_type == "match3" else (3, 36, 36),
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


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_recurrent_sac(action_sizes):
    step_size = 0.2 if action_sizes == (0, 1) else 0.5
    env = MemoryEnvironment(
        [BRAIN_NAME], action_sizes=action_sizes, step_size=step_size
    )
    new_networksettings = attr.evolve(
        SAC_TORCH_CONFIG.network_settings,
        memory=NetworkSettings.MemorySettings(memory_size=16, sequence_length=16),
    )
    new_hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters,
        batch_size=256,
        learning_rate=3e-4,
        buffer_init_steps=1000,
        steps_per_update=2,
    )
    config = attr.evolve(
        SAC_TORCH_CONFIG,
        hyperparameters=new_hyperparams,
        network_settings=new_networksettings,
        max_steps=4000,
    )
    check_environment_trains(env, {BRAIN_NAME: config}, training_seed=1337)


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_ghost(action_sizes):
    env = SimpleEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], action_sizes=action_sizes
    )
    self_play_settings = SelfPlaySettings(
        play_against_latest_model_ratio=1.0, save_steps=2000, swap_steps=2000
    )
    config = attr.evolve(PPO_TORCH_CONFIG, self_play=self_play_settings, max_steps=2500)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_ghost_fails(action_sizes):
    env = SimpleEnvironment(
        [BRAIN_NAME + "?team=0", BRAIN_NAME + "?team=1"], action_sizes=action_sizes
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


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_asymm_ghost(action_sizes):
    # Make opponent for asymmetric case
    brain_name_opp = BRAIN_NAME + "Opp"
    env = SimpleEnvironment(
        [BRAIN_NAME + "?team=0", brain_name_opp + "?team=1"], action_sizes=action_sizes
    )
    self_play_settings = SelfPlaySettings(
        play_against_latest_model_ratio=1.0,
        save_steps=10000,
        swap_steps=10000,
        team_change=400,
    )
    config = attr.evolve(PPO_TORCH_CONFIG, self_play=self_play_settings, max_steps=4000)
    check_environment_trains(env, {BRAIN_NAME: config, brain_name_opp: config})


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_asymm_ghost_fails(action_sizes):
    # Make opponent for asymmetric case
    brain_name_opp = BRAIN_NAME + "Opp"
    env = SimpleEnvironment(
        [BRAIN_NAME + "?team=0", brain_name_opp + "?team=1"], action_sizes=action_sizes
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
    def record_demo(action_sizes, num_visual=0, num_vector=1):
        env = RecordEnvironment(
            [BRAIN_NAME],
            action_sizes=action_sizes,
            num_visual=num_visual,
            num_vector=num_vector,
            n_demos=100,
        )
        # If we want to use true demos, we can solve the env in the usual way
        # Otherwise, we can just call solve to execute the optimal policy
        env.solve()
        agent_info_protos = env.demonstration_protos[BRAIN_NAME]
        meta_data_proto = DemonstrationMetaProto()
        continuous_action_size, discrete_action_size = action_sizes
        action_spec_proto = ActionSpecProto(
            num_continuous_actions=continuous_action_size,
            num_discrete_actions=discrete_action_size,
            discrete_branch_sizes=[2] if discrete_action_size > 0 else None,
        )
        brain_param_proto = BrainParametersProto(
            brain_name=BRAIN_NAME, is_training=True, action_spec=action_spec_proto
        )
        action_type = "Discrete" if action_sizes else "Continuous"
        demo_path_name = "1DTest" + action_type + ".demo"
        demo_path = str(tmpdir_factory.mktemp("tmp_demo").join(demo_path_name))
        write_demo(demo_path, meta_data_proto, brain_param_proto, agent_info_protos)
        return demo_path

    return record_demo


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("trainer_config", [PPO_TORCH_CONFIG, SAC_TORCH_CONFIG])
def test_gail(simple_record, action_sizes, trainer_config):
    demo_path = simple_record(action_sizes)
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes, step_size=0.2)
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


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_gail_visual_ppo(simple_record, action_sizes):
    demo_path = simple_record(action_sizes, num_visual=1, num_vector=0)
    env = SimpleEnvironment(
        [BRAIN_NAME],
        num_visual=1,
        num_vector=0,
        action_sizes=action_sizes,
        step_size=0.3,
    )
    bc_settings = BehavioralCloningSettings(demo_path=demo_path, steps=1500)
    reward_signals = {
        RewardSignalType.GAIL: GAILSettings(
            gamma=0.8, encoding_size=32, demo_path=demo_path
        )
    }
    hyperparams = attr.evolve(PPO_TORCH_CONFIG.hyperparameters, learning_rate=1e-3)
    config = attr.evolve(
        PPO_TORCH_CONFIG,
        reward_signals=reward_signals,
        hyperparameters=hyperparams,
        behavioral_cloning=bc_settings,
        max_steps=1000,
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.9)


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_gail_visual_sac(simple_record, action_sizes):
    demo_path = simple_record(action_sizes, num_visual=1, num_vector=0)
    env = SimpleEnvironment(
        [BRAIN_NAME],
        num_visual=1,
        num_vector=0,
        action_sizes=action_sizes,
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
