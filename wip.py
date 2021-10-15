from typing import Optional

import numpy as np
import ray
from gym import spaces
from gym.spaces import MultiDiscrete, Box
from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.contrib.maddpg import DEFAULT_CONFIG as MADDPG_CONFIG
from ray.rllib.contrib.maddpg.maddpg_policy import MADDPGTFPolicy
from ray.rllib.env import PettingZooEnv, EnvContext
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.tune import tune
from ray.tune.registry import register_env
from ray.tune.utils import merge_dicts
from supersuit import observation_lambda_v0, action_lambda_v1

from mlagents_envs.registry import default_registry
from mlagents_envs.registry.binary_utils import get_local_binary_path
from mlagents_envs.registry.remote_registry_entry import RemoteRegistryEntry
from pettingzoo_unity import UnityToPettingZooWrapper


class RandomPolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and
    `build_torch_policy` helpers instead for a real policy, which are
    described in the next sections.
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # example parameter
        self.w = 1.0

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]


def rllib_env_from_mla_registry(registry_name, config: EnvContext):
    unity_env = default_registry[registry_name].make(worker_id=config.worker_index)
    new_env = UnityToPettingZooWrapper(unity_env)
    new_env.metadata = {"name": registry_name}
    new_env.reset()
    orig_action_space = new_env.action_space
    orig_obs_space = new_env.observation_space

    # ----------------------------------------------------------------------
    # A BUNCH OF STUFF THAT SHOULD NOT BE NEEDED BUT IS ARCANE AT THE MOMENT
    # ----------------------------------------------------------------------
    def handle_actions(action, _space):
        if isinstance(orig_action_space, MultiDiscrete):
            return spaces.unflatten(orig_action_space, action)
        # Hack for some other issue I can't figure out.
        # Only needed when using PPO trainer, needs to be removed when running random examples.
        if registry_name in ("3DBall", "3DBallSingle"):
            # For some reason PPO is giving actions outside the action space...
            # I likely did something wrong but can't figure out what.
            return action - [1, 1]
        return action

    new_env = action_lambda_v1(new_env, handle_actions, lambda _: orig_action_space)

    def handle_obs(obs, space):
        if registry_name == "SoccerTwos":
            return spaces.flatten(orig_obs_space, obs[0]['observation'])
        if registry_name == "3DBall":
            if isinstance(obs, tuple):
                obs = obs[0]
            return obs
        if registry_name == "3DBallSingle":
            if isinstance(obs, tuple):
                obs = obs[0]
            return obs

    def handle_obs_space(space):
        if registry_name == "SoccerTwos":
            return spaces.flatten_space(orig_obs_space)
        if registry_name == "3DBall":
            return orig_obs_space
        if registry_name == "3DBallSingle":
            return orig_obs_space

    new_env = observation_lambda_v0(new_env, handle_obs, handle_obs_space)
    return PettingZooEnv(new_env)


def register_mla_env_in_rllib(mla_name: str, rllib_name: Optional[str] = None):
    register_env(rllib_name or mla_name,
                 lambda config: rllib_env_from_mla_registry(mla_name, config))


# MultiDiscrete([3]) has max value of 2, but when flattened becomes
# Box(low=[0], high=[3]) when it should result in Box(low=[0], high=[2])
def show_multidiscrete_flatten_bug():
    multi: MultiDiscrete = MultiDiscrete([3])
    assert not multi.contains([3])
    box: Box = spaces.flatten_space(multi)
    assert not box.contains([3])  # Fails because the shape now includes 3 in it's range when it shouldn't.


def random_actions_example_soccer():
    """
    Example manually running the rllib env with random actions.
    """
    try:
        env = rllib_env_from_mla_registry("SoccerTwos", EnvContext({}, worker_index=0))
        env.reset()
        # NOTE: action_space should be found via spaces.flatten_space(env.action_space) but flattening MultiDiscrete spaces
        #       seems broken.
        # TODO: File a bug around this based on the code documented in show_multidiscrete_flatten_bug
        action_space: spaces.Space = Box(low=np.array([0, 0, 0]), high=np.array([2, 2, 2]), shape=(3,), dtype=np.int64)
        for _ in range(10000):
            env.step({agent_id: action_space.sample() for agent_id in env.agents})
    finally:
        if env:
            env.close()


def random_actions_example_3dball():
    """
    Example manually running the rllib env with random actions.
    """
    try:
        env = rllib_env_from_mla_registry("3DBallSingle", EnvContext({}, worker_index=0))
        env.reset()
        for i in range(10000):
            steps = {agent_id: env.action_space.sample() for agent_id in env.agents}
            env.step(steps)
    finally:
        if env:
            env.close()


# TODO: Figure out why agents don't move, they "train" but don't really.
def maddpg_trainer_example():
    """
    Trying to use rllib MADDPG trainer to train.
    """
    register_mla_env_in_rllib("SoccerTwos")
    cfg = MADDPG_CONFIG
    cfg["agent_id"] = 0
    # From ValueError: Policy specs must be tuples/lists of (cls or None, obs_space, action_space, config), got None
    # TODO: action_space should be found via spaces.flatten_space(env.action_space) but flattening MultiDiscrete spaces
    #       seems broken. See show_multidiscrete_flatten_bug
    action_space: spaces.Space = Box(low=np.array([0, 0, 0]), high=np.array([2, 2, 2]), shape=(3,), dtype=np.int64)
    # Open the environment so we can derive the observation space.
    env = rllib_env_from_mla_registry("SoccerTwos", EnvContext({}, worker_index=0))
    env.reset()
    obs_space: spaces.Space = spaces.flatten_space(env.observation_space)
    env.close()
    cfg["multiagent"]["policies"] = {}
    cfg["multiagent"]["policies"]["shared_policy"] = (MADDPGTFPolicy, obs_space, action_space, {})
    # cfg["multiagent"]["policies"]["shared_policy"] = (RandomPolicy, obs_space, action_space, {})
    # Can't use policies_to_train because of https://github.com/ray-project/ray/issues/9651
    # cfg["multiagent"]["policies_to_train"] = ["shared_policy"]
    # def agent_to_policy(agent_id, _episode, **kwargs):
    #     if "team=1" in agent_id:
    #         return "shared_policy"
    #     return "random_policy"
    cfg["multiagent"]["policy_mapping_fn"] = lambda _agent_id, _episode, **kwargs: "shared_policy"
    cfg["num_workers"] = 5

    cfg["env"] = "SoccerTwos"
    tune.run(
        "contrib/MADDPG",
        stop={"training_iteration": 1000},
        config=cfg
    )


# TODO: Make this work with 3DBall instead of 3DBallSingle
def ppo_trainer_example(mla_env: str):
    """
    Trying to use rllib PPO trainer to train.
    """
    register_mla_env_in_rllib(mla_env)

    cfg = PPO_CONFIG
    cfg["multiagent"] = merge_dicts(COMMON_CONFIG["multiagent"], {
        "replay_mode": "lockstep",
        "policies": {"shared_policy": PolicySpec(policy_class=None, observation_space=None,
                                                 action_space=Box(low=np.array([-1, -1]), high=np.array([1, 1]),
                                                                  shape=(2,), dtype=np.int32), config=None)},
        "policy_mapping_fn": lambda agent_id, **kwargs: "shared_policy",
    })
    cfg["num_workers"] = 10

    # Hyperparam cloning from repo:
    cfg["entropy_coeff"] = 0.001 # Beta
    cfg["clip_param"] = 0.2  # epsilon
    cfg["lambda"] = 0.95 # lamdba
    cfg["lr"] = 0.0003  # learning_rate

    cfg["env"] = mla_env
    tune.run(
        "PPO",
        stop={"training_iteration": 100},
        config=cfg,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        # restore="~/ray-results/SOME_CHECKPOINT",
    )


def main():
    try:
        # Use locally so we don't shut down and restart every time for debugging
        # To start locally run `ray start --head --port=6379`
        ray.init(address="auto")
        # Add local mac build to the registry.
        default_registry.register(
            RemoteRegistryEntry(
                identifier="3DBallSingle",
                expected_reward=100,
                description=None,
                linux_url=None,
                darwin_url="file:///Users/henry.peteet/Documents/RandomBuilds/3DBallSingle.zip",
                win_url=None,
                additional_args=["--mlagents-scene-name", "Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity"]
            )
        )
        # Use in cloud
        # ray.init(num_cpus=10, local_mode=True)

        # Actual examples
        # random_actions_example_3dball()
        # random_actions_example_soccer()
        ppo_trainer_example("3DBall")
        # ppo_trainer_example("3DBallSingle")
        # maddpg_trainer_example()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
