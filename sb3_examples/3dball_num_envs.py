from math import ceil

from baselines.common.schedules import LinearSchedule
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from gym_unity.envs import make_mla_sb3_env, LimitedConfig

TOTAL_TAINING_STEPS_GOAL = (
    500000
)  # Same as config for CI 3dball... Not sure if MLA steps == SB3 steps.
NUM_ENVS = 12
STEPS_PER_UPDATE = 2048


# NOTE: This only achieves ~90/100 reward and is just a POC. Needs tuning to be useful.
def main():
    env = make_mla_sb3_env(
        LimitedConfig(
            env_path="/Users/henry.peteet/Documents/RandomBuilds/3DBallSingleNoVis",
            num_env=NUM_ENVS,
        )
    )
    # Log results in the "results" folder
    env = VecMonitor(env, "results")
    # Attempt to approximate settings from 3DBall.yaml
    schedule = LinearSchedule(
        schedule_timesteps=TOTAL_TAINING_STEPS_GOAL, final_p=0.0, initial_p=0.0003
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # TODO: Check if I am using schedule correctly.
        learning_rate=lambda progress: schedule.value(
            TOTAL_TAINING_STEPS_GOAL * progress
        ),
        tensorboard_log="results",
        n_steps=int(STEPS_PER_UPDATE),
    )
    training_rounds = ceil(TOTAL_TAINING_STEPS_GOAL / int(STEPS_PER_UPDATE * NUM_ENVS))
    for i in range(training_rounds):
        print(f"Training round {i + 1}/{training_rounds}")
        # NOTE: rest_num_timesteps should only happen the first time so that tensorboard logs are consistent.
        model.learn(total_timesteps=6000, reset_num_timesteps=(i == 0))
        model.policy.eval()
    env.close()


if __name__ == "__main__":
    main()
