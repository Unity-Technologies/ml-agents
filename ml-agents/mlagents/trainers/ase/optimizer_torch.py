import attr

from trainers.settings import OnPolicyHyperparamSettings, ScheduleType


@attr.s(auto_attribs=True)
class ASESettings(OnPolicyHyperparamSettings):
    latent_dim: int = 16
    beta: float = 5.0e-3
    epsilon: float = 0.2
    num_epoch: int = 3
    beta_sdo: float = 0.5
    omega_gp: float = 5
    omega_do: float = 0.01
    encoder_scaling: float = 1
    spu: int = 32768
    pv_mini_batch: int = 4096
    de_mini_batch: int = 1024
    gae_lambda: float = 0.95
    td_lambda: float = 0.95
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.CONSTANT
    beta_schedule: ScheduleType = ScheduleType.CONSTANT
    epsilon_schedule: ScheduleType = ScheduleType.CONSTANT
