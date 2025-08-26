import attr
from mlagents.trainers.settings import ScheduleType, OnPolicyHyperparamSettings
from mlagents.trainers.exception import TrainerConfigError

@attr.s(auto_attribs=True)
class DroneTrainerSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    lambd: float = 0.95
    num_epoch: int = attr.ib(default=1)
    shared_critic: bool = False

    @num_epoch.validator
    def _check_num_epoch_one(self, attribute, value):
        if value != 1:
            raise TrainerConfigError("A2C requires num_epoch = 1")

    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR