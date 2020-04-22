from typing import Dict, List

from mlagents_envs.base_env import BaseEnv, BehaviorName
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep, AllStepResult
from mlagents_envs.timers import timed
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.brain import BrainParameters
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents.trainers.brain_conversion_utils import behavior_spec_to_brain_parameters


class SimpleEnvManager(EnvManager):
    """
    Simple implementation of the EnvManager interface that only handles one BaseEnv at a time.
    This is generally only useful for testing; see SubprocessEnvManager for a production-quality implementation.
    """

    def __init__(self, env: BaseEnv, float_prop_channel: FloatPropertiesChannel):
        super().__init__()
        self.shared_float_properties = float_prop_channel
        self.env = env
        self.previous_step: EnvironmentStep = EnvironmentStep.empty(0)
        self.previous_all_action_info: Dict[str, ActionInfo] = {}

    def _step(self) -> List[EnvironmentStep]:
        all_action_info = self._take_step(self.previous_step)
        self.previous_all_action_info = all_action_info

        for brain_name, action_info in all_action_info.items():
            self.env.set_actions(brain_name, action_info.action)
        self.env.step()
        all_step_result = self._generate_all_results()

        step_info = EnvironmentStep(
            all_step_result, 0, self.previous_all_action_info, {}
        )
        self.previous_step = step_info
        return [step_info]

    def _reset_env(
        self, config: Dict[BehaviorName, float] = None
    ) -> List[EnvironmentStep]:  # type: ignore
        if config is not None:
            for k, v in config.items():
                self.shared_float_properties.set_property(k, v)
        self.env.reset()
        all_step_result = self._generate_all_results()
        self.previous_step = EnvironmentStep(all_step_result, 0, {}, {})
        return [self.previous_step]

    @property
    def external_brains(self) -> Dict[BehaviorName, BrainParameters]:
        result = {}
        for brain_name in self.env.get_behavior_names():
            result[brain_name] = behavior_spec_to_brain_parameters(
                brain_name, self.env.get_behavior_spec(brain_name)
            )
        return result

    @property
    def get_properties(self) -> Dict[BehaviorName, float]:
        return self.shared_float_properties.get_property_dict_copy()

    def close(self):
        self.env.close()

    @timed
    def _take_step(self, last_step: EnvironmentStep) -> Dict[BehaviorName, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        for brain_name, step_tuple in last_step.current_all_step_result.items():
            all_action_info[brain_name] = self.policies[brain_name].get_action(
                step_tuple[0],
                0,  # As there is only one worker, we assign the worker_id to 0.
            )
        return all_action_info

    def _generate_all_results(self) -> AllStepResult:
        all_step_result: AllStepResult = {}
        for brain_name in self.env.get_behavior_names():
            all_step_result[brain_name] = self.env.get_steps(brain_name)
        return all_step_result
