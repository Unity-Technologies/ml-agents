from typing import Dict, List

from mlagents_envs.base_env import BaseEnv
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep
from mlagents_envs.timers import timed
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.brain import BrainParameters, AllBrainInfo
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents.trainers.brain_conversion_utils import (
    step_result_to_brain_info,
    group_spec_to_brain_parameters,
)


class SimpleEnvManager(EnvManager):
    """
    Simple implementation of the EnvManager interface that only handles one BaseEnv at a time.
    This is generally only useful for testing; see SubprocessEnvManager for a production-quality implementation.
    """

    def __init__(self, env: BaseEnv, float_prop_channel: FloatPropertiesChannel):
        super().__init__()
        self.shared_float_properties = float_prop_channel
        self.env = env
        self.previous_step: EnvironmentStep = EnvironmentStep({}, {}, {})
        self.previous_all_action_info: Dict[str, ActionInfo] = {}

    def step(self) -> List[EnvironmentStep]:
        all_action_info = self._take_step(self.previous_step)
        self.previous_all_action_info = all_action_info

        for brain_name, action_info in all_action_info.items():
            self.env.set_actions(brain_name, action_info.action)
        self.env.step()
        all_brain_info = self._generate_all_brain_info()
        step_brain_info = all_brain_info

        step_info = EnvironmentStep(
            self.previous_step.current_all_brain_info,
            step_brain_info,
            self.previous_all_action_info,
        )
        self.previous_step = step_info
        return [step_info]

    def reset(
        self, config: Dict[str, float] = None
    ) -> List[EnvironmentStep]:  # type: ignore
        if config is not None:
            for k, v in config.items():
                self.shared_float_properties.set_property(k, v)
        self.env.reset()
        all_brain_info = self._generate_all_brain_info()
        self.previous_step = EnvironmentStep({}, all_brain_info, {})
        return [self.previous_step]

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        result = {}
        for brain_name in self.env.get_agent_groups():
            result[brain_name] = group_spec_to_brain_parameters(
                brain_name, self.env.get_agent_group_spec(brain_name)
            )
        return result

    @property
    def get_properties(self) -> Dict[str, float]:
        return self.shared_float_properties.get_property_dict_copy()

    def close(self):
        self.env.close()

    @timed
    def _take_step(self, last_step: EnvironmentStep) -> Dict[str, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        for brain_name, brain_info in last_step.current_all_brain_info.items():
            all_action_info[brain_name] = self.policies[brain_name].get_action(
                brain_info
            )
        return all_action_info

    def _generate_all_brain_info(self) -> AllBrainInfo:
        all_brain_info = {}
        for brain_name in self.env.get_agent_groups():
            all_brain_info[brain_name] = step_result_to_brain_info(
                self.env.get_step_result(brain_name),
                self.env.get_agent_group_spec(brain_name),
            )
        return all_brain_info
