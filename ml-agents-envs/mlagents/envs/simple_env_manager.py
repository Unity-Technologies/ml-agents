from typing import Any, Dict, List

from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.env_manager import EnvManager, EnvironmentStep
from mlagents.envs.timers import timed
from mlagents.envs.action_info import ActionInfo
from mlagents.envs.brain import BrainParameters


class SimpleEnvManager(EnvManager):
    """
    Simple implementation of the EnvManager interface that only handles one BaseUnityEnvironment at a time.
    This is generally only useful for testing; see SubprocessEnvManager for a production-quality implementation.
    """

    def __init__(self, env: BaseUnityEnvironment):
        super().__init__()
        self.env = env
        self.previous_step: EnvironmentStep = EnvironmentStep(None, {}, None)
        self.previous_all_action_info: Dict[str, ActionInfo] = {}

    def step(self) -> List[EnvironmentStep]:

        all_action_info = self._take_step(self.previous_step)
        self.previous_all_action_info = all_action_info

        actions = {}
        memories = {}
        texts = {}
        values = {}
        for brain_name, action_info in all_action_info.items():
            actions[brain_name] = action_info.action
            memories[brain_name] = action_info.memory
            texts[brain_name] = action_info.text
            values[brain_name] = action_info.value
        all_brain_info = self.env.step(actions, memories, texts, values)
        step_brain_info = all_brain_info

        step_info = EnvironmentStep(
            self.previous_step.current_all_brain_info,
            step_brain_info,
            self.previous_all_action_info,
        )
        self.previous_step = step_info
        return [step_info]

    def reset(
        self,
        config: Dict[str, float] = None,
        train_mode: bool = True,
        custom_reset_parameters: Any = None,
    ) -> List[EnvironmentStep]:  # type: ignore
        all_brain_info = self.env.reset(
            config=config,
            train_mode=train_mode,
            custom_reset_parameters=custom_reset_parameters,
        )
        self.previous_step = EnvironmentStep(None, all_brain_info, None)
        return [self.previous_step]

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        return self.env.external_brains

    @property
    def reset_parameters(self) -> Dict[str, float]:
        return self.env.reset_parameters

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
