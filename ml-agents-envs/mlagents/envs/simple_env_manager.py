from typing import Any, Dict, List

from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.env_manager import EnvManager, StepInfo, AgentStep
from mlagents.envs.brain import AgentInfo
from mlagents.envs.timers import timed
from mlagents.envs import ActionInfo, BrainParameters


class SimpleEnvManager(EnvManager):
    """
    Simple implementation of the EnvManager interface that only handles one BaseUnityEnvironment at a time.
    This is generally only useful for testing; see SubprocessEnvManager for a production-quality implementation.
    """

    def __init__(self, env: BaseUnityEnvironment):
        super().__init__()
        self.env = env
        self.previous_agent_steps: Dict[str, AgentStep] = {}
        self.previous_agent_action_infos: Dict[str, ActionInfo] = {}

    def step(self) -> List[AgentStep]:
        all_action_info = self._take_step(self.previous_agent_steps)

        if self.env.global_done:
            agent_infos = self.env.reset()
        else:
            actions = {}
            memories = {}
            texts = {}
            values = {}
            for brain_name, action_info in all_action_info.items():
                actions[brain_name] = action_info.action
                memories[brain_name] = action_info.memory
                texts[brain_name] = action_info.text
                values[brain_name] = action_info.value
            agent_infos = self.env.step(actions, memories, texts, values)

        steps: List[AgentStep] = []
        for agent_info in agent_infos:
            step = AgentStep(
                self.previous_agent_steps[agent_info.id].current_agent_info,
                agent_info,
                self.previous_agent_action_infos[agent_info.id],
            )
            self.previous_agent_steps[agent_info.id] = step
            steps.append(step)
        return steps

    def reset(
        self,
        config: Dict[str, float] = None,
        train_mode: bool = True,
        custom_reset_parameters: Any = None,
    ) -> List[AgentStep]:  # type: ignore
        agent_infos = self.env.reset(
            config=config,
            train_mode=train_mode,
            custom_reset_parameters=custom_reset_parameters,
        )
        steps: List[AgentStep] = []
        for agent_info in agent_infos:
            step = AgentStep(None, agent_info, None)
            self.previous_agent_steps[agent_info.id] = step
            steps.append(step)
        return steps

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        return self.env.external_brains

    @property
    def reset_parameters(self) -> Dict[str, float]:
        return self.env.reset_parameters

    def close(self):
        self.env.close()

    @timed
    def _take_step(self, agent_steps: Dict[str, AgentStep]) -> Dict[str, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        agent_steps_by_brain: Dict[str, List[AgentStep]] = {}
        for agent_id, agent_step in agent_steps.items():
            brain_name = agent_step.current_agent_info.brain_name
            if brain_name in agent_steps_by_brain.keys():
                agent_steps_by_brain[brain_name].append(agent_step)
            else:
                agent_steps_by_brain[brain_name] = [agent_step]
        for brain_name, brain_steps in agent_steps_by_brain.items():
            agent_infos = list(
                map(lambda a_step: a_step.current_agent_info, brain_steps)
            )
            all_action_info[brain_name] = self.policies[brain_name].get_action(
                agent_infos
            )
            for agent_idx, agent_info in enumerate(agent_infos):
                action_info: ActionInfo = all_action_info[brain_name]
                outputs_for_agent = get_per_agent_outputs(
                    action_info.outputs, agent_idx
                )
                self.previous_agent_action_infos[agent_info.id] = ActionInfo(
                    action_info.action[agent_idx],
                    action_info.memory[agent_idx]
                    if action_info.memory is not None
                    else None,
                    None,
                    action_info.value[agent_idx]
                    if action_info.value is not None
                    else None,
                    outputs_for_agent,
                )
        return all_action_info


def get_per_agent_outputs(outputs_dict, agent_idx):
    outputs_for_agent = {}
    for output_key, output_val in outputs_dict.items():
        if isinstance(output_val, dict):
            outputs_for_agent[output_key] = get_per_agent_outputs(output_val, agent_idx)
        elif hasattr(output_val, "__len__"):
            outputs_for_agent[output_key] = output_val[agent_idx]
        else:
            outputs_for_agent[output_key] = output_val
    return outputs_for_agent
