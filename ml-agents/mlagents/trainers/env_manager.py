from abc import ABC, abstractmethod
import logging
from typing import List, Dict, NamedTuple, Iterable
from mlagents_envs.base_env import BatchedStepResult, AgentGroupSpec, AgentGroup
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.agent_processor import AgentManager, AgentManagerQueue
from mlagents.trainers.action_info import ActionInfo

AllStepResult = Dict[AgentGroup, BatchedStepResult]
AllGroupSpec = Dict[AgentGroup, AgentGroupSpec]

logger = logging.getLogger("mlagents.trainers")


def get_global_agent_id(worker_id: int, agent_id: int) -> str:
    """
    Create an agent id that is unique across environment workers using the worker_id.
    """
    return f"${worker_id}-{agent_id}"


class EnvironmentStep(NamedTuple):
    current_all_step_result: AllStepResult
    worker_id: int
    brain_name_to_action_info: Dict[AgentGroup, ActionInfo]

    @property
    def name_behavior_ids(self) -> Iterable[AgentGroup]:
        return self.current_all_step_result.keys()

    @staticmethod
    def empty(worker_id: int) -> "EnvironmentStep":
        return EnvironmentStep({}, worker_id, {})


class EnvManager(ABC):
    def __init__(self):
        self.policies: Dict[AgentGroup, TFPolicy] = {}
        self.agent_managers: Dict[AgentGroup, AgentManager] = {}

    def set_policy(self, brain_name: AgentGroup, policy: TFPolicy) -> None:
        self.policies[brain_name] = policy
        if brain_name in self.agent_managers:
            self.agent_managers[brain_name].policy = policy

    def set_agent_manager(self, brain_name: AgentGroup, manager: AgentManager) -> None:
        self.agent_managers[brain_name] = manager

    @abstractmethod
    def _step(self) -> List[EnvironmentStep]:
        pass

    @abstractmethod
    def _reset_env(self, config: Dict = None) -> List[EnvironmentStep]:
        pass

    def reset(self, config: Dict = None) -> None:
        self._process_step_infos(self._reset_env(config))

    @property
    @abstractmethod
    def external_brains(self) -> Dict[AgentGroup, BrainParameters]:
        pass

    @property
    @abstractmethod
    def get_properties(self) -> Dict[AgentGroup, float]:
        pass

    @abstractmethod
    def close(self):
        pass

    def advance(self):
        # Get new policies if found
        for brain_name in self.external_brains:
            try:
                _policy = self.agent_managers[brain_name].policy_queue.get_nowait()
                self.set_policy(brain_name, _policy)
            except AgentManagerQueue.Empty:
                pass
        # Step the environment
        new_step_infos = self._step()
        # Add to AgentProcessor
        num_step_infos = self._process_step_infos(new_step_infos)
        return num_step_infos

    def _process_step_infos(self, step_infos: List[EnvironmentStep]) -> int:
        for step_info in step_infos:
            for name_behavior_id in step_info.name_behavior_ids:
                if name_behavior_id not in self.agent_managers:
                    logger.warning(
                        "Agent manager was not created for behavior id {}.".format(
                            name_behavior_id
                        )
                    )
                    continue
                self.agent_managers[name_behavior_id].add_experiences(
                    step_info.current_all_step_result[name_behavior_id],
                    step_info.worker_id,
                    step_info.brain_name_to_action_info.get(
                        name_behavior_id, ActionInfo([], [], {}, [])
                    ),
                )
        return len(step_infos)
