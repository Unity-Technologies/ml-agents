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
        self.first_step_infos: List[EnvironmentStep] = None

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

    def reset(self, config: Dict = None) -> int:
        for manager in self.agent_managers.values():
            manager.end_episode()
        # Save the first step infos, after the reset.
        # They will be processed on the first advance().
        self.first_step_infos = self._reset_env(config)
        return len(self.first_step_infos)

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
        # If we had just reset, process the first EnvironmentSteps.
        # Note that we do it here instead of in reset() so that on the very first reset(),
        # we can create the needed AgentManagers before calling advance() and processing the EnvironmentSteps.
        if self.first_step_infos is not None:
            self._process_step_infos(self.first_step_infos)
            self.first_step_infos = None
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
                        name_behavior_id, ActionInfo.empty()
                    ),
                )
        return len(step_infos)
