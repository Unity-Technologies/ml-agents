from abc import ABC, abstractmethod
from typing import List, Dict, NamedTuple, Iterable, Tuple
from mlagents_envs.base_env import (
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    BehaviorName,
)
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod

from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.agent_processor import AgentManager, AgentManagerQueue
from mlagents.trainers.action_info import ActionInfo
from mlagents_envs.logging_util import get_logger

AllStepResult = Dict[BehaviorName, Tuple[DecisionSteps, TerminalSteps]]
AllGroupSpec = Dict[BehaviorName, BehaviorSpec]

logger = get_logger(__name__)


class EnvironmentStep(NamedTuple):
    current_all_step_result: AllStepResult
    worker_id: int
    brain_name_to_action_info: Dict[BehaviorName, ActionInfo]
    environment_stats: Dict[str, Tuple[float, StatsAggregationMethod]]

    @property
    def name_behavior_ids(self) -> Iterable[BehaviorName]:
        return self.current_all_step_result.keys()

    @staticmethod
    def empty(worker_id: int) -> "EnvironmentStep":
        return EnvironmentStep({}, worker_id, {}, {})


class EnvManager(ABC):
    def __init__(self):
        self.policies: Dict[BehaviorName, TFPolicy] = {}
        self.agent_managers: Dict[BehaviorName, AgentManager] = {}
        self.first_step_infos: List[EnvironmentStep] = None

    def set_policy(self, brain_name: BehaviorName, policy: TFPolicy) -> None:
        self.policies[brain_name] = policy
        if brain_name in self.agent_managers:
            self.agent_managers[brain_name].policy = policy

    def set_agent_manager(
        self, brain_name: BehaviorName, manager: AgentManager
    ) -> None:
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
    def external_brains(self) -> Dict[BehaviorName, BrainParameters]:
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
        # Get new policies if found. Always get the latest policy.
        for brain_name in self.external_brains:
            _policy = None
            try:
                # We make sure to empty the policy queue before continuing to produce steps.
                # This halts the trainers until the policy queue is empty.
                while True:
                    _policy = self.agent_managers[brain_name].policy_queue.get_nowait()
            except AgentManagerQueue.Empty:
                if _policy is not None:
                    self.set_policy(brain_name, _policy)
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
                decision_steps, terminal_steps = step_info.current_all_step_result[
                    name_behavior_id
                ]
                self.agent_managers[name_behavior_id].add_experiences(
                    decision_steps,
                    terminal_steps,
                    step_info.worker_id,
                    step_info.brain_name_to_action_info.get(
                        name_behavior_id, ActionInfo.empty()
                    ),
                )

                self.agent_managers[name_behavior_id].record_environment_stats(
                    step_info.environment_stats, step_info.worker_id
                )
        return len(step_infos)
