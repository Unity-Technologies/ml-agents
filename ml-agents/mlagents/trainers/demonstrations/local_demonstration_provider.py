import os
from typing import List
import numpy as np


from mlagents_envs.base_env import ActionTuple, BehaviorSpec, ActionSpec
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents_envs.rpc_utils import steps_from_proto


from mlagents.trainers.demonstrations.demonstration_provider import (
    DemonstrationProvider,
    DemonstrationExperience,
    DemonstrationTrajectory,
)
from mlagents.trainers.demonstrations.demonstration_proto_utils import (
    load_demonstration,
)


class LocalDemonstrationProvider(DemonstrationProvider):
    def __init__(self, file_path: str):
        super().__init__()

        demo_paths = self._get_demo_files(file_path)
        behavior_spec, info_action_pairs, = load_demonstration(demo_paths)
        self._behavior_spec = behavior_spec
        self._info_action_pairs = info_action_pairs

    def get_behavior_spec(self) -> BehaviorSpec:
        return self._behavior_spec

    def pop_trajectories(self) -> List[DemonstrationTrajectory]:
        trajectories = LocalDemonstrationProvider._info_action_pairs_to_trajectories(
            self._behavior_spec, self._info_action_pairs
        )
        self._info_action_pairs = []
        return trajectories

    @staticmethod
    def _get_demo_files(path: str) -> List[str]:
        """
        Retrieves the demonstration file(s) from a path.
        :param path: Path of demonstration file or directory.
        :return: List of demonstration files

        Raises errors if |path| is invalid.
        """
        if os.path.isfile(path):
            if not path.endswith(".demo"):
                raise ValueError("The path provided is not a '.demo' file.")
            return [path]
        elif os.path.isdir(path):
            paths = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.endswith(".demo")
            ]
            if not paths:
                raise ValueError(
                    "There are no '.demo' files in the provided directory."
                )
            return paths
        else:
            raise FileNotFoundError(
                f"The demonstration file or directory {path} does not exist."
            )

    @staticmethod
    def _info_action_pairs_to_trajectories(
        behavior_spec: BehaviorSpec, info_action_pairs: List[AgentInfoActionPairProto]
    ) -> List[DemonstrationTrajectory]:
        trajectories_out: List[DemonstrationTrajectory] = []
        current_experiences = []
        previous_action = np.zeros(
            behavior_spec.action_spec.continuous_size, dtype=np.float32
        )  # TODO or discrete?
        for pair_index, pair in enumerate(info_action_pairs):

            # Extract the observations from the decision/terminal steps
            current_decision_step, current_terminal_step = steps_from_proto(
                [pair.agent_info], behavior_spec
            )
            if len(current_terminal_step) == 1:
                obs = list(current_terminal_step.values())[0].obs
            else:
                obs = list(current_decision_step.values())[0].obs

            action_tuple = LocalDemonstrationProvider._get_action_tuple(
                pair, behavior_spec.action_spec
            )

            exp = DemonstrationExperience(
                obs=obs,
                reward=pair.agent_info.reward,  # TODO next step's reward?
                done=pair.agent_info.done,
                action=action_tuple,
                prev_action=previous_action,
                interrupted=pair.agent_info.max_step_reached,
            )
            current_experiences.append(exp)
            previous_action = np.array(
                pair.action_info.vector_actions_deprecated, dtype=np.float32
            )
            if pair.agent_info.done or pair_index == len(info_action_pairs) - 1:
                trajectories_out.append(
                    DemonstrationTrajectory(experiences=current_experiences)
                )
                current_experiences = []

        return trajectories_out

    @staticmethod
    def _get_action_tuple(
        pair: AgentInfoActionPairProto, action_spec: ActionSpec
    ) -> ActionTuple:
        continuous_actions = None
        discrete_actions = None

        if (
            len(pair.action_info.continuous_actions) == 0
            and len(pair.action_info.discrete_actions) == 0
        ):
            if action_spec.continuous_size > 0:
                continuous_actions = pair.action_info.vector_actions_deprecated
            else:
                discrete_actions = pair.action_info.vector_actions_deprecated
        else:
            if action_spec.continuous_size > 0:
                continuous_actions = pair.action_info.continuous_actions
            if action_spec.discrete_size > 0:
                discrete_actions = pair.action_info.discrete_actions

        # TODO 2D?
        continuous_np = (
            np.array(continuous_actions, dtype=np.float32)
            if continuous_actions
            else None
        )
        discrete_np = (
            np.array(discrete_actions, dtype=np.float32) if discrete_actions else None
        )

        return ActionTuple(continuous_np, discrete_np)
