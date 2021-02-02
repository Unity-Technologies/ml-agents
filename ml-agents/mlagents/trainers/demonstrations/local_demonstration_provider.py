from typing import List, Optional

from mlagents_envs.base_env import BehaviorSpec, ActionSpec

from mlagents.trainers.trajectory import Trajectory, AgentExperience
from mlagents.trainers.demonstrations.demonstration_provider import (
    DemonstrationProvider,
)
from mlagents.trainers.demonstrations.demonstration_proto_utils import (
    load_demonstration,
)


import os
from typing import List, Tuple
import numpy as np
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.rpc_utils import behavior_spec_from_proto, steps_from_proto
from mlagents_envs.base_env import BehaviorSpec, ActionTuple
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.timers import timed, hierarchical_timer
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore
from google.protobuf.internal.encoder import _EncodeVarint  # type: ignore


class LocalDemonstrationProver(DemonstrationProvider):
    def __init__(self, file_path: str):
        super().__init__()
        self._trajectories: List[Trajectory] = []
        self._load(file_path)

    def get_behavior_spec(self) -> BehaviorSpec:
        pass

    def get_trajectories(self) -> List[Trajectory]:
        pass

    def _load(self, file_path: str) -> None:
        demo_paths = self._get_demo_files(file_path)
        behavior_spec, info_action_pairs, = load_demonstration(demo_paths)

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
        behavior_spec: BehaviorSpec,
        info_action_pairs: List[AgentInfoActionPairProto]
    ) -> List[Trajectory]:
        trajectories_out = []
        current_experiences = []
        previous_action: np.zeros(behavior_spec.action_spec.continuous_size, dtype=np.float32) # TODO or discrete?
        for pair in info_action_pairs:
            obs = None # TODO
            action_tuple = LocalDemonstrationProver._get_action_tuple(pair, behavior_spec.action_spec)
            action_mask = None
            if pair.agent_info.action_mask:
                # TODO 2D?
                action_mask = np.ndarray([bool(m) for m in pair.agent_info.action_mask], dtype=np.bool)

            exp = AgentExperience(
                obs=obs,
                reward=pair.agent_info.reward,
                done=pair.agent_info.done,
                action=action_tuple,
                action_probs=None,
                action_mask=action_mask,
                prev_action=previous_action,
                interrupted=pair.agent_info.max_step_reached,
                memory=None,
            )
            current_experiences.append(exp)
            previous_action = np.ndarray(pair.action_info.vector_actions_deprecated, dtype=np.float32)
            if pair.agent_info.done:
                trajectories_out.append(
                    Trajectory(steps=current_experiences, ne)
                )


    @staticmethod
    def _get_action_tuple(pair: AgentInfoActionPairProto, action_spec: ActionSpec) -> ActionTuple:
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
        continuous_np = np.ndarray(continuous_actions, dtype=np.float32) if continuous_actions else None
        discrete_np = np.ndarray(discrete_actions, dtype=np.float32) if discrete_actions else None
        return ActionTuple(continuous_np, discrete_np)
