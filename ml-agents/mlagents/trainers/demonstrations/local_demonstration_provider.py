from typing import List

from mlagents_envs.base_env import BehaviorSpec

from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.demonstrations.demonstration_provider import DemonstrationProvider
from mlagents.trainers.demonstrations.demonstration_proto_utils import load_demonstration


import os
from typing import List, Tuple
import numpy as np
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.rpc_utils import behavior_spec_from_proto, steps_from_proto
from mlagents_envs.base_env import BehaviorSpec
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
        behavior_spec, info_action_pair, _ = load_demonstration(demo_paths)


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
                raise ValueError("There are no '.demo' files in the provided directory.")
            return paths
        else:
            raise FileNotFoundError(
                f"The demonstration file or directory {path} does not exist."
            )
