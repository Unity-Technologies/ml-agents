import logging
import os
import gzip
import struct
import itertools
import numpy as np
from typing import List, Tuple, Iterator, Optional, TypeVar
from typing_extensions import Protocol
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.brain_conversion_utils import group_spec_to_brain_parameters
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents.trainers.trajectory import SplitObservations
from mlagents_envs.rpc_utils import (
    agent_group_spec_from_proto,
    batched_step_result_from_proto,
)
from mlagents_envs.base_env import AgentGroupSpec
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.timers import timed
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore

logger = logging.getLogger("mlagents.trainers")

T = TypeVar("T", covariant=True)


class SizedIterable(Protocol[T]):
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[T]:
        pass


class AgentInfoActionPairProtoDemoGenerator(SizedIterable[AgentInfoActionPairProto]):
    _gen: SizedIterable[AgentInfoActionPairProto]
    nested: Optional[SizedIterable[AgentInfoActionPairProto]]

    def __init__(
        self,
        file_path: str,
        starting_offset: int,
        total_expected: int,
        nested: Optional[SizedIterable[AgentInfoActionPairProto]] = None,
    ):
        self.file_path = file_path
        self.starting_offset = starting_offset
        self.total_expected = total_expected
        self.nested: Optional[SizedIterable[AgentInfoActionPairProto]] = nested

    def __len__(self) -> int:
        return self.total_expected + (0 if self.nested is None else len(self.nested))

    def _generator(self):
        return (
            itertools.chain(self._local_generator(), self.nested)
            if self.nested
            else self._local_generator()
        )

    def _local_generator(self) -> Iterator[AgentInfoActionPairProto]:
        count = 0
        is_compressed = self.file_path.endswith(".gz")
        file_size = 0

        if is_compressed:
            with open(self.file_path, "rb") as f:
                f.seek(-4, 2)
                file_size = struct.unpack("I", f.read(4))[0]
        else:
            file_size = os.path.getsize(self.file_path)

        with gzip.open(self.file_path) if self.file_path.endswith(".gz") else open(
            self.file_path, "rb"
        ) as fp:
            fp.seek(self.starting_offset, 0)
            while fp.tell() < file_size:
                size = fp.read(33)
                next_pos, offset = _DecodeVarint32(size, 0)
                fp.seek(-33 + offset, os.SEEK_CUR)
                data = fp.read(next_pos)
                agent_info_action = AgentInfoActionPairProto()
                agent_info_action.ParseFromString(data)
                count += 1
                yield agent_info_action
                if count == self.total_expected:
                    break

    def __iter__(self) -> Iterator[AgentInfoActionPairProto]:
        self._gen = self._generator()
        return self

    def __next__(self):
        return next(self._gen)


@timed
def make_demo_buffer(
    pair_infos: List[AgentInfoActionPairProto],
    group_spec: AgentGroupSpec,
    sequence_length: int,
) -> AgentBuffer:
    # Create and populate buffer using experiences
    demo_raw_buffer = AgentBuffer()
    demo_processed_buffer = AgentBuffer()
    # TODO: Rewrite below so this doesn't need to be stored in memory, adapt to agent processer via batched step results
    pair_infos = list(pair_infos)

    for idx, current_pair_info in enumerate(pair_infos):
        if idx > len(pair_infos) - 2:
            break
        next_pair_info = pair_infos[idx + 1]
        current_step_info = batched_step_result_from_proto(
            [current_pair_info.agent_info], group_spec
        )
        next_step_info = batched_step_result_from_proto(
            [next_pair_info.agent_info], group_spec
        )
        previous_action = (
            np.array(pair_infos[idx].action_info.vector_actions, dtype=np.float32) * 0
        )
        if idx > 0:
            previous_action = np.array(
                pair_infos[idx - 1].action_info.vector_actions, dtype=np.float32
            )
        curr_agent_id = current_step_info.agent_id[0]
        current_agent_step_info = current_step_info.get_agent_step_result(curr_agent_id)
        next_agent_id = next_step_info.agent_id[0]
        next_agent_step_info = next_step_info.get_agent_step_result(next_agent_id)

        demo_raw_buffer["done"].append(next_agent_step_info.done)
        demo_raw_buffer["rewards"].append(next_agent_step_info.reward)
        split_obs = SplitObservations.from_observations(current_agent_step_info.obs)
        for i, obs in enumerate(split_obs.visual_observations):
            demo_raw_buffer["visual_obs%d" % i].append(obs)
        demo_raw_buffer["vector_obs"].append(split_obs.vector_observations)
        demo_raw_buffer["actions"].append(current_pair_info.action_info.vector_actions)
        demo_raw_buffer["prev_action"].append(previous_action)
        if next_step_info.done:
            demo_raw_buffer.resequence_and_append(
                demo_processed_buffer, batch_size=None, training_length=sequence_length
            )
            demo_raw_buffer.reset_agent()
    demo_raw_buffer.resequence_and_append(
        demo_processed_buffer, batch_size=None, training_length=sequence_length
    )
    return demo_processed_buffer


@timed
def demo_to_buffer(
    file_path: str, sequence_length: int
) -> Tuple[BrainParameters, AgentBuffer]:
    """
    Loads demonstration file and uses it to fill training buffer.
    :param file_path: Location of demonstration file (.demo).
    :param sequence_length: Length of trajectories to fill buffer.
    :return:
    """
    group_spec, info_action_pair, _ = load_demonstration(file_path)
    demo_buffer = make_demo_buffer(info_action_pair, group_spec, sequence_length)
    brain_params = group_spec_to_brain_parameters("DemoBrain", group_spec)
    return brain_params, demo_buffer


def get_demo_files(path: str) -> List[str]:
    """
    Retrieves the demonstration file(s) from a path.
    :param path: Path of demonstration file or directory.
    :return: List of demonstration files

    Raises errors if |path| is invalid.
    """
    if os.path.isfile(path):
        if not (path.endswith(".demo") or path.endswith(".demo.gz")):
            raise ValueError("The path provided is not a '.demo' file.")
        return [path]
    elif os.path.isdir(path):
        paths = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".demo") or name.endswith(".demo.gz")
        ]
        if not paths:
            raise ValueError("There are no '.demo' files in the provided directory.")
        return paths
    else:
        raise FileNotFoundError(
            f"The demonstration file or directory {path} does not exist."
        )


@timed
def load_demonstration(
    file_path: str,
) -> Tuple[BrainParameters, SizedIterable[AgentInfoActionPairProto], int]:
    """
    Loads and parses a demonstration file.
    :param file_path: Location of demonstration file (.demo).
    :return: BrainParameter and list of AgentInfoActionPairProto containing demonstration data.
    """

    # First 32 bytes of file dedicated to meta-data.
    INITIAL_POS = 33
    file_paths = get_demo_files(file_path)
    brain_param_proto = None
    total_expected = 0
    info_action_pairs: Optional[SizedIterable[AgentInfoActionPairProto]] = None
    for _file_path in file_paths:
        is_compressed = _file_path.endswith(".gz")
        with gzip.open(_file_path) if is_compressed else open(_file_path, "rb") as fp:
            next_pos, obs_decoded = 0, 0
            expected_from_file = 0
            for obs_decoded in range(3):
                size = fp.read(33)
                next_pos, offset = _DecodeVarint32(size, 0)
                fp.seek(-33 + offset, os.SEEK_CUR)
                data = fp.read(next_pos)
                if obs_decoded == 0:
                    meta_data_proto = DemonstrationMetaProto()
                    meta_data_proto.ParseFromString(data)
                    expected_from_file = meta_data_proto.number_steps
                    total_expected += expected_from_file
                    fp.seek(INITIAL_POS, 0)  # Seek to magic header offset value
                if obs_decoded == 1:
                    brain_param_proto = BrainParametersProto()
                    brain_param_proto.ParseFromString(data)
                if (
                    obs_decoded > 1
                ):  # Read the first AgentInfoActionPairProto to get observation shape information
                    agent_info_action = AgentInfoActionPairProto()
                    agent_info_action.ParseFromString(data)
                    group_spec = agent_group_spec_from_proto(
                        brain_param_proto, agent_info_action.agent_info
                    )
                    fp.seek(
                        -(next_pos + offset), os.SEEK_CUR
                    )  # Rollback read and pass responsibility to AgentInfoActionPairProtoGenerator
                    info_action_pairs = AgentInfoActionPairProtoDemoGenerator(
                        _file_path, fp.tell(), expected_from_file, info_action_pairs
                    )

    if not group_spec:
        raise RuntimeError(
            f"No BrainParameters found in demonstration file at {file_path}."
        )
    if not info_action_pairs:
        raise RuntimeError(
            f"No AgentInfoActionPair found in demonstration file at {file_path}."
        )
    return group_spec, info_action_pairs, total_expected
