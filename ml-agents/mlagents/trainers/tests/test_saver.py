import pytest
import os
import unittest
import tempfile

import numpy as np
from mlagents.tf_utils import tf



VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 32
NUM_AGENTS = 12


# def test_load_save(tmp_path):
#     path1 = os.path.join(tmp_path, "runid1")
#     path2 = os.path.join(tmp_path, "runid2")
#     trainer_params = TrainerSettings()
#     policy = create_policy_mock(trainer_params, model_path=path1)
#     policy.initialize_or_load()
#     policy._set_step(2000)

#     mock_brain_name = "MockBrain"
#     checkpoint_path = f"{policy.model_path}/{mock_brain_name}-2000"
#     serialization_settings = SerializationSettings(policy.model_path, mock_brain_name)
#     policy.checkpoint(checkpoint_path, serialization_settings)

#     assert len(os.listdir(tmp_path)) > 0

#     # Try load from this path
#     policy2 = create_policy_mock(trainer_params, model_path=path1, load=True, seed=1)
#     policy2.initialize_or_load()
#     _compare_two_policies(policy, policy2)
#     assert policy2.get_current_step() == 2000

#     # Try initialize from path 1
#     trainer_params.output_path = path2
#     trainer_params.init_path = path1
#     policy3 = create_policy_mock(trainer_params, model_path=path1, load=False, seed=2)
#     policy3.initialize_or_load()

#     _compare_two_policies(policy2, policy3)
#     # Assert that the steps are 0.
#     assert policy3.get_current_step() == 0


# class ModelVersionTest(unittest.TestCase):
#     def test_version_compare(self):
#         # Test write_stats
#         with self.assertLogs("mlagents.trainers", level="WARNING") as cm:
#             path1 = tempfile.mkdtemp()
#             trainer_params = TrainerSettings()
#             policy = create_policy_mock(trainer_params, model_path=path1)
#             policy.initialize_or_load()
#             policy._check_model_version(
#                 "0.0.0"
#             )  # This is not the right version for sure
#             # Assert that 1 warning has been thrown with incorrect version
#             assert len(cm.output) == 1
#             policy._check_model_version(__version__)  # This should be the right version
#             # Assert that no additional warnings have been thrown wth correct ver
#             assert len(cm.output) == 1


# if __name__ == "__main__":
#     pytest.main()
