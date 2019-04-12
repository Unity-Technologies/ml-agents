import unittest.mock as mock
from mlagents.trainers import TrainerMetrics


class TestTrainerMetrics:
    def test_field_names(self):
        field_names = [
            "Brain name",
            "Time to update policy",
            "Time since start of training",
            "Time for last experience collection",
            "Number of experiences used for training",
            "Mean return",
        ]
        from mlagents.trainers.trainer_metrics import FIELD_NAMES

        assert FIELD_NAMES == field_names

    @mock.patch(
        "mlagents.trainers.trainer_metrics.time", mock.MagicMock(return_value=42)
    )
    def test_experience_collection_timer(self):
        mock_path = "fake"
        mock_brain_name = "fake"
        trainer_metrics = TrainerMetrics(path=mock_path, brain_name=mock_brain_name)
        trainer_metrics.start_experience_collection_timer()
        trainer_metrics.end_experience_collection_timer()
        assert trainer_metrics.delta_last_experience_collection == 0

    @mock.patch(
        "mlagents.trainers.trainer_metrics.time", mock.MagicMock(return_value=42)
    )
    def test_policy_update_timer(self):
        mock_path = "fake"
        mock_brain_name = "fake"
        fake_buffer_length = 350
        fake_mean_return = 0.3
        trainer_metrics = TrainerMetrics(path=mock_path, brain_name=mock_brain_name)
        trainer_metrics.start_experience_collection_timer()
        trainer_metrics.end_experience_collection_timer()
        trainer_metrics.start_policy_update_timer(
            number_experiences=fake_buffer_length, mean_return=fake_mean_return
        )
        trainer_metrics.end_policy_update()
        fake_row = [mock_brain_name, 0, 0, 0, 350, "0.300"]
        assert trainer_metrics.rows[0] == fake_row
