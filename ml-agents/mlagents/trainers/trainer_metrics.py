# # Unity ML-Agents Toolkit
import logging
import csv
from time import time

LOGGER = logging.getLogger("mlagents.trainers")
FIELD_NAMES = [
    "Brain name",
    "Time to update policy",
    "Time since start of training",
    "Time for last experience collection",
    "Number of experiences used for training",
    "Mean return",
]


class TrainerMetrics:
    """
        Helper class to track, write training metrics. Tracks time since object
        of this class is initialized.
    """

    def __init__(self, path: str, brain_name: str):
        """
        :str path: Fully qualified path where CSV is stored.
        :str brain_name: Identifier for the Brain which we are training
        """
        self.path = path
        self.brain_name = brain_name
        self.rows = []
        self.time_start_experience_collection = None
        self.time_training_start = time()
        self.last_buffer_length = None
        self.last_mean_return = None
        self.time_policy_update_start = None
        self.delta_last_experience_collection = None
        self.delta_policy_update = None

    def start_experience_collection_timer(self):
        """
        Inform Metrics class that experience collection is starting. Intended to be idempotent
        """
        if self.time_start_experience_collection is None:
            self.time_start_experience_collection = time()

    def end_experience_collection_timer(self):
        """
        Inform Metrics class that experience collection is done.
        """
        if self.time_start_experience_collection:
            curr_delta = time() - self.time_start_experience_collection
            if self.delta_last_experience_collection is None:
                self.delta_last_experience_collection = curr_delta
            else:
                self.delta_last_experience_collection += curr_delta
        self.time_start_experience_collection = None

    def add_delta_step(self, delta: float):
        """
        Inform Metrics class about time to step in environment.
        """
        if self.delta_last_experience_collection:
            self.delta_last_experience_collection += delta
        else:
            self.delta_last_experience_collection = delta

    def start_policy_update_timer(self, number_experiences: int, mean_return: float):
        """
        Inform Metrics class that policy update has started.
        :int number_experiences: Number of experiences in Buffer at this point.
        :float mean_return: Return averaged across all cumulative returns since last policy update
        """
        self.last_buffer_length = number_experiences
        self.last_mean_return = mean_return
        self.time_policy_update_start = time()

    def _add_row(self, delta_train_start):
        row = [self.brain_name]
        row.extend(
            format(c, ".3f") if isinstance(c, float) else c
            for c in [
                self.delta_policy_update,
                delta_train_start,
                self.delta_last_experience_collection,
                self.last_buffer_length,
                self.last_mean_return,
            ]
        )
        self.delta_last_experience_collection = None
        self.rows.append(row)

    def end_policy_update(self):
        """
        Inform Metrics class that policy update has started.
        """
        if self.time_policy_update_start:
            self.delta_policy_update = time() - self.time_policy_update_start
        else:
            self.delta_policy_update = 0
        delta_train_start = time() - self.time_training_start
        LOGGER.debug(
            " Policy Update Training Metrics for {}: "
            "\n\t\tTime to update Policy: {:0.3f} s \n"
            "\t\tTime elapsed since training: {:0.3f} s \n"
            "\t\tTime for experience collection: {:0.3f} s \n"
            "\t\tBuffer Length: {} \n"
            "\t\tReturns : {:0.3f}\n".format(
                self.brain_name,
                self.delta_policy_update,
                delta_train_start,
                self.delta_last_experience_collection,
                self.last_buffer_length,
                self.last_mean_return,
            )
        )
        self._add_row(delta_train_start)

    def write_training_metrics(self):
        """
        Write Training Metrics to CSV
        """
        with open(self.path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(FIELD_NAMES)
            for row in self.rows:
                writer.writerow(row)
