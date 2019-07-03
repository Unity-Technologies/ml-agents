# # Unity ML-Agents Toolkit
import logging
import csv
from time import time, perf_counter

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Generator
import json


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
        self.rows: List[List[Optional[str]]] = []
        self.time_start_experience_collection: Optional[float] = None
        self.time_training_start = time()
        self.last_buffer_length: Optional[int] = None
        self.last_mean_return: Optional[float] = None
        self.time_policy_update_start: Optional[float] = None
        self.delta_last_experience_collection: Optional[float] = None
        self.delta_policy_update: Optional[float] = None

    def start_experience_collection_timer(self) -> None:
        """
        Inform Metrics class that experience collection is starting. Intended to be idempotent
        """
        if self.time_start_experience_collection is None:
            self.time_start_experience_collection = time()

    def end_experience_collection_timer(self) -> None:
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

    def add_delta_step(self, delta: float) -> None:
        """
        Inform Metrics class about time to step in environment.
        """
        if self.delta_last_experience_collection:
            self.delta_last_experience_collection += delta
        else:
            self.delta_last_experience_collection = delta

    def start_policy_update_timer(
        self, number_experiences: int, mean_return: float
    ) -> None:
        """
        Inform Metrics class that policy update has started.
        :int number_experiences: Number of experiences in Buffer at this point.
        :float mean_return: Return averaged across all cumulative returns since last policy update
        """
        self.last_buffer_length = number_experiences
        self.last_mean_return = mean_return
        self.time_policy_update_start = time()

    def _add_row(self, delta_train_start: float) -> None:
        row: List[Optional[str]] = [self.brain_name]
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

    def end_policy_update(self) -> None:
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

    def write_training_metrics(self) -> None:
        """
        Write Training Metrics to CSV
        """
        with open(self.path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(FIELD_NAMES)
            for row in self.rows:
                writer.writerow(row)

        json_path = self.path.replace("csv", "hierarchy.json")
        with open(json_path, "w") as file:
            json.dump(get_timer_tree(), file, indent=2)


class TimerNode:
    __slots__ = ["children", "total", "count"]

    def __init__(self):
        self.children: Dict[str, "TimerNode"] = {}
        self.total: float = 0.0
        self.count: int = 0

    def get_child(self, name: str) -> "TimerNode":
        child = self.children.get(name)
        if child is None:
            child = TimerNode()
            self.children[name] = child
        return child

    def add_time(self, elapsed: float) -> None:
        self.total += elapsed
        self.count += 1


class TimerStack:
    __slots__ = ["root", "stack"]

    def __init__(self):
        self.root = TimerNode()
        self.stack = [self.root]

    def push(self, name: str) -> TimerNode:
        current: TimerNode = self.stack[-1]
        next_timer = current.get_child(name)
        self.stack.append(next_timer)
        return next_timer

    def pop(self) -> None:
        self.stack.pop()

    def get_timing_tree(self, node: TimerNode = None) -> Dict[str, Any]:
        if node is None:
            node = self.root

        res: Dict[str, Any] = {"total": node.total, "count": node.count}

        child_total = 0.0
        if node.children:
            res["children"] = []
            for child_name, child_node in node.children.items():
                child_res: Dict[str, Any] = {
                    "name": child_name,
                    **self.get_timing_tree(child_node),
                }
                res["children"].append(child_res)
                child_total += child_res["total"]

        # "self" time is total time minus all time spent on children
        res["self"] = max(0.0, node.total - child_total)

        return res


_global_timer_stack = TimerStack()


@contextmanager
def hierarchical_timer(
    name: str, timer_stack: TimerStack = _global_timer_stack
) -> Generator:
    timer_node = timer_stack.push(name)
    start_time = perf_counter()

    try:
        # The wrapped code block will run here.
        yield
    finally:
        # This will trigger either when the context manager exits, or an exception is raised.
        # We'll accumulate the time, and the exception (if any) gets raised automatically.
        elapsed = perf_counter() - start_time
        timer_node.add_time(elapsed)
        timer_stack.pop()


def get_timer_tree(timer_stack: TimerStack = _global_timer_stack) -> Dict[str, Any]:
    return timer_stack.get_timing_tree()
