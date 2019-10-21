"""
Lightweight, hierarchical timers for profiling sections of code.

Example:

@timed
def foo(t):
    time.sleep(t)

def main():
    for i in range(3):
        foo(i + 1)
    with hierarchical_timer("context"):
        foo(1)

    print(get_timer_tree())

This would produce a timer tree like
    (root)
        "foo"
        "context"
            "foo"

The total time and counts are tracked for each block of code; in this example "foo" and "context.foo" are considered
distinct blocks, and are tracked separately.

The decorator and contextmanager are equivalent; the context manager may be more useful if you want more control
over the timer name, or are splitting up multiple sections of a large function.
"""

import math
from time import perf_counter

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, TypeVar


class TimerNode:
    """
    Represents the time spent in a block of code.
    """

    __slots__ = ["children", "total", "count", "is_parallel"]

    def __init__(self):
        # Note that since dictionary keys are the node names, we don't explicitly store the name on the TimerNode.
        self.children: Dict[str, TimerNode] = {}
        self.total: float = 0.0
        self.count: int = 0
        self.is_parallel = False

    def get_child(self, name: str) -> "TimerNode":
        """
        Get the child node corresponding to the name (and create if it doesn't already exist).
        """
        child = self.children.get(name)
        if child is None:
            child = TimerNode()
            self.children[name] = child
        return child

    def add_time(self, elapsed: float) -> None:
        """
        Accumulate the time spent in the node (and increment the count).
        """
        self.total += elapsed
        self.count += 1

    def merge(
        self, other: "TimerNode", root_name: str = None, is_parallel: bool = True
    ) -> None:
        """
        Add the other node to this node, then do the same recursively on its children.
        :param other: The other node to merge
        :param root_name: Optional name of the root node being merged.
        :param is_parallel: Whether or not the code block was executed in parallel.
        :return:
        """
        if root_name:
            node = self.get_child(root_name)
        else:
            node = self

        node.total += other.total
        node.count += other.count
        node.is_parallel |= is_parallel
        for other_child_name, other_child_node in other.children.items():
            child = node.get_child(other_child_name)
            child.merge(other_child_node, is_parallel=is_parallel)


class GaugeNode:
    """
    Tracks the most recent value of a metric. This is analogous to gauges in statsd.
    """

    __slots__ = ["value", "min_value", "max_value", "count"]

    def __init__(self, value: float):
        self.value = value
        self.min_value = value
        self.max_value = value
        self.count = 1

    def update(self, new_value: float) -> None:
        self.min_value = min(self.min_value, new_value)
        self.max_value = max(self.max_value, new_value)
        self.value = new_value
        self.count += 1

    def as_dict(self) -> Dict[str, float]:
        return {
            "value": self.value,
            "min": self.min_value,
            "max": self.max_value,
            "count": self.count,
        }


class TimerStack:
    """
    Tracks all the time spent. Users shouldn't use this directly, they should use the contextmanager below to make
    sure that pushes and pops are already matched.
    """

    __slots__ = ["root", "stack", "start_time", "gauges"]

    def __init__(self):
        self.root = TimerNode()
        self.stack = [self.root]
        self.start_time = perf_counter()
        self.gauges: Dict[str, GaugeNode] = {}

    def reset(self):
        self.root = TimerNode()
        self.stack = [self.root]
        self.start_time = perf_counter()
        self.gauges: Dict[str, GaugeNode] = {}

    def push(self, name: str) -> TimerNode:
        """
        Called when entering a new block of code that is timed (e.g. with a contextmanager).
        """
        current_node: TimerNode = self.stack[-1]
        next_node = current_node.get_child(name)
        self.stack.append(next_node)
        return next_node

    def pop(self) -> None:
        """
        Called when exiting a new block of code that is timed (e.g. with a contextmanager).
        """
        self.stack.pop()

    def get_root(self) -> TimerNode:
        """
        Update the total time and count of the root name, and return it.
        """
        root = self.root
        root.total = perf_counter() - self.start_time
        root.count = 1
        return root

    def get_timing_tree(self, node: TimerNode = None) -> Dict[str, Any]:
        """
        Recursively build a tree of timings, suitable for output/archiving.
        """
        res: Dict[str, Any] = {}
        if node is None:
            # Special case the root - total is time since it was created, and count is 1
            node = self.get_root()
            res["name"] = "root"

            # Only output gauges at top level
            if self.gauges:
                res["gauges"] = self._get_gauges()

        res["total"] = node.total
        res["count"] = node.count

        if node.is_parallel:
            # Note when the block ran in parallel, so that it's less confusing that a timer is less that its children.
            res["is_parallel"] = True

        child_total = 0.0
        child_list = []
        for child_name, child_node in node.children.items():
            child_res: Dict[str, Any] = {
                "name": child_name,
                **self.get_timing_tree(child_node),
            }
            child_list.append(child_res)
            child_total += child_res["total"]

        # "self" time is total time minus all time spent on children
        res["self"] = max(0.0, node.total - child_total)
        if child_list:
            res["children"] = child_list

        return res

    def set_gauge(self, name: str, value: float) -> None:
        if math.isnan(value):
            return
        gauge_node = self.gauges.get(name)
        if gauge_node:
            gauge_node.update(value)
        else:
            self.gauges[name] = GaugeNode(value)

    def _get_gauges(self) -> List[Dict[str, Any]]:
        gauges = []
        for gauge_name, gauge_node in self.gauges.items():
            gauge_dict: Dict[str, Any] = {"name": gauge_name, **gauge_node.as_dict()}
            gauges.append(gauge_dict)
        return gauges


# Global instance of a TimerStack. This is generally all that we need for profiling, but you can potentially
# create multiple instances and pass them to the contextmanager
_global_timer_stack = TimerStack()


@contextmanager
def hierarchical_timer(name: str, timer_stack: TimerStack = None) -> Generator:
    """
    Creates a scoped timer around a block of code. This time spent will automatically be incremented when
    the context manager exits.
    """
    timer_stack = timer_stack or _global_timer_stack
    timer_node = timer_stack.push(name)
    start_time = perf_counter()

    try:
        # The wrapped code block will run here.
        yield timer_node
    finally:
        # This will trigger either when the context manager exits, or an exception is raised.
        # We'll accumulate the time, and the exception (if any) gets raised automatically.
        elapsed = perf_counter() - start_time
        timer_node.add_time(elapsed)
        timer_stack.pop()


# This is used to ensure the signature of the decorated function is preserved
# See also https://github.com/python/mypy/issues/3157
FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def timed(func: FuncT) -> FuncT:
    """
    Decorator for timing a function or method. The name of the timer will be the qualified name of the function.
    Usage:
        @timed
        def my_func(x, y):
            return x + y
    Note that because this doesn't take arguments, the global timer stack is always used.
    """

    def wrapped(*args, **kwargs):
        with hierarchical_timer(func.__qualname__):
            return func(*args, **kwargs)

    return wrapped  # type: ignore


def set_gauge(name: str, value: float, timer_stack: TimerStack = None) -> None:
    """
    Updates the value of the gauge (or creates it if it hasn't been set before).
    """
    timer_stack = timer_stack or _global_timer_stack
    timer_stack.set_gauge(name, value)


def get_timer_tree(timer_stack: TimerStack = None) -> Dict[str, Any]:
    """
    Return the tree of timings from the TimerStack as a dictionary (or the global stack if none is provided)
    """
    timer_stack = timer_stack or _global_timer_stack
    return timer_stack.get_timing_tree()


def get_timer_root(timer_stack: TimerStack = None) -> TimerNode:
    """
    Get the root TimerNode of the timer_stack (or the global TimerStack if not specified)
    """
    timer_stack = timer_stack or _global_timer_stack
    return timer_stack.get_root()


def reset_timers(timer_stack: TimerStack = None) -> None:
    """
    Reset the timer_stack (or the global TimerStack if not specified)
    """
    timer_stack = timer_stack or _global_timer_stack
    timer_stack.reset()
