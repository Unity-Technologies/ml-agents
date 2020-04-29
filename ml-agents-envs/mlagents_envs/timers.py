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
import sys
import time
import threading

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

TIMER_FORMAT_VERSION = "0.1.0"


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

    __slots__ = ["value", "min_value", "max_value", "count", "_timestamp"]

    def __init__(self, value: float):
        self.value = value
        self.min_value = value
        self.max_value = value
        self.count = 1
        # Internal timestamp so we can determine priority.
        self._timestamp = time.time()

    def update(self, new_value: float) -> None:
        self.min_value = min(self.min_value, new_value)
        self.max_value = max(self.max_value, new_value)
        self.value = new_value
        self.count += 1
        self._timestamp = time.time()

    def merge(self, other: "GaugeNode") -> None:
        if self._timestamp < other._timestamp:
            # Keep the "later" value
            self.value = other.value
            self._timestamp = other._timestamp
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)
        self.count += other.count

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

    __slots__ = ["root", "stack", "start_time", "gauges", "metadata"]

    def __init__(self):
        self.root = TimerNode()
        self.stack = [self.root]
        self.start_time = time.perf_counter()
        self.gauges: Dict[str, GaugeNode] = {}
        self.metadata: Dict[str, str] = {}
        self._add_default_metadata()

    def reset(self):
        self.root = TimerNode()
        self.stack = [self.root]
        self.start_time = time.perf_counter()
        self.gauges: Dict[str, GaugeNode] = {}
        self.metadata: Dict[str, str] = {}
        self._add_default_metadata()

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
        root.total = time.perf_counter() - self.start_time
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

            if self.metadata:
                self.metadata["end_time_seconds"] = str(int(time.time()))
                res["metadata"] = self.metadata

        res["total"] = node.total
        res["count"] = node.count

        if node.is_parallel:
            # Note when the block ran in parallel, so that it's less confusing that a timer is less that its children.
            res["is_parallel"] = True

        child_total = 0.0
        child_dict = {}
        for child_name, child_node in node.children.items():
            child_res: Dict[str, Any] = self.get_timing_tree(child_node)
            child_dict[child_name] = child_res
            child_total += child_res["total"]

        # "self" time is total time minus all time spent on children
        res["self"] = max(0.0, node.total - child_total)
        if child_dict:
            res["children"] = child_dict

        return res

    def set_gauge(self, name: str, value: float) -> None:
        if math.isnan(value):
            return
        gauge_node = self.gauges.get(name)
        if gauge_node:
            gauge_node.update(value)
        else:
            self.gauges[name] = GaugeNode(value)

    def add_metadata(self, key: str, value: str) -> None:
        self.metadata[key] = value

    def _get_gauges(self) -> Dict[str, Dict[str, float]]:
        gauges = {}
        for gauge_name, gauge_node in self.gauges.items():
            gauges[gauge_name] = gauge_node.as_dict()
        return gauges

    def _add_default_metadata(self):
        self.metadata["timer_format_version"] = TIMER_FORMAT_VERSION
        self.metadata["start_time_seconds"] = str(int(time.time()))
        self.metadata["python_version"] = sys.version
        self.metadata["command_line_arguments"] = " ".join(sys.argv)


# Maintain a separate "global" timer per thread, so that they don't accidentally conflict with each other.
_thread_timer_stacks: Dict[int, TimerStack] = {}


def _get_thread_timer() -> TimerStack:
    ident = threading.get_ident()
    if ident not in _thread_timer_stacks:
        timer_stack = TimerStack()
        _thread_timer_stacks[ident] = timer_stack
    return _thread_timer_stacks[ident]


def get_timer_stack_for_thread(t: threading.Thread) -> Optional[TimerStack]:
    if t.ident is None:
        # Thread hasn't started, shouldn't ever happen
        return None
    return _thread_timer_stacks.get(t.ident)


@contextmanager
def hierarchical_timer(name: str, timer_stack: TimerStack = None) -> Generator:
    """
    Creates a scoped timer around a block of code. This time spent will automatically be incremented when
    the context manager exits.
    """
    timer_stack = timer_stack or _get_thread_timer()
    timer_node = timer_stack.push(name)
    start_time = time.perf_counter()

    try:
        # The wrapped code block will run here.
        yield timer_node
    finally:
        # This will trigger either when the context manager exits, or an exception is raised.
        # We'll accumulate the time, and the exception (if any) gets raised automatically.
        elapsed = time.perf_counter() - start_time
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
    timer_stack = timer_stack or _get_thread_timer()
    timer_stack.set_gauge(name, value)


def merge_gauges(gauges: Dict[str, GaugeNode], timer_stack: TimerStack = None) -> None:
    """
    Merge the gauges from another TimerStack with the provided one (or the
    current thread's stack if none is provided).
    :param gauges:
    :param timer_stack:
    :return:
    """
    timer_stack = timer_stack or _get_thread_timer()
    for n, g in gauges.items():
        if n in timer_stack.gauges:
            timer_stack.gauges[n].merge(g)
        else:
            timer_stack.gauges[n] = g


def add_metadata(key: str, value: str, timer_stack: TimerStack = None) -> None:
    timer_stack = timer_stack or _get_thread_timer()
    timer_stack.add_metadata(key, value)


def get_timer_tree(timer_stack: TimerStack = None) -> Dict[str, Any]:
    """
    Return the tree of timings from the TimerStack as a dictionary (or the
     current thread's  stack if none is provided)
    """
    timer_stack = timer_stack or _get_thread_timer()
    return timer_stack.get_timing_tree()


def get_timer_root(timer_stack: TimerStack = None) -> TimerNode:
    """
    Get the root TimerNode of the timer_stack (or the current thread's
    TimerStack if not specified)
    """
    timer_stack = timer_stack or _get_thread_timer()
    return timer_stack.get_root()


def reset_timers(timer_stack: TimerStack = None) -> None:
    """
    Reset the timer_stack (or the current thread's TimerStack if not specified)
    """
    timer_stack = timer_stack or _get_thread_timer()
    timer_stack.reset()
