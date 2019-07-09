# # Unity ML-Agents Toolkit
from time import perf_counter

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, TypeVar


class TimerNode:
    """
    Represents the time spent in a block of code.
    """

    __slots__ = ["children", "total", "count"]

    def __init__(self):
        self.children: Dict[str, "TimerNode"] = {}
        self.total: float = 0.0
        self.count: int = 0

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


class TimerStack:
    """
    Tracks all the time spent. Users shouldn't use this directly, they should use the contextmanager below to make
    sure that pushes and pops are already matched.
    """

    __slots__ = ["root", "stack"]

    def __init__(self):
        self.root = TimerNode()
        self.stack = [self.root]

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

    def get_timing_tree(self, node: TimerNode = None) -> Dict[str, Any]:
        """
        Recursively build a tree of timings, suitable for output/archiving.
        """
        if node is None:
            node = self.root

        res: Dict[str, Any] = {"total": node.total, "count": node.count}

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
        yield
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
    Decorator for timing a function or method.
    Usage:
        @timed
        def my_func(x, y):
            return x + y
    Note that because this doesn't take arguments, the global timer stack is always used.
    """

    def wrapped(*args, **kwargs):
        with hierarchical_timer(func.__name__):
            return func(*args, **kwargs)

    return wrapped  # type: ignore


def get_timer_tree(timer_stack: TimerStack = None) -> Dict[str, Any]:
    """
    Return the tree of timings from the TimerStack as a dictionary (or the global stack if none is provided)
    """
    timer_stack = timer_stack or _global_timer_stack
    return timer_stack.get_timing_tree()
