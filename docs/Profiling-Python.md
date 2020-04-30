# Profiling in Python

As part of the ML-Agents Tookit, we provide a lightweight profiling system, in
order to identity hotspots in the training process and help spot regressions
from changes.

Timers are hierarchical, meaning that the time tracked in a block of code can be
further split into other blocks if desired. This also means that a function that
is called from multiple places in the code will appear in multiple places in the
timing output.

All timers operate using a "global" instance by default, but this can be
overridden if necessary (mainly for testing).

## Adding Profiling

There are two ways to indicate code should be included in profiling. The
simplest way is to add the `@timed` decorator to a function or method of
interested.

```python
class TrainerController:
    # ....
    @timed
    def advance(self, env: EnvManager) -> int:
        # do stuff
```

You can also used the `hierarchical_timer` context manager.

```python
with hierarchical_timer("communicator.exchange"):
    outputs = self.communicator.exchange(step_input)
```

The context manager may be easier than the `@timed` decorator for profiling
different parts of a large function, or profiling calls to abstract methods that
might not use decorator.

## Output

By default, at the end of training, timers are collected and written in json
format to `{summaries_dir}/{run_id}_timers.json`. The output consists of node
objects with the following keys:

- total (float): The total time in seconds spent in the block, including child
  calls.
- count (int): The number of times the block was called.
- self (float): The total time in seconds spent in the block, excluding child
  calls.
- children (dictionary): A dictionary of child nodes, keyed by the node name.
- is_parallel (bool): Indicates that the block of code was executed in multiple
  threads or processes (see below). This is optional and defaults to false.

### Parallel execution

#### Subprocesses

For code that executes in multiple processes (for example,
SubprocessEnvManager), we periodically send the timer information back to the
"main" process, aggregate the timers there, and flush them in the subprocess.
Note that (depending on the number of processes) this can result in timers where
the total time may exceed the parent's total time. This is analogous to the
difference between "real" and "user" values reported from the unix `time`
command. In the timer output, blocks that were run in parallel are indicated by
the `is_parallel` flag.

#### Threads

Timers currently use `time.perf_counter()` to track time spent, which may not
give accurate results for multiple threads. If this is problematic, set
`threaded: false` in your trainer configuration.
