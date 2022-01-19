# Customizing Training via Plugins

ML-Agents provides support for running your own python implementations of specific interfaces during the training
process. These interfaces are currently fairly limited, but will be expanded in the future.

**Note:** Plugin interfaces should currently be considered "in beta", and they may change in future releases.

## How to Write Your Own Plugin
[This video](https://www.youtube.com/watch?v=fY3Y_xPKWNA) explains the basics of how to create a plugin system using
setuptools, and is the same approach that ML-Agents' plugin system is based on.

The `ml-agents-plugin-examples` directory contains a reference implementation of each plugin interface, so it's a good
starting point.

### setup.py
If you don't already have a `setup.py` file for your python code, you'll need to add one. `ml-agents-plugin-examples`
has a [minimal example](../ml-agents-plugin-examples/setup.py) of this.

In the call to `setup()`, you'll need to add to the `entry_points` dictionary for each plugin interface that you
implement. The form of this is `{entry point name}={plugin module}:{plugin function}`. For example, in
 `ml-agents-plugin-examples`:
```python
entry_points={
    ML_AGENTS_STATS_WRITER: [
        "example=mlagents_plugin_examples.example_stats_writer:get_example_stats_writer"
    ]
}
```
* `ML_AGENTS_STATS_WRITER` (which is a string constant, `mlagents.stats_writer`) is the name of the plugin interface.
This must be one of the provided interfaces ([see below](#plugin-interfaces)).
* `example` is the plugin implementation name. This can be anything.
* `mlagents_plugin_examples.example_stats_writer` is the plugin module. This points to the module where the
plugin registration function is defined.
* `get_example_stats_writer` is the plugin registration function. This is called when running `mlagents-learn`. The
arguments and expected return type for this are different for each plugin interface.

### Local Installation
Once you've defined `entry_points` in your `setup.py`, you will need to run
```
pip install -e [path to your plugin code]
```
in the same python virtual environment that you have `mlagents` installed.

## Plugin Interfaces

### StatsWriter
The StatsWriter class receives various information from the training process, such as the average Agent reward in
each summary period. By default, we log this information to the console and write it to
[TensorBoard](Using-Tensorboard.md).

#### Interface
The `StatsWriter.write_stats()` method must be implemented in any derived classes. It takes a "category" parameter,
which typically is the behavior name of the Agents being trained, and a dictionary of `StatSummary` values with
string keys. Additionally, `StatsWriter.on_add_stat()` may be extended to register a callback handler for each stat
emission.

#### Registration
The `StatsWriter` registration function takes a `RunOptions` argument and returns a list of `StatsWriter`s. An
example implementation is provided in [`mlagents_plugin_examples`](../ml-agents-plugin-examples/mlagents_plugin_examples/example_stats_writer.py)
