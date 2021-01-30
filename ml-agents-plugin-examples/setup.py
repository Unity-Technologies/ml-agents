from setuptools import setup

setup(
    name="mlagents_plugin_examples",
    version="0.0.1",
    # Example of how to add your own registration functions that will be called
    # by mlagents-learn.
    #
    # Here, the get_example_stats_writer() function in mlagents_plugin_examples/example_stats_writer.py
    # will get registered with the "mlagents.stats_writer" plugin interface.
    entry_points={
        "mlagents.stats_writer": [
            "example=mlagents_plugin_examples.example_stats_writer:get_example_stats_writer"
        ]
    },
)
