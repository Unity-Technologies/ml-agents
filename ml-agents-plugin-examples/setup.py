from setuptools import setup

setup(
    name="mlagents_plugin_examples",
    version="0.0.1",
    entry_points={
        "mlagents.stats_writer": [
            "example=mlagents_plugin_examples.example_stats_writer:get_example_stats_writer"
        ]
    },
)
