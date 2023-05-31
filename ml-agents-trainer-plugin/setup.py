from setuptools import setup
from mlagents.plugins import ML_AGENTS_TRAINER_TYPE

setup(
    name="mlagents_trainer_plugin",
    version="0.0.1",
    # Example of how to add your own registration functions that will be called
    # by mlagents-learn.
    #
    # Here, the get_example_stats_writer() function in mlagents_plugin_examples/example_stats_writer.py
    # will get registered with the ML_AGENTS_STATS_WRITER plugin interface.
    entry_points={
        ML_AGENTS_TRAINER_TYPE: [
            "a2c=mlagents_trainer_plugin.a2c.a2c_trainer:get_type_and_setting",
            "dqn=mlagents_trainer_plugin.dqn.dqn_trainer:get_type_and_setting",
        ]
    },
)
