import sys
from typing import Dict, Tuple, Any

# importlib.metadata is new in python3.8
# We use the backport for older python versions.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata  # pylint: disable=E0611


from mlagents_envs import logging_util
from mlagents.plugins import ML_AGENTS_TRAINER_TYPE
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.poca.trainer import POCATrainer
from mlagents.trainers.ppo.optimizer_torch import PPOSettings
from mlagents.trainers.sac.optimizer_torch import SACSettings
from mlagents.trainers.poca.optimizer_torch import POCASettings
from mlagents import plugins as mla_plugins

logger = logging_util.get_logger(__name__)


def get_default_trainer_types() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    The Trainers that mlagents-learn always uses:
    """

    mla_plugins.all_trainer_types.update(
        {
            PPOTrainer.get_trainer_name(): PPOTrainer,
            SACTrainer.get_trainer_name(): SACTrainer,
            POCATrainer.get_trainer_name(): POCATrainer,
        }
    )
    # global all_trainer_settings
    mla_plugins.all_trainer_settings.update(
        {
            PPOTrainer.get_trainer_name(): PPOSettings,
            SACTrainer.get_trainer_name(): SACSettings,
            POCATrainer.get_trainer_name(): POCASettings,
        }
    )

    return mla_plugins.all_trainer_types, mla_plugins.all_trainer_settings


def register_trainer_plugins() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Registers all Trainer plugins (including the default one),
    and evaluates them, and returns the list of all the Trainer implementations.
    """
    if ML_AGENTS_TRAINER_TYPE not in importlib_metadata.entry_points():
        logger.warning(
            f"Unable to find any entry points for {ML_AGENTS_TRAINER_TYPE}, even the default ones. "
            "Uninstalling and reinstalling ml-agents via pip should resolve. "
            "Using default plugins for now."
        )
        return get_default_trainer_types()

    entry_points = importlib_metadata.entry_points()[ML_AGENTS_TRAINER_TYPE]

    for entry_point in entry_points:

        try:
            logger.debug(f"Initializing Trainer plugins: {entry_point.name}")
            plugin_func = entry_point.load()
            plugin_trainer_types, plugin_trainer_settings = plugin_func()
            logger.debug(
                f"Found {len(plugin_trainer_types)} Trainers for plugin {entry_point.name}"
            )
            mla_plugins.all_trainer_types.update(plugin_trainer_types)
            mla_plugins.all_trainer_settings.update(plugin_trainer_settings)
        except BaseException:
            # Catch all exceptions from setting up the plugin, so that bad user code doesn't break things.
            logger.exception(
                f"Error initializing Trainer plugins for {entry_point.name}. This plugin will not be used."
            )
    return mla_plugins.all_trainer_types, mla_plugins.all_trainer_settings
