from typing import Dict, Any

ML_AGENTS_STATS_WRITER = "mlagents.stats_writer"
ML_AGENTS_TRAINER_TYPE = "mlagents.trainer_type"

# TODO: the real type is Dict[str, HyperparamSettings]
all_trainer_types: Dict[str, Any] = {}
all_trainer_settings: Dict[str, Any] = {}
