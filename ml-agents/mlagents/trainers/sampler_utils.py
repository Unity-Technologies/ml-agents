import numpy as np
from enum import Enum
from typing import Dict, List

from mlagents.trainers.exception import SamplerException


class SamplerUtils:
    """
    Maintain a directory of available samplers and their configs.
    Validates sampler configs are correct.
    """

    NAME_TO_ARGS = {
        "uniform": ["min_value", "max_value"],
        "gaussian": ["mean", "st_dev"],
        "multirangeuniform": ["intervals"],
    }
    NAME_TO_FLOAT_REPR = {"uniform": 0.0, "gaussian": 1.0, "multirangeuniform": 2.0}

    @staticmethod
    def validate_and_structure_config(
        param: str, config: Dict[str, List[float]]
    ) -> List[float]:
        # Config must have a valid type
        if (
            "sampler-type" not in config
            or config["sampler-type"] not in SamplerUtils.NAME_TO_ARGS
        ):
            raise SamplerException(
                f"The sampler config for environment parameter {param} does not contain a sampler-type or the sampler-type is invalid."
            )
        # Check args are correct
        sampler_type = config.pop("sampler-type")
        if list(config.keys()) != SamplerUtils.NAME_TO_ARGS[sampler_type]:
            raise SamplerException(
                "The sampler config for environment parameter {} does not contain the correct arguments. Please specify {}.".format(
                    param, SamplerUtils.NAME_TO_ARGS[config["sampler-type"]]
                )
            )
        return [SamplerUtils.NAME_TO_FLOAT_REPR[sampler_type]] + list(config.values())
