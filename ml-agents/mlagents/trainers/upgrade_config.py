# NOTE: This upgrade script is a temporary measure for the transition between the old-format
# configuration file and the new format. It will be marked for deprecation once the
# Python CLI and configuration files are finalized, and removed the following release.

import attr
import cattr
import yaml
from typing import Dict, Any, Optional
import argparse
from mlagents.trainers.settings import TrainerSettings, NetworkSettings, TrainerType
from mlagents.trainers.cli_utils import load_config
from mlagents.trainers.exception import TrainerConfigError


# Take an existing trainer config (e.g. trainer_config.yaml) and turn it into the new format.
def convert_behaviors(old_trainer_config: Dict[str, Any]) -> Dict[str, Any]:
    all_behavior_config_dict = {}
    default_config = old_trainer_config.get("default", {})
    for behavior_name, config in old_trainer_config.items():
        if behavior_name != "default":
            config = default_config.copy()
            config.update(old_trainer_config[behavior_name])

            # Convert to split TrainerSettings, Hyperparameters, NetworkSettings
            # Set trainer_type and get appropriate hyperparameter settings
            try:
                trainer_type = config["trainer"]
            except KeyError:
                raise TrainerConfigError(
                    "Config doesn't specify a trainer type. "
                    "Please specify trainer: in your config."
                )
            new_config = {}
            new_config["trainer_type"] = trainer_type
            hyperparam_cls = TrainerType(trainer_type).to_settings()
            # Try to absorb as much as possible into the hyperparam_cls
            new_config["hyperparameters"] = cattr.structure(config, hyperparam_cls)

            # Try to absorb as much as possible into the network settings
            new_config["network_settings"] = cattr.structure(config, NetworkSettings)
            # Deal with recurrent
            try:
                if config["use_recurrent"]:
                    new_config[
                        "network_settings"
                    ].memory = NetworkSettings.MemorySettings(
                        sequence_length=config["sequence_length"],
                        memory_size=config["memory_size"],
                    )
            except KeyError:
                raise TrainerConfigError(
                    "Config doesn't specify use_recurrent. "
                    "Please specify true or false for use_recurrent in your config."
                )
            # Absorb the rest into the base TrainerSettings
            for key, val in config.items():
                if key in attr.fields_dict(TrainerSettings):
                    new_config[key] = val

            # Structure the whole thing
            all_behavior_config_dict[behavior_name] = cattr.structure(
                new_config, TrainerSettings
            )
    return all_behavior_config_dict


def write_to_yaml_file(unstructed_config: Dict[str, Any], output_config: str) -> None:
    with open(output_config, "w") as f:
        try:
            yaml.dump(unstructed_config, f, sort_keys=False)
        except TypeError:  # Older versions of pyyaml don't support sort_keys
            yaml.dump(unstructed_config, f)


def remove_nones(config: Dict[Any, Any]) -> Dict[str, Any]:
    new_config = {}
    for key, val in config.items():
        if isinstance(val, dict):
            new_config[key] = remove_nones(val)
        elif val is not None:
            new_config[key] = val
    return new_config


# Take a sampler from the old format and convert to new sampler structure
def convert_samplers(old_sampler_config: Dict[str, Any]) -> Dict[str, Any]:
    new_sampler_config: Dict[str, Any] = {}
    for parameter, parameter_config in old_sampler_config.items():
        if parameter == "resampling-interval":
            print(
                "resampling-interval is no longer necessary for parameter randomization and is being ignored."
            )
            continue
        new_sampler_config[parameter] = {}
        new_sampler_config[parameter]["sampler_type"] = parameter_config["sampler-type"]
        new_samp_parameters = dict(parameter_config)  # Copy dict
        new_samp_parameters.pop("sampler-type")
        new_sampler_config[parameter]["sampler_parameters"] = new_samp_parameters
    return new_sampler_config


def convert_samplers_and_curriculum(
    parameter_dict: Dict[str, Any], curriculum: Dict[str, Any]
) -> Dict[str, Any]:
    for key, sampler in parameter_dict.items():
        if "sampler_parameters" not in sampler:
            parameter_dict[key]["sampler_parameters"] = {}
        for argument in [
            "seed",
            "min_value",
            "max_value",
            "mean",
            "st_dev",
            "intervals",
        ]:
            if argument in sampler:
                parameter_dict[key]["sampler_parameters"][argument] = sampler[argument]
                parameter_dict[key].pop(argument)
    param_set = set(parameter_dict.keys())
    for behavior_name, behavior_dict in curriculum.items():
        measure = behavior_dict["measure"]
        min_lesson_length = behavior_dict.get("min_lesson_length", 1)
        signal_smoothing = behavior_dict.get("signal_smoothing", False)
        thresholds = behavior_dict["thresholds"]
        num_lessons = len(thresholds) + 1
        parameters = behavior_dict["parameters"]
        for param_name in parameters.keys():
            if param_name in param_set:
                print(
                    f"The parameter {param_name} has both a sampler and a curriculum. Will ignore curriculum"
                )
            else:
                param_set.add(param_name)
                parameter_dict[param_name] = {"curriculum": []}
                for lesson_index in range(num_lessons - 1):
                    parameter_dict[param_name]["curriculum"].append(
                        {
                            f"Lesson{lesson_index}": {
                                "completion_criteria": {
                                    "measure": measure,
                                    "behavior": behavior_name,
                                    "signal_smoothing": signal_smoothing,
                                    "min_lesson_length": min_lesson_length,
                                    "threshold": thresholds[lesson_index],
                                },
                                "value": parameters[param_name][lesson_index],
                            }
                        }
                    )
                lesson_index += 1  # This is the last lesson
                parameter_dict[param_name]["curriculum"].append(
                    {
                        f"Lesson{lesson_index}": {
                            "value": parameters[param_name][lesson_index]
                        }
                    }
                )
    return parameter_dict


def parse_args():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "trainer_config_path",
        help="Path to old format (<=0.18.X) trainer configuration YAML.",
    )
    argparser.add_argument(
        "--curriculum",
        help="Path to old format (<=0.16.X) curriculum configuration YAML.",
        default=None,
    )
    argparser.add_argument(
        "--sampler",
        help="Path to old format (<=0.16.X) parameter randomization configuration YAML.",
        default=None,
    )
    argparser.add_argument(
        "output_config_path", help="Path to write converted YAML file."
    )
    args = argparser.parse_args()
    return args


def convert(
    config: Dict[str, Any],
    old_curriculum: Optional[Dict[str, Any]],
    old_param_random: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if "behaviors" not in config:
        print("Config file format version :  version <= 0.16.X")
        behavior_config_dict = convert_behaviors(config)
        full_config = {"behaviors": behavior_config_dict}

        # Convert curriculum and sampler. note that we don't validate these; if it was correct
        # before it should be correct now.
        if old_curriculum is not None:
            full_config["curriculum"] = old_curriculum

        if old_param_random is not None:
            sampler_config_dict = convert_samplers(old_param_random)
            full_config["parameter_randomization"] = sampler_config_dict

        # Convert config to dict
        config = cattr.unstructure(full_config)
    if "curriculum" in config or "parameter_randomization" in config:
        print("Config file format version :  0.16.X < version <= 0.18.X")
        full_config = {"behaviors": config["behaviors"]}

        param_randomization = config.get("parameter_randomization", {})
        if "resampling-interval" in param_randomization:
            param_randomization.pop("resampling-interval")
        if len(param_randomization) > 0:
            # check if we use the old format sampler-type vs sampler_type
            if (
                "sampler-type"
                in param_randomization[list(param_randomization.keys())[0]]
            ):
                param_randomization = convert_samplers(param_randomization)

        full_config["environment_parameters"] = convert_samplers_and_curriculum(
            param_randomization, config.get("curriculum", {})
        )

        # Convert config to dict
        config = cattr.unstructure(full_config)
    return config


def main() -> None:
    args = parse_args()
    print(
        f"Converting {args.trainer_config_path} and saving to {args.output_config_path}."
    )

    old_config = load_config(args.trainer_config_path)
    curriculum_config_dict = None
    old_sampler_config_dict = None
    if args.curriculum is not None:
        curriculum_config_dict = load_config(args.curriculum)
    if args.sampler is not None:
        old_sampler_config_dict = load_config(args.sampler)
    new_config = convert(old_config, curriculum_config_dict, old_sampler_config_dict)
    unstructed_config = remove_nones(new_config)
    write_to_yaml_file(unstructed_config, args.output_config_path)


if __name__ == "__main__":
    main()
