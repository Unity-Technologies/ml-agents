import attr
import cattr
import yaml
from typing import Dict, Any
import argparse
from mlagents.trainers.settings import TrainerSettings, NetworkSettings, TrainerType
from mlagents.trainers.cli_utils import load_config


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
            trainer_type = config["trainer"]
            new_config = {}
            new_config["trainer_type"] = trainer_type
            hyperparam_cls = TrainerType(trainer_type).to_settings()
            # Try to absorb as much as possible into the hyperparam_cls
            new_config["hyperparameters"] = cattr.structure(config, hyperparam_cls)

            # Try to absorb as much as possible into the network settings
            new_config["network_settings"] = cattr.structure(config, NetworkSettings)
            # Deal with recurrent
            if config["use_recurrent"]:
                new_config["network_settings"].memory = NetworkSettings.MemorySettings(
                    sequence_length=config["sequence_length"],
                    memory_size=config["memory_size"],
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


def write_to_yaml_file(config: Dict[str, Any], output_config: str):
    unstructed_config = cattr.unstructure(config)
    unstructed_config = remove_nones(unstructed_config)
    with open(output_config, "w") as f:
        try:
            yaml.dump(unstructed_config, f, sort_keys=False)
        except TypeError:  # Older versions of pyyaml don't support sort_keys
            yaml.dump(unstructed_config, f)


def remove_nones(config: Dict[Any, Any]):
    new_config = {}
    for key, val in config.items():
        if isinstance(val, dict):
            new_config[key] = remove_nones(val)
        elif val is not None:
            new_config[key] = val
    return new_config


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "trainer_config_path",
        help="Path to old format (<=0.16.X) trainer configuration YAML.",
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
    print(
        f"Converting {args.trainer_config_path} and saving to {args.output_config_path}."
    )

    old_config = load_config(args.trainer_config_path)
    behavior_config_dict = convert_behaviors(old_config)
    full_config = {"behaviors": behavior_config_dict}

    # Convert curriculum and sampler. note that we don't validate these; if it was correct
    # before it should be correct now.
    if args.curriculum is not None:
        curriculum_config_dict = load_config(args.curriculum)
        full_config["curriculum"] = curriculum_config_dict

    if args.sampler is not None:
        sampler_config_dict = load_config(args.curriculum)
        full_config["parameter_randomization"] = sampler_config_dict

    write_to_yaml_file(full_config, args.output_config_path)
