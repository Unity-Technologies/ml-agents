import argparse
from typing import Optional, List
from mlagents.trainers.learn import RunOptions, run_cli, load_config


def parse_command_line(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("experiment_config_path")
    return parser.parse_args(argv)


def main():
    """
    Provides an alternative CLI interface to mlagents-learn, 'mlagents-run-experiment'.
    Accepts a JSON/YAML formatted mlagents.trainers.learn.RunOptions object, and executes
    the run loop as defined in mlagents.trainers.learn.run_cli.
    """
    args = parse_command_line()
    expt_config = load_config(args.experiment_config_path)
    run_cli(RunOptions(**expt_config))


if __name__ == "__main__":
    main()
