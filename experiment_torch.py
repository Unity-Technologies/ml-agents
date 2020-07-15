import json
import os
import torch
import tensorflow as tf
import argparse
from mlagents.trainers.learn import run_cli, parse_command_line
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.ppo.trainer import TestingConfiguration
from mlagents_envs.timers import _thread_timer_stacks


def run_experiment(
    name: str,
    steps: int,
    use_torch: bool,
    algo: str,
    num_torch_threads: int,
    use_gpu: bool,
    num_envs: int = 1,
    config_name=None,
):
    TestingConfiguration.env_name = name
    TestingConfiguration.max_steps = steps
    TestingConfiguration.use_torch = use_torch
    TestingConfiguration.device = "cuda:0" if use_gpu else "cpu"
    if use_gpu:
        tf.device("/GPU:0")
    else:
        tf.device("/device:CPU:0")
    if not torch.cuda.is_available() and use_gpu:
        return (
            name,
            str(steps),
            str(use_torch),
            algo,
            str(num_torch_threads),
            str(num_envs),
            str(use_gpu),
            "na",
            "na",
            "na",
            "na",
            "na",
            "na",
            "na",
        )
    if config_name is None:
        config_name = name
    run_options = parse_command_line(
        [f"config/{algo}/{config_name}.yaml", "--num-envs", f"{num_envs}"]
    )
    run_options.checkpoint_settings.run_id = (
        f"{name}_test_" + str(steps) + "_" + ("torch" if use_torch else "tf")
    )
    run_options.checkpoint_settings.force = True
    # run_options.env_settings.num_envs = num_envs
    for trainer_settings in run_options.behaviors.values():
        trainer_settings.threaded = False
    timers_path = os.path.join(
        "results", run_options.checkpoint_settings.run_id, "run_logs", "timers.json"
    )
    if use_torch:
        torch.set_num_threads(num_torch_threads)
    run_cli(run_options)
    StatsReporter.writers.clear()
    StatsReporter.stats_dict.clear()
    _thread_timer_stacks.clear()
    with open(timers_path) as timers_json_file:
        timers_json = json.load(timers_json_file)
        total = timers_json["total"]
        tc_advance = timers_json["children"]["TrainerController.start_learning"][
            "children"
        ]["TrainerController.advance"]
        evaluate = timers_json["children"]["TrainerController.start_learning"][
            "children"
        ]["TrainerController.advance"]["children"]["env_step"]["children"][
            "SubprocessEnvManager._take_step"
        ][
            "children"
        ]
        update = timers_json["children"]["TrainerController.start_learning"][
            "children"
        ]["TrainerController.advance"]["children"]["trainer_advance"]["children"][
            "_update_policy"
        ][
            "children"
        ]
        tc_advance_total = tc_advance["total"]
        tc_advance_count = tc_advance["count"]
    if use_torch:
        if algo == "ppo":
            update_total = update["TorchPPOOptimizer.update"]["total"]
            update_count = update["TorchPPOOptimizer.update"]["count"]
        else:
            update_total = update["SACTrainer._update_policy"]["total"]
            update_count = update["SACTrainer._update_policy"]["count"]
        evaluate_total = evaluate["TorchPolicy.evaluate"]["total"]
        evaluate_count = evaluate["TorchPolicy.evaluate"]["count"]
    else:
        if algo == "ppo":
            update_total = update["TFPPOOptimizer.update"]["total"]
            update_count = update["TFPPOOptimizer.update"]["count"]
        else:
            update_total = update["SACTrainer._update_policy"]["total"]
            update_count = update["SACTrainer._update_policy"]["count"]
        evaluate_total = evaluate["NNPolicy.evaluate"]["total"]
        evaluate_count = evaluate["NNPolicy.evaluate"]["count"]
    # todo: do total / count
    return (
        name,
        str(steps),
        str(use_torch),
        algo,
        str(num_torch_threads),
        str(num_envs),
        str(use_gpu),
        str(total),
        str(tc_advance_total),
        str(tc_advance_count),
        str(update_total),
        str(update_count),
        str(evaluate_total),
        str(evaluate_count),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=25000, type=int, help="The number of steps")
    parser.add_argument("--num-envs", default=1, type=int, help="The number of envs")
    parser.add_argument(
        "--gpu", default=False, action="store_true", help="If true, will use the GPU"
    )
    parser.add_argument(
        "--threads",
        default=False,
        action="store_true",
        help="If true, will try both 1 and 8 threads for torch",
    )
    parser.add_argument(
        "--ball",
        default=False,
        action="store_true",
        help="If true, will only do 3dball",
    )
    parser.add_argument(
        "--sac",
        default=False,
        action="store_true",
        help="If true, will run sac instead of ppo",
    )
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    envs_config_tuples = [
        ("3DBall", "3DBall"),
        ("GridWorld", "GridWorld"),
        ("PushBlock", "PushBlock"),
        ("Hallway", "Hallway"),
        ("CrawlerStaticTarget", "CrawlerStatic"),
        ("VisualHallway", "VisualHallway"),
    ]
    if args.ball:
        envs_config_tuples = [("3DBall", "3DBall")]

    algo = "ppo"
    if args.sac:
        algo = "sac"

    labels = (
        "name",
        "steps",
        "use_torch",
        "algorithm",
        "num_torch_threads",
        "num_envs",
        "use_gpu",
        "total",
        "tc_advance_total",
        "tc_advance_count",
        "update_total",
        "update_count",
        "evaluate_total",
        "evaluate_count",
    )

    results = []
    results.append(labels)
    f = open(
        f"result_data_steps_{args.steps}_algo_{algo}_envs_{args.num_envs}_gpu_{args.gpu}_thread_{args.threads}.txt",
        "w",
    )
    f.write(" ".join(labels) + "\n")

    for env_config in envs_config_tuples:
        data = run_experiment(
            name=env_config[0],
            steps=args.steps,
            use_torch=True,
            algo=algo,
            num_torch_threads=1,
            use_gpu=args.gpu,
            num_envs=args.num_envs,
            config_name=env_config[1],
        )
        results.append(data)
        f.write(" ".join(data) + "\n")

        if args.threads:
            data = run_experiment(
                name=env_config[0],
                steps=args.steps,
                use_torch=True,
                algo=algo,
                num_torch_threads=8,
                use_gpu=args.gpu,
                num_envs=args.num_envs,
                config_name=env_config[1],
            )
            results.append(data)
            f.write(" ".join(data) + "\n")

        data = run_experiment(
            name=env_config[0],
            steps=args.steps,
            use_torch=False,
            algo=algo,
            num_torch_threads=1,
            use_gpu=args.gpu,
            num_envs=args.num_envs,
            config_name=env_config[1],
        )
        results.append(data)
        f.write(" ".join(data) + "\n")
    for r in results:
        print(*r)
    f.close()


if __name__ == "__main__":
    main()
