
import json
import os
import torch
from mlagents.trainers.learn import run_cli, parse_command_line
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.ppo.trainer import TestingConfiguration
from mlagents_envs.timers import _thread_timer_stacks


results = {}

def run_experiment(name:str, steps:int, torch:bool, config_name=None):
	TestingConfiguration.env_name = name
	TestingConfiguration.max_steps = steps
	TestingConfiguration.use_torch = torch
	if config_name is None:
		config_name = name
	run_options = parse_command_line([f"config/ppo/{config_name}.yaml"])
	run_options.checkpoint_settings.run_id = f"{name}_test_" +str(steps) +"_"+("torch" if torch else "tf")
	run_options.checkpoint_settings.force = True
	for trainer_settings in run_options.behaviors.values():
		trainer_settings.threaded = False
	timers_path = os.path.join("results", run_options.checkpoint_settings.run_id, "run_logs", "timers.json")
	run_cli(run_options)
	StatsReporter.writers.clear()
	StatsReporter.stats_dict.clear()
	_thread_timer_stacks.clear()
	with open(timers_path) as timers_json_file:
		timers_json = json.load(timers_json_file)
		total = timers_json["total"]
		evaluate = timers_json["children"]["TrainerController.start_learning"]["children"]["TrainerController.advance"]["children"]["env_step"]["children"]["SubprocessEnvManager._take_step"]["children"]
		update = timers_json["children"]["TrainerController.start_learning"]["children"]["TrainerController.advance"]["children"]["trainer_advance"]["children"]["_update_policy"]["children"]
	if torch:
		update = update["TorchPPOOptimizer.update"]["total"]
		evaluate = evaluate["TorchPolicy.evaluate"]["total"]
	else:
		update = update["TFPPOOptimizer.update"]["total"]
		evaluate = evaluate["NNPolicy.evaluate"]["total"]
	# todo: do total / count
	return total, update, evaluate



results["3DBall Torch"] = run_experiment("3DBall", 20000, True)
results["3DBall TF"] = run_experiment("3DBall", 20000, False)
results["GridWorld Torch"] = run_experiment("GridWorld", 2000, True)
results["GridWorld TF"] = run_experiment("GridWorld", 2000, False)
results["PushBlock Torch"] = run_experiment("PushBlock", 20000, True)
results["PushBlock TF"] = run_experiment("PushBlock", 20000, False)
results["Hallway Torch"] = run_experiment("Hallway", 20000, True)
results["Hallway TF"] = run_experiment("Hallway", 20000, False)
results["CrawlerStaticTarget Torch"] = run_experiment("CrawlerStaticTarget", 50000, True, "CrawlerStatic")
results["CrawlerStaticTarget TF"] = run_experiment("CrawlerStaticTarget", 50000, False, "CrawlerStatic")

torch.set_num_threads(1)

results["3DBall Torch 1 thread"] = run_experiment("3DBall", 20000, True)
results["GridWorld Torch 1 thread"] = run_experiment("GridWorld", 2000, True)
results["PushBlock Torch 1 thread"] = run_experiment("PushBlock", 20000, True)
results["Hallway Torch 1 thread"] = run_experiment("Hallway", 20000, True)
results["CrawlerStaticTarget Torch 1 thread"] = run_experiment("CrawlerStaticTarget", 50000, True, "CrawlerStatic")


print("experiment,\t", "total,\t", "update,\t", "evaluate")
for key, value in results.items():
	print(key + ",\t", str(value[0])+ ",\t", str(value[1])+ ",\t", str(value[2])+ "\t")



