# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Behavioral Cloning)
# Contains an implementation of Behavioral Cloning Algorithm

import logging
import copy

from mlagents.trainers.bc.trainer import BCTrainer
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.trainer import UnityTrainerException

logger = logging.getLogger("mlagents.trainers")


class OfflineBCTrainer(BCTrainer):
    """The OfflineBCTrainer is an implementation of Offline Behavioral Cloning."""

    def __init__(self, brain, trainer_parameters, training, load, seed, run_id):
        """
        Responsible for collecting experiences and training PPO model.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The identifier of the current run
        """
        super(OfflineBCTrainer, self).__init__(
            brain, trainer_parameters, training, load, seed, run_id
        )

        self.param_keys = [
            "batch_size",
            "summary_freq",
            "max_steps",
            "batches_per_epoch",
            "use_recurrent",
            "hidden_units",
            "learning_rate",
            "num_layers",
            "sequence_length",
            "memory_size",
            "model_path",
            "demo_path",
        ]

        self.check_param_keys()
        self.batches_per_epoch = trainer_parameters["batches_per_epoch"]
        self.n_sequences = max(
            int(trainer_parameters["batch_size"] / self.policy.sequence_length), 1
        )

        brain_params, self.demonstration_buffer = demo_to_buffer(
            trainer_parameters["demo_path"], self.policy.sequence_length
        )

        policy_brain = copy.deepcopy(brain.__dict__)
        expert_brain = copy.deepcopy(brain_params.__dict__)
        policy_brain.pop("brain_name")
        expert_brain.pop("brain_name")
        if expert_brain != policy_brain:
            raise UnityTrainerException(
                "The provided demonstration is not compatible with the "
                "brain being used for performance evaluation."
            )

    def __str__(self):
        return """Hyperparameters for the Imitation Trainer of brain {0}: \n{1}""".format(
            self.brain_name,
            "\n".join(
                [
                    "\t{0}:\t{1}".format(x, self.trainer_parameters[x])
                    for x in self.param_keys
                ]
            ),
        )
