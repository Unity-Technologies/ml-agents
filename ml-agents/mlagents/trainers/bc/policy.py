import logging

import numpy as np
from mlagents.trainers.bc.models import BehavioralCloningModel
from mlagents.trainers.tf_policy import TFPolicy

logger = logging.getLogger("mlagents.trainers")


class BCPolicy(TFPolicy):
    def __init__(self, seed, brain, trainer_parameters, load):
        """
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_parameters: Defined training parameters.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        super(BCPolicy, self).__init__(seed, brain, trainer_parameters)

        with self.graph.as_default():
            with self.graph.as_default():
                self.model = BehavioralCloningModel(
                    h_size=int(trainer_parameters["hidden_units"]),
                    lr=float(trainer_parameters["learning_rate"]),
                    n_layers=int(trainer_parameters["num_layers"]),
                    m_size=self.m_size,
                    normalize=False,
                    use_recurrent=trainer_parameters["use_recurrent"],
                    brain=brain,
                    seed=seed,
                )

        if load:
            self._load_graph()
        else:
            self._initialize_graph()

        self.inference_dict = {"action": self.model.sample_action}
        self.update_dict = {
            "policy_loss": self.model.loss,
            "update_batch": self.model.update,
        }
        if self.use_recurrent:
            self.inference_dict["memory_out"] = self.model.memory_out

        self.evaluate_rate = 1.0
        self.update_rate = 0.5

    def evaluate(self, brain_info):
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo input to network.
        :return: Results of evaluation.
        """
        feed_dict = {
            self.model.dropout_rate: self.evaluate_rate,
            self.model.sequence_length: 1,
        }

        feed_dict = self.fill_eval_dict(feed_dict, brain_info)
        if self.use_recurrent:
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = self.make_empty_memory(len(brain_info.agents))
            feed_dict[self.model.memory_in] = brain_info.memories
        run_out = self._execute_model(feed_dict, self.inference_dict)
        return run_out

    def update(self, mini_batch, num_sequences):
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """

        feed_dict = {
            self.model.dropout_rate: self.update_rate,
            self.model.batch_size: num_sequences,
            self.model.sequence_length: self.sequence_length,
        }
        if self.use_continuous_act:
            feed_dict[self.model.true_action] = mini_batch["actions"]
        else:
            feed_dict[self.model.true_action] = mini_batch["actions"]
            feed_dict[self.model.action_masks] = np.ones(
                (num_sequences, sum(self.brain.vector_action_space_size))
            )
        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = mini_batch["vector_obs"]
        for i, _ in enumerate(self.model.visual_in):
            visual_obs = mini_batch["visual_obs%d" % i]
            feed_dict[self.model.visual_in[i]] = visual_obs
        if self.use_recurrent:
            feed_dict[self.model.memory_in] = np.zeros([num_sequences, self.m_size])
        run_out = self._execute_model(feed_dict, self.update_dict)
        return run_out
