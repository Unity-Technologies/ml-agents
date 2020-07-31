from typing import Dict, Any
import numpy as np

from mlagents.trainers.policy.torch_policy import TorchPolicy
from .model_torch import TorchBCModel
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.settings import BehavioralCloningSettings
from mlagents.trainers.torch.utils import ModelUtils


class TorchBCModule:
    def __init__(
        self,
        policy: TorchPolicy,
        settings: BehavioralCloningSettings,
        policy_learning_rate: float,
        default_batch_size: int,
        default_num_epoch: int,
    ):
        """
        A BC trainer that can be used inline with RL.
        :param policy: The policy of the learning model
        :param policy_learning_rate: The initial Learning Rate of the policy. Used to set an appropriate learning rate
            for the pretrainer.
        :param default_batch_size: The default batch size to use if batch_size isn't provided.
        :param default_num_epoch: The default num_epoch to use if num_epoch isn't provided.
        :param strength: The proportion of learning rate used to update through BC.
        :param steps: The number of steps to anneal BC training over. 0 for continuous training.
        :param demo_path: The path to the demonstration file.
        :param batch_size: The batch size to use during BC training.
        :param num_epoch: Number of epochs to train for during each update.
        :param samples_per_update: Maximum number of samples to train on during each BC update.
        """
        self.policy = policy
        self.current_lr = policy_learning_rate * settings.strength
        params = list(self.policy.actor_critic.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=self.current_lr
        )

        _, self.demonstration_buffer = demo_to_buffer(
            settings.demo_path, policy.sequence_length, policy.behavior_spec
        )

        self.batch_size = (
            settings.batch_size if settings.batch_size else default_batch_size
        )
        self.num_epoch = settings.num_epoch if settings.num_epoch else default_num_epoch
        self.n_sequences = max(
            min(self.batch_size, self.demonstration_buffer.num_experiences)
            // policy.sequence_length,
            1,
        )

        self.has_updated = False
        self.use_recurrent = self.policy.use_recurrent
        self.samples_per_update = settings.samples_per_update
        self.out_dict = {
            "loss": self.model.loss,
            "update": self.model.update_batch,
            "learning_rate": self.model.annealed_learning_rate,
        }

    def update(self) -> Dict[str, Any]:
        """
        Updates model using buffer.
        :param max_batches: The maximum number of batches to use per update.
        :return: The loss of the update.
        """
        # Don't continue training if the learning rate has reached 0, to reduce training time.
        if self.current_lr <= 0:
            return {"Losses/Pretraining Loss": 0}

        batch_losses = []
        possible_demo_batches = (
            self.demonstration_buffer.num_experiences // self.n_sequences
        )
        possible_batches = possible_demo_batches

        max_batches = self.samples_per_update // self.n_sequences

        n_epoch = self.num_epoch
        for _ in range(n_epoch):
            self.demonstration_buffer.shuffle(
                sequence_length=self.policy.sequence_length
            )
            if max_batches == 0:
                num_batches = possible_batches
            else:
                num_batches = min(possible_batches, max_batches)
            for i in range(num_batches // self.policy.sequence_length):
                demo_update_buffer = self.demonstration_buffer
                start = i * self.n_sequences * self.policy.sequence_length
                end = (i + 1) * self.n_sequences * self.policy.sequence_length
                mini_batch_demo = demo_update_buffer.make_mini_batch(start, end)
                run_out = self._update_batch(mini_batch_demo, self.n_sequences)
                loss = run_out["loss"]
                #self.current_lr = update_stats["learning_rate"]
                batch_losses.append(loss)
        self.has_updated = True
        update_stats = {"Losses/Pretraining Loss": np.mean(batch_losses)}
        return update_stats

    def _update_batch(
        self, mini_batch_demo: Dict[str, Any], n_sequences: int
    ) -> Dict[str, Any]:
        """
        Helper function for update_batch.
        """
        vec_obs = [ModelUtils.list_to_tensor(mini_batch_demo["vector_obs"])]
        act_masks = ModelUtils.list_to_tensor(mini_batch_demo["action_mask"])
        if self.policy.use_continuous_act:
            expert_actions = ModelUtils.list_to_tensor(mini_batch_demo["actions"]).unsqueeze(-1)
        else:
            expert_actions = ModelUtils.list_to_tensor(mini_batch_demo["actions"], dtype=torch.long)

        memories = [
            ModelUtils.list_to_tensor(mini_batch_demo["memory"][i])
            for i in range(0, len(mini_batch_demo["memory"]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        if self.policy.use_vis_obs:
            vis_obs = []
            for idx, _ in enumerate(
                self.policy.actor_critic.network_body.visual_encoders
            ):
                vis_ob = ModelUtils.list_to_tensor(mini_batch_demo["visual_obs%d" % idx])
                vis_obs.append(vis_ob)
        else:
            vis_obs = []

        selected_actions, log_probs, entropies, values, memories = self.policy.sample_actions(
            vec_obs,
            vis_obs,
            masks=act_masks,
            memories=memories,
            seq_len=self.policy.sequence_length,
            all_log_probs=True,
        )

        bc_loss = self._behavioral_cloning_loss(selected_actions, all_log_probs, expert_actions)
        self.optimizer.zero_grad()
        bc_loss.backward()

        self.optimizer.step()
        run_out = {
            "loss": bc_loss.detach().cpu().numpy(),
        }
        return run_out

        def _behavioral_cloning_loss(self, selected_actions, log_probs, expert_actions)
            if self.policy.use_continuous_act:
                loss = (selected_actions - expert_actions) ** 2
            else:
                loss = -torch.log(torch.nn.Softmax(log_probs) + 1e-7) * expert_actions
            bc_loss = torch.mean(loss)
            return bc_loss
