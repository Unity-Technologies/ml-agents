import abc
from typing import NamedTuple, List
import numpy as np

from mlagents.tf_utils import tf
from mlagents.trainers.models import LearningModel

EPSILON = 1e-6  # Small value to avoid divide by zero


class OutputDistribution(abc.ABC):
    @abc.abstractproperty
    def log_probs(self) -> tf.Tensor:
        """
        Returns a Tensor that when evaluated, produces the per-action log probabilities of this distribution.
        The shape of this Tensor should be equivalent to (batch_size x the number of actions) produced in sample.
        """
        pass

    @abc.abstractproperty
    def total_log_probs(self) -> tf.Tensor:
        """
        Returns a Tensor that when evaluated, produces the total log probability for a single sample.
        The shape of this Tensor should be equivalent to (batch_size x 1) produced in sample.
        """
        pass

    @abc.abstractproperty
    def sample(self) -> tf.Tensor:
        """
        Returns a Tensor that when evaluated, produces a sample of this OutputDistribution.
        """
        pass

    @abc.abstractproperty
    def entropy(self) -> tf.Tensor:
        """
        Returns a Tensor that when evaluated, produces the entropy of this distribution.
        """
        pass


class GaussianDistribution(OutputDistribution):
    """
    A Gaussian output distribution for continuous actions.
    """

    class MuSigmaTensors(NamedTuple):
        mu: tf.Tensor
        log_sigma: tf.Tensor
        sigma: tf.Tensor

    def __init__(
        self,
        logits: tf.Tensor,
        act_size: List[int],
        pass_gradients: bool = False,
        log_sigma_min: float = -20,
        log_sigma_max: float = 2,
    ):
        encoded = self._create_mu_log_sigma(
            logits, act_size, log_sigma_min, log_sigma_max
        )
        sampled_policy = self._create_sampled_policy(encoded)
        if not pass_gradients:
            self._sampled_policy = tf.stop_gradient(sampled_policy)
        else:
            self._sampled_policy = sampled_policy
        self._all_probs = self._get_log_probs(self._sampled_policy, encoded)
        self._total_prob = tf.reduce_sum(self._all_probs, axis=1, keepdims=True)
        self._entropy = self._create_entropy(encoded)

    def _create_mu_log_sigma(
        self,
        logits: tf.Tensor,
        act_size: List[int],
        log_sigma_min: float,
        log_sigma_max: float,
    ) -> "GaussianDistribution.MuSigmaTensors":

        mu = tf.layers.dense(
            logits,
            act_size[0],
            activation=None,
            name="mu",
            kernel_initializer=LearningModel.scaled_init(0.01),
            reuse=tf.AUTO_REUSE,
        )

        # Policy-dependent log_sigma_sq
        log_sigma = tf.layers.dense(
            logits,
            act_size[0],
            activation=None,
            name="log_std",
            kernel_initializer=LearningModel.scaled_init(0.01),
        )
        log_sigma = tf.clip_by_value(log_sigma, log_sigma_min, log_sigma_max)
        sigma = tf.exp(log_sigma)
        return self.MuSigmaTensors(mu, log_sigma, sigma)

    def _create_sampled_policy(
        self, encoded: "GaussianDistribution.MuSigmaTensors"
    ) -> tf.Tensor:
        epsilon = tf.random_normal(tf.shape(encoded.mu))
        sampled_policy = encoded.mu + encoded.sigma * epsilon

        return sampled_policy

    def _get_log_probs(
        self, sampled_policy: tf.Tensor, encoded: "GaussianDistribution.MuSigmaTensors"
    ) -> tf.Tensor:
        _gauss_pre = -0.5 * (
            ((sampled_policy - encoded.mu) / (encoded.sigma + EPSILON)) ** 2
            + 2 * encoded.log_sigma
            + np.log(2 * np.pi)
        )
        return tf.reduce_sum(_gauss_pre, axis=1, keepdims=True)

    def _create_entropy(
        self, encoded: "GaussianDistribution.MuSigmaTensors"
    ) -> tf.Tensor:
        single_dim_entropy = 0.5 * tf.reduce_mean(
            tf.log(2 * np.pi * np.e) + tf.square(encoded.log_sigma)
        )
        # Make entropy the right shape
        return tf.ones_like(tf.reshape(encoded.mu[:, 0], [-1])) * single_dim_entropy

    @property
    def total_log_probs(self) -> tf.Tensor:
        return self._total_prob

    @property
    def log_probs(self) -> tf.Tensor:
        return self._all_probs

    @property
    def sample(self) -> tf.Tensor:
        return self._sampled_policy

    @property
    def entropy(self) -> tf.Tensor:
        return self._entropy


class TanhSquashedGaussianDistribution(GaussianDistribution):
    """
    A Gaussian distribution that is squashed by a tanh at the output. We adjust the log-probabilities
    to account for this squashing. From: Haarnoja et. al, https://arxiv.org/abs/1801.01290
    """

    def __init__(
        self,
        logits: tf.Tensor,
        act_size: List[int],
        pass_gradients: bool = False,
        log_sigma_min: float = -20,
        log_sigma_max: float = 2,
    ):
        super().__init__(logits, act_size, pass_gradients, log_sigma_min, log_sigma_max)
        self._squashed_policy = tf.tanh(self._sampled_policy)
        self._corrected_probs = self._do_squash_correction_for_tanh(
            self._all_probs, self._squashed_policy
        )
        self._corrected_total_prob = tf.reduce_sum(
            self._corrected_probs, axis=1, keepdims=True
        )

    def _do_squash_correction_for_tanh(self, probs, squashed_policy):
        """
        Adjust probabilities for squashed sample before output
        """
        all_probs = probs
        all_probs -= tf.log(1 - squashed_policy ** 2 + EPSILON)
        return all_probs

    @property
    def log_probs(self) -> tf.Tensor:
        return self._corrected_probs

    @property
    def total_log_probs(self) -> tf.Tensor:
        return self._corrected_total_prob

    @property
    def sample(self) -> tf.Tensor:
        return self._squashed_policy


# def MultiCategoricalDistribution(OutputDistribution):

#     def __init__(
#         self,
#         logits: tf.Tensor,
#         act_size: List[int],
#     ):
#         encoded = self._create_mu_log_sigma(
#             logits, act_size, log_sigma_min, log_sigma_max
#         )
#         sampled_policy = self._create_sampled_policy(encoded)
#         if not pass_gradients:
#             self._sampled_policy = tf.stop_gradient(sampled_policy)
#         else:
#             self._sampled_policy = sampled_policy
#         self._all_probs = self._get_log_probs(self._sampled_policy, encoded)
#         self._total_prob = tf.reduce_sum(self._all_probs, axis=1, keepdims=True)
#         self._entropy = self._create_entropy(encoded)

#     def _create_policy_branches(self, logits, act_size: List[int]) -> List[tf.Tensor]:
#         policy_branches = []
#         for size in act_size:
#             policy_branches.append(
#                 tf.layers.dense(
#                     logits,
#                     size,
#                     activation=None,
#                     use_bias=False,
#                     kernel_initializer=LearningModel.scaled_init(0.01),
#                 )
#             )
#         return policy_branches

#     def _
#         raw_log_probs = tf.concat(policy_branches, axis=1, name="action_probs")

#         self.action_masks = tf.placeholder(
#             shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
#         )
#         output, self.action_probs, normalized_logits = LearningModel.create_discrete_action_masking_layer(
#             raw_log_probs, self.action_masks, self.act_size
#         )

#         self.output = tf.identity(output)
#         self.all_log_probs = tf.identity(normalized_logits, name="action")


#         self.action_holder = tf.placeholder(
#             shape=[None, len(policy_branches)], dtype=tf.int32, name="action_holder"
#         )
#         self.action_oh = tf.concat(
#             [
#                 tf.one_hot(self.action_holder[:, i], self.act_size[i])
#                 for i in range(len(self.act_size))
#             ],
#             axis=1,
#         )
#         self.selected_actions = tf.stop_gradient(self.action_oh)

#         action_idx = [0] + list(np.cumsum(self.act_size))

#         self.entropy = tf.reduce_sum(
#             (
#                 tf.stack(
#                     [
#                         tf.nn.softmax_cross_entropy_with_logits_v2(
#                             labels=tf.nn.softmax(
#                                 self.all_log_probs[:, action_idx[i] : action_idx[i + 1]]
#                             ),
#                             logits=self.all_log_probs[
#                                 :, action_idx[i] : action_idx[i + 1]
#                             ],
#                         )
#                         for i in range(len(self.act_size))
#                     ],
#                     axis=1,
#                 )
#             ),
#             axis=1,
#         )

#         self.log_probs = tf.reduce_sum(
#             (
#                 tf.stack(
#                     [
#                         -tf.nn.softmax_cross_entropy_with_logits_v2(
#                             labels=self.action_oh[:, action_idx[i] : action_idx[i + 1]],
#                             logits=normalized_logits[
#                                 :, action_idx[i] : action_idx[i + 1]
#                             ],
#                         )
#                         for i in range(len(self.act_size))
#                     ],
#                     axis=1,
#                 )
#             ),
#             axis=1,
#             keepdims=True,
#         )
