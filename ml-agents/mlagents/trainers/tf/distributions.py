import abc
from typing import NamedTuple, List, Tuple
import numpy as np

from mlagents.tf_utils import tf
from mlagents.trainers.tf.models import ModelUtils

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


class DiscreteOutputDistribution(OutputDistribution):
    @abc.abstractproperty
    def sample_onehot(self) -> tf.Tensor:
        """
        Returns a one-hot version of the output.
        """


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
        reparameterize: bool = False,
        tanh_squash: bool = False,
        condition_sigma: bool = True,
        log_sigma_min: float = -20,
        log_sigma_max: float = 2,
    ):
        """
        A Gaussian output distribution for continuous actions.
        :param logits: Hidden layer to use as the input to the Gaussian distribution.
        :param act_size: List containing the number of continuous actions.
        :param reparameterize: Whether or not to use the reparameterization trick (block gradients through
            log probability calculation.)
        :param tanh_squash: Squash the output using tanh, constraining it between -1 and 1.
            From: Haarnoja et. al, https://arxiv.org/abs/1801.01290
        :param log_sigma_min: Minimum log standard deviation to clip by.
        :param log_sigma_max: Maximum log standard deviation to clip by.
        """
        encoded = self._create_mu_log_sigma(
            logits,
            act_size,
            log_sigma_min,
            log_sigma_max,
            condition_sigma=condition_sigma,
        )
        self._sampled_policy = self._create_sampled_policy(encoded)
        if not reparameterize:
            _sampled_policy_probs = tf.stop_gradient(self._sampled_policy)
        else:
            _sampled_policy_probs = self._sampled_policy
        self._all_probs = self._create_log_probs(_sampled_policy_probs, encoded)
        if tanh_squash:
            self._sampled_policy = tf.tanh(self._sampled_policy)
            self._all_probs = self._do_squash_correction_for_tanh(
                self._all_probs, self._sampled_policy
            )
        self._total_prob = tf.reduce_sum(self._all_probs, axis=1, keepdims=True)
        self._entropy = self._create_entropy(encoded)

    def _create_mu_log_sigma(
        self,
        logits: tf.Tensor,
        act_size: List[int],
        log_sigma_min: float,
        log_sigma_max: float,
        condition_sigma: bool,
    ) -> "GaussianDistribution.MuSigmaTensors":

        mu = tf.layers.dense(
            logits,
            act_size[0],
            activation=None,
            name="mu",
            kernel_initializer=ModelUtils.scaled_init(0.01),
            reuse=tf.AUTO_REUSE,
        )

        if condition_sigma:
            # Policy-dependent log_sigma_sq
            log_sigma = tf.layers.dense(
                logits,
                act_size[0],
                activation=None,
                name="log_std",
                kernel_initializer=ModelUtils.scaled_init(0.01),
            )
        else:
            log_sigma = tf.get_variable(
                "log_std",
                [act_size[0]],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
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

    def _create_log_probs(
        self, sampled_policy: tf.Tensor, encoded: "GaussianDistribution.MuSigmaTensors"
    ) -> tf.Tensor:
        _gauss_pre = -0.5 * (
            ((sampled_policy - encoded.mu) / (encoded.sigma + EPSILON)) ** 2
            + 2 * encoded.log_sigma
            + np.log(2 * np.pi)
        )
        return _gauss_pre

    def _create_entropy(
        self, encoded: "GaussianDistribution.MuSigmaTensors"
    ) -> tf.Tensor:
        single_dim_entropy = 0.5 * tf.reduce_mean(
            tf.log(2 * np.pi * np.e) + 2 * encoded.log_sigma
        )
        # Make entropy the right shape
        return tf.ones_like(tf.reshape(encoded.mu[:, 0], [-1])) * single_dim_entropy

    def _do_squash_correction_for_tanh(self, probs, squashed_policy):
        """
        Adjust probabilities for squashed sample before output
        """
        adjusted_probs = probs - tf.log(1 - squashed_policy ** 2 + EPSILON)
        return adjusted_probs

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


class MultiCategoricalDistribution(DiscreteOutputDistribution):
    """
    A categorical distribution for multi-branched discrete actions. Also supports action masking.
    """

    def __init__(self, logits: tf.Tensor, act_size: List[int], action_masks: tf.Tensor):
        """
        A categorical distribution for multi-branched discrete actions.
        :param logits: Hidden layer to use as the input to the Gaussian distribution.
        :param act_size: List containing the number of discrete actions per branch.
        :param action_masks: Tensor representing action masks. Should be of length sum(act_size), and 0 for masked
            and 1 for unmasked.
        """
        unmasked_log_probs = self._create_policy_branches(logits, act_size)
        (
            self._sampled_policy,
            self._all_probs,
            action_index,
        ) = self._get_masked_actions_probs(unmasked_log_probs, act_size, action_masks)
        self._sampled_onehot = self._action_onehot(self._sampled_policy, act_size)
        self._entropy = self._create_entropy(self._all_probs, action_index, act_size)
        self._total_prob = self._get_log_probs(
            self._sampled_onehot, self._all_probs, action_index, act_size
        )

    def _create_policy_branches(
        self, logits: tf.Tensor, act_size: List[int]
    ) -> List[tf.Tensor]:
        policy_branches = []
        for size in act_size:
            policy_branches.append(
                tf.layers.dense(
                    logits,
                    size,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=ModelUtils.scaled_init(0.01),
                )
            )
        return policy_branches

    def _get_masked_actions_probs(
        self,
        unmasked_log_probs: List[tf.Tensor],
        act_size: List[int],
        action_masks: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, np.ndarray]:
        output, _, all_log_probs = ModelUtils.create_discrete_action_masking_layer(
            unmasked_log_probs, action_masks, act_size
        )

        action_idx = [0] + list(np.cumsum(act_size))
        return output, all_log_probs, action_idx

    def _action_onehot(self, sample: tf.Tensor, act_size: List[int]) -> tf.Tensor:
        action_oh = tf.concat(
            [tf.one_hot(sample[:, i], act_size[i]) for i in range(len(act_size))],
            axis=1,
        )
        return action_oh

    def _get_log_probs(
        self,
        sample_onehot: tf.Tensor,
        all_log_probs: tf.Tensor,
        action_idx: List[int],
        act_size: List[int],
    ) -> tf.Tensor:
        log_probs = tf.reduce_sum(
            (
                tf.stack(
                    [
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=sample_onehot[:, action_idx[i] : action_idx[i + 1]],
                            logits=all_log_probs[:, action_idx[i] : action_idx[i + 1]],
                        )
                        for i in range(len(act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
            keepdims=True,
        )
        return log_probs

    def _create_entropy(
        self, all_log_probs: tf.Tensor, action_idx: List[int], act_size: List[int]
    ) -> tf.Tensor:
        entropy = tf.reduce_sum(
            (
                tf.stack(
                    [
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=tf.nn.softmax(
                                all_log_probs[:, action_idx[i] : action_idx[i + 1]]
                            ),
                            logits=all_log_probs[:, action_idx[i] : action_idx[i + 1]],
                        )
                        for i in range(len(act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
        )

        return entropy

    @property
    def log_probs(self) -> tf.Tensor:
        return self._all_probs

    @property
    def total_log_probs(self) -> tf.Tensor:
        return self._total_prob

    @property
    def sample(self) -> tf.Tensor:
        return self._sampled_policy

    @property
    def sample_onehot(self) -> tf.Tensor:
        return self._sampled_onehot

    @property
    def entropy(self) -> tf.Tensor:
        return self._entropy
