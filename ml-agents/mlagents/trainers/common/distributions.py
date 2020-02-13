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
        Returns a Tensor that when evaluated, produces the log probabilities of this distribution.
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
    def log_probs(self) -> tf.Tensor:
        return tf.identity(self._all_probs, name="action_probs")

    @property
    def sample(self) -> tf.Tensor:
        return self._sampled_policy

    @property
    def entropy(self) -> tf.Tensor:
        return self._entropy


class TanhSquashedGaussianDistribution(GaussianDistribution):
    @property
    def log_probs(self) -> tf.Tensor:
        all_probs = self._all_probs
        all_probs -= tf.reduce_sum(
            tf.log(1 - self.sample ** 2 + EPSILON), axis=1, keepdims=True
        )
        return all_probs

    @property
    def sample(self) -> tf.Tensor:
        return tf.tanh(self._sampled_policy)
