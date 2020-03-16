import pytest

from mlagents.tf_utils import tf

import yaml

from mlagents.trainers.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)


@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
        trainer: ppo
        batch_size: 32
        beta: 5.0e-3
        buffer_size: 512
        epsilon: 0.2
        hidden_units: 128
        lambd: 0.95
        learning_rate: 3.0e-4
        max_steps: 5.0e4
        normalize: true
        num_epoch: 5
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 1000
        use_recurrent: false
        normalize: true
        memory_size: 8
        curiosity_strength: 0.0
        curiosity_enc_size: 1
        summary_path: test
        model_path: test
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


VECTOR_ACTION_SPACE = [2]
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 32
NUM_AGENTS = 12


def test_gaussian_distribution():
    with tf.Graph().as_default():
        logits = tf.Variable(initial_value=[[0, 0]], trainable=True, dtype=tf.float32)
        distribution = GaussianDistribution(
            logits,
            act_size=VECTOR_ACTION_SPACE,
            reparameterize=False,
            tanh_squash=False,
        )
        sess = tf.Session()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            output = sess.run(distribution.sample)
            for _ in range(10):
                output = sess.run([distribution.sample, distribution.log_probs])
                for out in output:
                    assert out.shape[1] == VECTOR_ACTION_SPACE[0]
                output = sess.run([distribution.total_log_probs])
                assert output[0].shape[0] == 1


def test_tanh_distribution():
    with tf.Graph().as_default():
        logits = tf.Variable(initial_value=[[0, 0]], trainable=True, dtype=tf.float32)
        distribution = GaussianDistribution(
            logits, act_size=VECTOR_ACTION_SPACE, reparameterize=False, tanh_squash=True
        )
        sess = tf.Session()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            output = sess.run(distribution.sample)
            for _ in range(10):
                output = sess.run([distribution.sample, distribution.log_probs])
                for out in output:
                    assert out.shape[1] == VECTOR_ACTION_SPACE[0]
                # Assert action never exceeds [-1,1]
                action = output[0][0]
                for act in action:
                    assert act >= -1 and act <= 1
                output = sess.run([distribution.total_log_probs])
                assert output[0].shape[0] == 1


def test_multicategorical_distribution():
    with tf.Graph().as_default():
        logits = tf.Variable(initial_value=[[0, 0]], trainable=True, dtype=tf.float32)
        action_masks = tf.Variable(
            initial_value=[[1 for _ in range(sum(DISCRETE_ACTION_SPACE))]],
            trainable=True,
            dtype=tf.float32,
        )
        distribution = MultiCategoricalDistribution(
            logits, act_size=DISCRETE_ACTION_SPACE, action_masks=action_masks
        )
        sess = tf.Session()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            output = sess.run(distribution.sample)
            for _ in range(10):
                sample, log_probs, entropy = sess.run(
                    [distribution.sample, distribution.log_probs, distribution.entropy]
                )
                assert len(log_probs[0]) == sum(DISCRETE_ACTION_SPACE)
                # Assert action never exceeds [-1,1]
                assert len(sample[0]) == len(DISCRETE_ACTION_SPACE)
                for i, act in enumerate(sample[0]):
                    assert act >= 0 and act <= DISCRETE_ACTION_SPACE[i]
                output = sess.run([distribution.total_log_probs])
                assert output[0].shape[0] == 1
                # Make sure entropy is correct
                assert entropy[0] > 3.8

            # Test masks
            mask = []
            for space in DISCRETE_ACTION_SPACE:
                mask.append(1)
                for _action_space in range(1, space):
                    mask.append(0)
            for _ in range(10):
                sample, log_probs = sess.run(
                    [distribution.sample, distribution.log_probs],
                    feed_dict={action_masks: [mask]},
                )
                for act in sample[0]:
                    assert act >= 0 and act <= 1
                output = sess.run([distribution.total_log_probs])
