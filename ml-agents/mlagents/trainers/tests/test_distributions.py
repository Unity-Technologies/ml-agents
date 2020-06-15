import pytest

from mlagents.tf_utils import tf

from mlagents.trainers.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)


VECTOR_ACTION_SPACE = [2]
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 32
NUM_AGENTS = 12


def test_gaussian_distribution():
    with tf.Graph().as_default():
        logits = tf.Variable(initial_value=[[1, 1]], trainable=True, dtype=tf.float32)
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
            # Test entropy is correct
            log_std_tensor = tf.get_default_graph().get_tensor_by_name(
                "log_std/BiasAdd:0"
            )
            feed_dict = {log_std_tensor: [[1.0, 1.0]]}
            entropy = sess.run([distribution.entropy], feed_dict=feed_dict)
            # Entropy with log_std of 1.0 should be 2.42
            assert pytest.approx(entropy[0], 0.01) == 2.42


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
