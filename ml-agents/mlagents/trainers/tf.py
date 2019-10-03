import tensorflow as tf

# TODO better version check, this will do for now though
is_tf2 = tf.__version__ == "2.0.0"

if is_tf2:
    import tensorflow.compat.v1 as tf

    tf_variance_scaling = tf.initializers.variance_scaling
    tf_flatten = tf.layers.flatten
    tf_rnn = tf.nn.rnn_cell

    tf.disable_v2_behavior()
else:
    import tensorflow.contrib.layers as c_layers

    tf_variance_scaling = c_layers.variance_scaling_initializer
    tf_flatten = c_layers.flatten
    tf_rnn = tf.contrib.rnn
