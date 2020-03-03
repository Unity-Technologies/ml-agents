from mlagents.tf_utils import tf

from mlagents.trainers.policy.tf_policy import TFPolicy


class BCModel(object):
    def __init__(
        self, policy: TFPolicy, learning_rate: float = 3e-4, anneal_steps: int = 0
    ):
        """
        Tensorflow operations to perform Behavioral Cloning on a Policy model
        :param policy: The policy of the learning algorithm
        :param lr: The initial learning Rate for behavioral cloning
        :param anneal_steps: Number of steps over which to anneal BC training
        """
        self.policy = policy
        self.expert_visual_in = self.policy.visual_in
        self.obs_in_expert = self.policy.vector_in
        self.make_inputs()
        self.create_loss(learning_rate, anneal_steps)

    def make_inputs(self) -> None:
        """
        Creates the input layers for the discriminator
        """
        self.done_expert = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.done_policy = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        if self.policy.brain.vector_action_space_type == "continuous":
            action_length = self.policy.act_size[0]
            self.action_in_expert = tf.placeholder(
                shape=[None, action_length], dtype=tf.float32
            )
            self.expert_action = tf.identity(self.action_in_expert)
        else:
            action_length = len(self.policy.act_size)
            self.action_in_expert = tf.placeholder(
                shape=[None, action_length], dtype=tf.int32
            )
            self.expert_action = tf.concat(
                [
                    tf.one_hot(self.action_in_expert[:, i], act_size)
                    for i, act_size in enumerate(self.policy.act_size)
                ],
                axis=1,
            )

    def create_loss(self, learning_rate: float, anneal_steps: int) -> None:
        """
        Creates the loss and update nodes for the BC module
        :param learning_rate: The learning rate for the optimizer
        :param anneal_steps: Number of steps over which to anneal the learning_rate
        """
        selected_action = self.policy.output
        if self.policy.use_continuous_act:
            self.loss = tf.reduce_mean(
                tf.squared_difference(selected_action, self.expert_action)
            )
        else:
            log_probs = self.policy.all_log_probs
            self.loss = tf.reduce_mean(
                -tf.log(tf.nn.softmax(log_probs) + 1e-7) * self.expert_action
            )

        if anneal_steps > 0:
            self.annealed_learning_rate = tf.train.polynomial_decay(
                learning_rate, self.policy.global_step, anneal_steps, 0.0, power=1.0
            )
        else:
            self.annealed_learning_rate = tf.Variable(learning_rate)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.annealed_learning_rate, name="bc_adam"
        )
        self.update_batch = optimizer.minimize(self.loss)
