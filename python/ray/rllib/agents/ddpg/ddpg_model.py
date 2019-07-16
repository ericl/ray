from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class DDPGModel(TFModelV2):
    """Extension of standard TFModel for DDPG.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 actor_hidden_activation="relu",
                 actor_hiddens=(400, 300),
                 critic_hidden_activation="relu",
                 critic_hiddens=(400, 300),
                 parameter_noise=False,
                 twin_q=False,
                 exploration_ou_sigma=0.2):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation (str): activation for actor network
            actor_hiddens (list): hidden layers sizes for actor network
            critic_hidden_activation (str): activation for critic network
            critic_hiddens (list): hidden layers sizes for critic network
            parameter_noise (bool): use param noise exploration
            twin_q (bool): build twin Q networks
            exploration_ou_sigma (float): ou noise sigma for exploration

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of DDPGModel.
        """

        super(DDPGModel, self).__init__(obs_space, action_space, num_outputs,
                                        model_config, name)
        self.exploration_ou_sigma = exploration_ou_sigma

        self.action_dim = np.product(action_space.shape)
        self.model_out = tf.keras.layers.Input(
            shape=(num_outputs, ), name="model_out")
        self.actions = tf.keras.layers.Input(
            shape=(self.action_dim, ), name="actions")

        def build_action_net(action_out):
            activation = getattr(tf.nn, actor_hidden_activation)
            for hidden in actor_hiddens:
                if parameter_noise:
                    import tensorflow.contrib.layers as layers
                    action_out = layers.fully_connected(
                        action_out,
                        num_outputs=hidden,
                        activation_fn=activation,
                        normalizer_fn=layers.layer_norm)
                else:
                    action_out = tf.layers.dense(
                        action_out, units=hidden, activation=activation)
            action_out = tf.layers.dense(
                action_out, units=self.action_dim, activation=None)
            return tf.reshape(self.action_space.shape, action_out)

        pi_out = tf.keras.layers.Lambda(build_action_net)(self.model_out)
        self.action_net = tf.keras.Model(self.model_out, pi_out)
        self.register_variables(self.action_net.variables)

        # Noise vars for P network except for layer normalization vars
        if parameter_noise:
            self._build_parameter_noise([
                var for var in self.action_net.variables
                if "LayerNorm" not in var.name
            ])

        def build_q_net(model_out, actions):
            q_out = tf.concat([model_out, actions], axis=1)
            activation = getattr(tf.nn, critic_hidden_activation)
            for hidden in critic_hiddens:
                q_out = tf.layers.dense(
                    q_out, units=hidden, activation=activation)
            return tf.layers.dense(q_out, units=1, activation=None)

        q_out = tf.keras.layers.Lambda(build_q_net)(
            [self.model_out, self.actions])
        self.q_net = tf.keras.Model([self.model_out, self.actions], q_out)
        self.register_variables(self.q_out.variables)

        if twin_q:
            twin_q_out = tf.keras.layers.Lambda(build_q_net)(
                [self.model_out, self.actions])
            self.twin_q_net = tf.keras.Model([self.model_out, self.actions],
                                             twin_q_out)
            self.register_variables(self.twin_q_out.variables)

    def forward(self, input_dict, state, seq_lens):
        """This generates the model_out tensor input.

        You must implement this as documented in modelv2.py."""
        raise NotImplementedError

    def get_policy_output(self, model_out):
        """Return the (unscaled) output of the policy network.

        This returns the unscaled outputs of pi(s).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_dim] with range [-inf, inf].
        """
        return self.action_net(model_out)

    def get_q_values(self, model_out, actions):
        """Return the Q estimates for the most recent forward pass.

        This implements Q(s, a).

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Tensor): action values that correspond with the most
                recent batch of observations passed through forward(), of shape
                [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        return self.q_net(model_out, actions)

    def _build_parameter_noise(self, pnet_params):
        self.parameter_noise_sigma_val = self.exploration_ou_sigma
        self.parameter_noise_sigma = tf.get_variable(
            initializer=tf.constant_initializer(
                self.parameter_noise_sigma_val),
            name="parameter_noise_sigma",
            shape=(),
            trainable=False,
            dtype=tf.float32)
        self.parameter_noise = []
        # No need to add any noise on LayerNorm parameters
        for var in pnet_params:
            noise_var = tf.get_variable(
                name=var.name.split(":")[0] + "_noise",
                shape=var.shape,
                initializer=tf.constant_initializer(.0),
                trainable=False)
            self.parameter_noise.append(noise_var)
        remove_noise_ops = list()
        for var, var_noise in zip(pnet_params, self.parameter_noise):
            remove_noise_ops.append(tf.assign_add(var, -var_noise))
        self.remove_noise_op = tf.group(*tuple(remove_noise_ops))
        generate_noise_ops = list()
        for var_noise in self.parameter_noise:
            generate_noise_ops.append(
                tf.assign(
                    var_noise,
                    tf.random_normal(
                        shape=var_noise.shape,
                        stddev=self.parameter_noise_sigma)))
        with tf.control_dependencies(generate_noise_ops):
            add_noise_ops = list()
            for var, var_noise in zip(pnet_params, self.parameter_noise):
                add_noise_ops.append(tf.assign_add(var, var_noise))
            self.add_noise_op = tf.group(*tuple(add_noise_ops))
        self.pi_distance = None

    def update_action_noise(self, distance_in_action_space, session):
        """Update the model action noise settings.

        This is called internally by the DDPG policy."""

        self.pi_distance = distance_in_action_space
        if (distance_in_action_space <
                    policy.config["exploration_ou_sigma"] * policy.cur_noise_scale):
            # multiplying the sampled OU noise by noise scale is
            # equivalent to multiplying the sigma of OU by noise scale
            self.parameter_noise_sigma_val *= 1.01
        else:
            self.parameter_noise_sigma_val /= 1.01
        self.parameter_noise_sigma.load(
            self.parameter_noise_sigma_val, session=session)
