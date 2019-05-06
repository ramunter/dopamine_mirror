# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compact implementation of a DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random

from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import circular_replay_buffer
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gin.tf

slim = tf.contrib.slim
tfd = tfp.distributions


# These are aliases which are used by other classes.
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = atari_lib.NATURE_DQN_DTYPE
NATURE_DQN_STACK_SIZE = atari_lib.NATURE_DQN_STACK_SIZE
bayesian_dqn_network = atari_lib.bayesian_dqn_network

Dist = collections.namedtuple(
    'Dist', ['mean', 'inv_cov', 'a', 'b'])

Parameters = collections.namedtuple(
    'Parameters', ['coef', 'noise'])


@gin.configurable
class BDQNAgent(dqn_agent.DQNAgent):
    """An implementation of the DQN agent."""

    def __init__(self,
                 sess,
                 num_actions,
                 observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=atari_lib.NATURE_DQN_DTYPE,
                 stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
                 network=bayesian_dqn_network,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 tf_device='/cpu:*',
                 use_staging=True,
                 max_tf_checkpoints_to_keep=4,
                 optimizer=tf.train.RMSPropOptimizer(
                     learning_rate=0.00025,
                     decay=0.95,
                     momentum=0.0,
                     epsilon=0.00001,
                     centered=True),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 bayes_update_period=1000):
        """Initializes the agent and constructs the components of its graph.

        Args:
          sess: `tf.Session`, for executing ops.
          num_actions: int, number of actions the agent can take at any state.
          observation_shape: tuple of ints describing the observation shape.
          observation_dtype: tf.DType, specifies the type of the observations. Note
            that if your inputs are continuous, you should set this to tf.float32.
          stack_size: int, number of frames to use in state stack.
          network: function expecting three parameters:
            (num_actions, network_type, state). This function will return the
            network_type object containing the tensors output by the network.
            See dopamine.discrete_domains.atari_lib.nature_dqn_network as
            an example.
          gamma: float, discount factor with the usual RL meaning.
          update_horizon: int, horizon at which updates are performed, the 'n' in
            n-step update.
          min_replay_history: int, number of transitions that should be experienced
            before the agent begins training its value function.
          update_period: int, period between DQN updates.
          target_update_period: int, update period for the target network.
          epsilon_fn: function expecting 4 parameters:
            (decay_period, step, warmup_steps, epsilon). This function should return
            the epsilon value used for exploration during training.
          epsilon_train: float, the value to which the agent's epsilon is eventually
            decayed during training.
          epsilon_eval: float, epsilon used when evaluating the agent.
          epsilon_decay_period: int, length of the epsilon decay schedule.
          tf_device: str, Tensorflow device on which the agent's graph is executed.
          use_staging: bool, when True use a staging area to prefetch the next
            training batch, speeding training up by about 30%.
          max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
            keep.
          optimizer: `tf.train.Optimizer`, for training the value function.
          summary_writer: SummaryWriter object for outputting training statistics.
            Summary writing disabled if set to None.
          summary_writing_frequency: int, frequency with which summaries will be
            written. Lower values will result in slower training.
        """
        self._bayes_update_period = bayes_update_period

        super().__init__(
            sess=sess,
            num_actions=num_actions,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size,
            network=network,
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            tf_device=tf_device,
            use_staging=use_staging,
            optimizer=optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)

        self._build_bayes_network()
        self._update_bayes_op = self._build_update_bayes_op()

        self._train_op = self._rebuild_train_op()
        if self.summary_writer is not None:
            self._merged_summaries = tf.summary.merge_all()

        print("===========")
        print(self._replay.memory._batch_size)

    def _get_network_type(self):
        """Returns the type of the outputs of a BDQN value network.
        Returns:
          net_type: _network_type object defining the outputs of the network.
        """
        return collections.namedtuple('BDQN_network', ['q_values', 'encoding', ])

    def _build_dist(self, size, action):
        mu = tf.get_variable("dist_mean"+str(action), initializer=tf.cast(
            np.zeros((size, 1)), dtype=tf.float32))
        inv_cov = tf.get_variable("dist_invcov"+str(action), initializer=tf.cast(
            np.eye(size)*10, dtype=tf.float32))
        b = tf.get_variable(
            "dist_b"+str(action), initializer=tf.cast(0.1, dtype=tf.float32))
        a = tf.get_variable(
            "dist_a"+str(action), initializer=tf.cast(10, dtype=tf.float32))
        dist = Dist(mu, inv_cov, a, b)
        return dist

    def _build_reset_priors_op(self):
        updates = [0]*self.num_actions
        for i in range(0, self.num_actions):
            updates[i] = Dist(self._dists[i].mean.assign(self.prior_0.mean),
                              self._dists[i].inv_cov.assign(
                                  self.prior_0.inv_cov),
                              self._dists[i].a.assign(self.prior_0.a),
                              self._dists[i].b.assign(self.prior_0.b))

        return updates

    def _build_parameters(self, size, action):
        reg_coef = tf.get_variable("reg_coef"+str(action), initializer=tf.cast(
            np.zeros((1, size)), dtype=tf.float32))
        noise_var = tf.get_variable(
            "noise_coef"+str(action), initializer=tf.cast(0.0, dtype=tf.float32))
        parameters = Parameters(reg_coef, noise_var)
        return parameters

    def _build_sample_coef_op(self):
        update_tensor = [0]*self.num_actions*2
        for action in range(0, self.num_actions):

            dist = self._dists[action]

            # Sample noise variance
            var_dist = tfd.InverseGamma(
                concentration=dist.a, rate=dist.b)
            var_sample = var_dist.sample(1)

            # Sample coefficients
            # tril = tf.linalg.inv(tf.cholesky(dist.inv_cov))
            tril = tf.linalg.inv(tf.cholesky((1/var_sample)*dist.inv_cov))
            coef_dist = tfd.MultivariateNormalTriL(
                loc=tf.reshape(dist.mean, [-1]), scale_tril=tril)
            coef_sample = coef_dist.sample(1)
            update_tensor[action] = self._parameters[action].coef.assign(
                coef_sample)
            update_tensor[self.num_actions+action] = self._parameters[action].noise.assign(
                tf.squeeze(var_sample))

        return update_tensor

    def _build_bayes_network(self):
        """Creates the bayes dists for each actors and connects bayesian regression
        to the DQN.
        """
        # Setup bayesian dists
        # with tf.variable_scope("Bayes"):
        self._dists = [0]*self.num_actions
        self._parameters = [0]*self.num_actions
        encoding_size = self._net_outputs.encoding.get_shape()[1]
        for i in range(0, self.num_actions):
            self._dists[i] = self._build_dist(encoding_size, i)
            self._parameters[i] = self._build_parameters(encoding_size, i)

        self.prior_0 = self._build_dist(encoding_size, -1)
        self._reset_priors_op = self._build_reset_priors_op()

        # Sampler
        self._sample_coef_op = self._build_sample_coef_op()

        # Action Selection
        # with tf.variable_scope('Online_Actions'):
        self._q_argmax = tf.argmax(
            self.samples_per_action(self._net_outputs.encoding), axis=0)
        self._q_values = self.samples_per_action(self._net_outputs.encoding)
        # Replay Setup
        # with tf.variable_scope('Bayes_Replay'):
        self._replay_target_net_outputs = self.target_convnet(
            self._replay.states)
        self._sample_replay_next_q_max = tf.reduce_max(
            self.samples_per_action(self._replay_next_target_net_outputs.encoding), axis=0)

    def bayesian_output(self, action, encoding):
        sample = encoding@tf.transpose(
            self._parameters[action].coef) + \
            tfd.Normal(loc=0, scale=tf.sqrt(
                self._parameters[action].noise)).sample(1)
        return tf.squeeze(sample)

    def bayesian_average(self, action, encoding):
        sample = encoding@tf.transpose(
            self._parameters[action].coef) + \
            tfd.Normal(loc=0, scale=tf.sqrt(
                self._parameters[action].noise)).sample(1)
        return tf.squeeze(sample)

    def samples_per_action(self, encoding):
        return tf.stack([self.bayesian_output(action, encoding) for action in range(0, self.num_actions)])

    def averages_per_action(self, encoding):
        return tf.stack([self.bayesian_average(action, encoding) for action in range(0, self.num_actions)], axis=1)

    def _build_update_bayes_op(self):

        updates = [0]*self.num_actions
        for action in range(0, self.num_actions):
            with tf.variable_scope('masked_target'):
                boolean_mask = tf.equal(self._replay.actions, action)
                rewards = tf.boolean_mask(self._replay.rewards, boolean_mask)
                terminals = tf.boolean_mask(
                    self._replay.terminals, boolean_mask)
                state_q_encoding = tf.boolean_mask(
                    self._replay_target_net_outputs.encoding, boolean_mask)
                next_state_q_encoding = tf.boolean_mask(
                    self._replay_next_target_net_outputs.encoding, boolean_mask)
                _next_replay_q_max = tf.reduce_max(
                    self.averages_per_action(next_state_q_encoding))

                print("NEXT replay q _max is ", _next_replay_q_max.get_shape())

                target = rewards + self.cumulative_gamma * \
                    _next_replay_q_max * \
                    (1. - tf.cast(terminals, tf.float32))

                updates[action] = self._update_single_dist_op(
                    self._dists[action], state_q_encoding, target)

                tf.summary.scalar(str(action),
                                  self._dists[action].b /
                                  (self._dists[action].a - 1),
                                  family="mean_noise")

        return updates

    def _rebuild_train_op(self):
        """Reuilds the training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """
        replay_action_one_hot = tf.one_hot(
            self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
        replay_chosen_q = tf.reduce_sum(
            self.averages_per_action(
                self._replay_next_target_net_outputs.encoding) * replay_action_one_hot,
            reduction_indices=1,
            name='replay_chosen_q')

        target = tf.stop_gradient(self._build_target_q_op())
        loss = tf.losses.huber_loss(
            target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('HuberLoss', tf.reduce_mean(loss))
        return self.optimizer.minimize(tf.reduce_mean(loss))

    def _update_single_dist_op(self, dist, encoding, target):
        with tf.variable_scope('dist_update'):
            xT = tf.transpose(encoding)
            xTx = xT@encoding

            target = tf.reshape(target, [-1, 1], name="target")

            update_inv_cov = xTx + dist.inv_cov

            update_mu = tf.linalg.inv(update_inv_cov)@ \
                (xT@target + dist.inv_cov@dist.mean)

            update_a = dist.a + tf.cast(tf.size(target)/2, tf.float32)

            update_b = dist.b + (dist.a-1)/dist.b*0.5 *\
                tf.squeeze(tf.transpose(target)@target +
                           tf.transpose(dist.mean)@dist.inv_cov@dist.mean -
                           tf.transpose(update_mu)@update_inv_cov@update_mu)

            tf.summary.scalar("mean",
                              tf.reduce_sum(dist.mean))

            tf.summary.scalar("cov",
                              tf.reduce_mean(tf.linalg.inv(dist.inv_cov)))

            update = Dist(dist.mean.assign(update_mu),
                          dist.inv_cov.assign(update_inv_cov),
                          dist.a.assign(update_a),
                          dist.b.assign(update_b))

        return update

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """

        # self._sess.run(self._sample_coef_op)
        # params = self._sess.run(self._parameters)
        # print("\nMean Action 1\n", np.sum(params[0].coef))
        # print("Mean Action 2\n", np.sum(params[1].coef))
        return super().begin_episode(observation)

    def _select_action(self):
        # if self.eval_mode:
            # epsilon = self.epsilon_eval
        #     else:
        #         epsilon = self.epsilon_fn(
        #             self.epsilon_decay_period,
        #             self.training_steps,
        #             self.min_replay_history,
        #             self.epsilon_train)
        #     if random.random() <= epsilon:
        #         # Choose a random action with probability epsilon.
        #         return random.randint(0, self.num_actions - 1)
        #     else:
        #         # Choose the action with highest Q-value at the current state.
        self._sess.run(self._sample_coef_op)
        action, q_values = self._sess.run([self._q_argmax, self._q_values], {
            self.state_ph: self.state})
        return action

    # def _select_action(self):
    #     """Select an action from the set of available actions.

    #     Chooses an action randomly with probability self._calculate_epsilon(), and
    #     otherwise acts greedily according to the current Q-value estimates.

    #     Returns:
    #        int, the selected action.
    #     """
    #     if self.eval_mode:
    #         epsilon = self.epsilon_eval
    #     else:
    #         epsilon = self.epsilon_fn(
    #             self.epsilon_decay_period,
    #             self.training_steps,
    #             self.min_replay_history,
    #             self.epsilon_train)
    #     if random.random() <= epsilon:
    #         # Choose a random action with probability epsilon.
    #         return random.randint(0, self.num_actions - 1)
    #     else:
    #         # Choose the action with highest Q-value at the current state.
    #         self._sess.run(self._sample_coef_op)

    #         action, q_values = self._sess.run([self._q_argmax, self._q_values], {
    #                                           self.state_ph: self.state})
    #         # print(q_values)
    #         return action

    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.memory.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                self._sess.run(self._train_op)
                if (self.summary_writer is not None and
                    self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary = self._sess.run(self._merged_summaries)
                    self.summary_writer.add_summary(
                        summary, self.training_steps)

            if self.training_steps % self.target_update_period == 0:
                self._sess.run(self._sync_qt_ops)

            if self.training_steps % self._bayes_update_period == 0:
                self._sess.run(self._reset_priors_op)
                for _ in range(0, 20):
                    self._sess.run(self._update_bayes_op)

                # dists = self._sess.run(self._dists)
                # for action in range(0, self.num_actions):
                #     print("Action mean: ", sum(dists[action].mean))

        self.training_steps += 1
