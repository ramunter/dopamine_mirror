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
nature_bdqn_network = atari_lib.bayesian_dqn_network

np.set_printoptions(precision=3)


@gin.configurable
class SimpleBDQNAgent(object):
    """An implementation of the DQN agent."""

    def __init__(self,
                 sess,
                 num_actions,
                 observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=atari_lib.NATURE_DQN_DTYPE,
                 stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
                 network=atari_lib.bayesian_dqn_network,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 sample_weight_period=8000,
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
                 summary_writing_frequency=10,
                 coef_var=0.01,
                 noise_var=1):
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
        assert isinstance(observation_shape, tuple)
        tf.logging.info('Creating %s agent with the following parameters:',
                        self.__class__.__name__)
        tf.logging.info('\t gamma: %f', gamma)
        tf.logging.info('\t update_horizon: %f', update_horizon)
        tf.logging.info('\t min_replay_history: %d', min_replay_history)
        tf.logging.info('\t update_period: %d', update_period)
        tf.logging.info('\t target_update_period: %d', target_update_period)
        tf.logging.info('\t tf_device: %s', tf_device)
        tf.logging.info('\t use_staging: %s', use_staging)
        tf.logging.info('\t optimizer: %s', optimizer)

        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size
        self.network = network
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.min_replay_history = min_replay_history
        self.target_update_period = target_update_period
        self.update_period = update_period
        self.sample_weight_period = sample_weight_period
        self.eval_mode = False
        self.training_steps = 0
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.summary_writing_frequency = summary_writing_frequency

        self.coef_var = coef_var
        with tf.device(tf_device):
            # Create a placeholder for the state input to the DQN network.
            # The last axis indicates the number of consecutive frames stacked.
            state_shape = (1,) + self.observation_shape + (stack_size,)
            self.state = np.zeros(state_shape)
            self.state_ph = tf.placeholder(self.observation_dtype, state_shape,
                                           name='state_ph')
            self._replay = self._build_replay_buffer(use_staging)

            self._build_networks()
            self._train_op = self._build_train_op()
            self._sync_qt_ops = self._build_sync_op()

        if self.summary_writer is not None:
            # All tf.summaries should have been defined prior to running this.
            self._merged_summaries = tf.summary.merge_all()

        self._sess = sess
        self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)

        # Variables to be initialized by the agent once it interacts with the
        # environment.
        self._observation = None
        self._last_observation = None

    @property
    def mean_initializer(self):
        return tf.cast(np.random.rand(self.num_actions, self.encoding_size), dtype=tf.float32)

    @property
    def weights_initializer(self):
        return tf.cast(np.random.rand(self.num_actions, self.encoding_size), dtype=tf.float32)

    @property
    def cov_initializer(self):
        cov = np.zeros(
            (self.num_actions, self.encoding_size, self.encoding_size))
        for i in range(self.num_actions):
            cov[i, :, :] = np.eye(self.encoding_size)*self.coef_var
        return tf.cast(cov, dtype=tf.float32)

    @property
    def cov_decomp_initializer(self):
        cov_decomp = np.zeros(
            (self.num_actions, self.encoding_size, self.encoding_size))
        cov = np.eye(self.encoding_size)*self.coef_var
        for i in range(self.num_actions):
            cov_decomp[i, :, :] = np.linalg.cholesky(
                ((cov+np.transpose(cov))/2.))  # Whats going on here?

        return tf.cast(cov_decomp, dtype=tf.float32)

    @property
    def phiphiT_initializer(self):
        return tf.cast(tf.zeros((self.num_actions, self.encoding_size, self.encoding_size)), dtype=tf.float32)

    @property
    def phiY_initializer(self):
        return tf.cast(np.zeros((self.num_actions, self.encoding_size)), dtype=tf.float32)

    @property
    def alpha_initializer(self):
        return tf.cast(1*np.ones((self.num_actions)), dtype=tf.float32)

    @property
    def beta_initializer(self):
        return tf.cast(1*np.ones((self.num_actions)), dtype=tf.float32)
    
    @property
    def noise_initializer(self):
        return tf.cast(np.zeros((self.num_actions)), dtype=tf.float32)

    def _get_network_type(self):
        """Returns the type of the outputs of a BDQN value network.
        Returns:
          net_type: _network_type object defining the outputs of the network.
        """
        return collections.namedtuple('BDQN_network', ['encoding'])

    def _network_template(self, state):
        """Builds the convolutional network used to compute the agent's Q-values.

        Args:
        state: `tf.Tensor`, contains the agent's current state.

                value=self.cov_decomp_initializer, validate_shape=True))
        Returns:
                value=self.cov_decomp_initializer, validate_shape=True))
        net: _network_type object containing the tensors output by the network.
                value=self.cov_decomp_initializer, validate_shape=True))
        """
        return self.network(self.num_actions, self._get_network_type(), state)

    def _build_networks(self):
        """Builds the Q-value network computations needed for acting and training.

        These are:
        self.online_convnet: For computing the current state's Q-values.
        self.target_convnet: For computing the next state's target Q-values.
        self._net_outputs: The actual Q-values.
        self._q_argmax: The action maximizing the current state's Q-values.
        self._replay_net_outputs: The replayed states' Q-values.
        self._replay_next_target_net_outputs: The replayed next states' target
            Q-values (see Mnih et al., 2015 for details).
        """
        # Calling online_convnet will generate a new graph as defined in
        # self._get_network_template using whatever input is passed, but will always
        # share the same weights.
        self.online_convnet = tf.make_template(
            'Online', self._network_template)
        self.target_convnet = tf.make_template(
            'Target', self._network_template)
        self._net_outputs = self.online_convnet(self.state_ph)
        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        # using a deep network, but may affect performance with a linear
        # approximation scheme.

        # Build bayes reg variables

        self.encoding_size = int(self._net_outputs.encoding.get_shape()[1])

        self.mean = tf.get_variable(
            "parameters/mean", initializer=self.mean_initializer, trainable=False)
        self.weight_samples = tf.get_variable(
            "parameters/weight_samples", initializer=self.weights_initializer, trainable=False)
        self.cov = tf.get_variable(
            "parameters/cov", initializer=self.cov_initializer, trainable=False)
        self.cov_decomp = tf.get_variable(
            "parameters/cov_decomp", initializer=self.cov_decomp_initializer, trainable=False)
        self.phiphiT = tf.get_variable(
            "parameters/phiphiT", initializer=self.phiphiT_initializer, trainable=False)
        self.phiY = tf.get_variable(
            "parameters/phiY", initializer=self.phiY_initializer, trainable=False)
        self.noise_var = tf.get_variable("parameters/noise_var", initializer=self.noise_initializer, trainable=False)
        self.a = tf.get_variable(
            "parameters/alpha", initializer=self.alpha_initializer, trainable=False)
        self.b = tf.get_variable(
            "parameters/beta", initializer=self.beta_initializer, trainable=False)

        self.reset_priors_op = self.build_reset_priors_op()
        self.sample_weights_op = self.build_sample_weights_op()

        self._q_argmax = tf.argmax(
            tf.matmul(self.mean, self._net_outputs.encoding,
                      transpose_b=True), name="Mean_Q")[0]
        self._sample_q_argmax = tf.argmax(
            tf.matmul(self.weight_samples,
                      self._net_outputs.encoding, transpose_b=True) +
            tfd.MultivariateNormalDiag(
                loc=[0]*self.num_actions, scale_diag=self.noise_var).sample(1),
            name="Sample_Q")[0]

        self._replay_net_outputs = self.online_convnet(self._replay.states)
        self._replay_next_target_net_outputs = self.target_convnet(
            self._replay.next_states)
        self._replay_next_net_outputs = self.online_convnet(
            self._replay.next_states)

        self.bayes_reg_op = self.build_bayes_reg_op()

    def build_sample_weights_op(self):
        with tf.name_scope("Sample_Weights"):
            samples = []
            noise_var = []
            for a in range(self.num_actions):

                samples.append(tf.squeeze(
                    tfd.MultivariateNormalTriL(
                        loc=self.mean[a, :],
                        scale_tril=self.cov_decomp[a, :, :]).sample(1)
                        )
                )

                tf.summary.histogram(
                    str(a), samples[a], family="Weight_Histograms")

                var_dist = tfd.InverseGamma(concentration=self.a[a], rate=self.b[a])
                noise_var.append(var_dist.sample(1))

                if self.summary_writer:
                    tf.summary.histogram(str(a), var_dist.sample(1000), family="noise_dist")

                    temp = tf.squeeze(
                        tfd.MultivariateNormalTriL(
                            loc=self.mean[a, :],
                            scale_tril=self.cov_decomp[a, :, :]).sample(1000)
                        )

                    for weight in range(self.encoding_size):
                        tf.summary.histogram(
                            str(weight), temp[weight, :], family="per_weight_dist")

            weight_samples = tf.stack(samples, axis=0)
            noise_var = tf.squeeze(tf.stack(noise_var))

            return [tf.assign(self.weight_samples, weight_samples, validate_shape=True),
                    tf.assign(self.noise_var, noise_var, validate_shape=True)]

    def build_reset_priors_op(self):
        with tf.name_scope("Reset_Priors"):
            priors = []
            priors.append(
                tf.assign(self.phiphiT, value=self.phiphiT_initializer, validate_shape=True))
            priors.append(
                tf.assign(self.phiY, value=self.phiY_initializer, validate_shape=True))
            priors.append(
                tf.assign(self.a, value=self.alpha_initializer, validate_shape=True))
            priors.append(
                tf.assign(self.b, value=self.beta_initializer, validate_shape=True))
            return priors

    def build_bayes_reg_op(self):
        with tf.name_scope("Bayes_Reg"):

            updates = [0]*self.num_actions
            for action in range(0, self.num_actions):
                boolean_mask = tf.equal(self._replay.actions, action)
                rewards = tf.boolean_mask(self._replay.rewards, boolean_mask)
                terminals = tf.boolean_mask(
                    self._replay.terminals, boolean_mask)
                state_q_encoding = tf.boolean_mask(
                    self._replay_net_outputs.encoding, boolean_mask)
                next_state_q_encoding = tf.boolean_mask(
                    self._replay_next_target_net_outputs.encoding, boolean_mask)

                replay_next_q_argmax=tf.one_hot(
                    tf.argmax(tf.matmul(
                        self.mean, next_state_q_encoding, transpose_b=True)), self.num_actions, name="argmax_next_q")

                replay_next_qt_max=tf.reduce_sum(
                    tf.transpose(tf.matmul(self.weight_samples, next_state_q_encoding,
                                        transpose_b=True)) * replay_next_q_argmax,
                    reduction_indices=1,
                    name='qt_max')

                target = rewards + self.cumulative_gamma * \
                    replay_next_qt_max * (1. - tf.cast(terminals, tf.float32))

                updates[action] = self._update_single_prior_op(
                    action, state_q_encoding, target)

            return updates

    def _update_single_prior_op(self, action, encoding, target):
        target = tf.reshape(target, [-1, 1], name="target")

        phiphiT = self.phiphiT[action, :, :] + \
            tf.matmul(encoding, encoding, transpose_a=True)
        phiY = self.phiY[action, :, None] + \
            tf.matmul(encoding, target, transpose_a=True)

        inv_cov = phiphiT + \
            tf.linalg.inv(tf.eye(self.encoding_size)*self.coef_var)
        cov = tf.linalg.inv(inv_cov)
        mean = cov@phiY

        # phiphiT = tf.matmul(encoding, encoding, transpose_a=True)
        # phiY = tf.matmul(encoding, target, transpose_a=True)

        # inv_cov = phiphiT + \
        #     tf.linalg.inv(self.cov[action,:,:])

        # cov = self.noise_var*tf.linalg.inv(inv_cov)

        # mean = cov@(phiY+inv_cov@self.mean[action,:,None])

        cov_decomp = tf.linalg.cholesky((cov+tf.transpose(cov))/2.)

        a = self.a[action] + tf.cast(tf.size(target)/2, tf.float32)

        b = self.b[action] + 0.5 *\
            tf.squeeze(tf.transpose(target)@target +
                        tf.transpose(mean)@inv_cov@mean -
                        tf.transpose(self.mean[action, :, None])@tf.linalg.inv(self.cov[action, :, :])@self.mean[action, :, None])

        tf.summary.scalar(str(action), a,  family="alpha")

        tf.summary.scalar(str(action), b, family="beta")
        update=[tf.assign(self.phiphiT[action, :, :], phiphiT),
                  tf.assign(self.phiY[action, :], tf.squeeze(phiY)),
                  tf.assign(self.mean[action, :], tf.squeeze(mean)),
                  tf.assign(self.cov[action, :, :], cov),
                  tf.assign(self.cov_decomp[action, :, :], cov_decomp),
                  tf.assign(self.a[action], a),
                  tf.assign(self.b[action], b)]

        return update

    def _build_replay_buffer(self, use_staging):
        """Creates the replay buffer used by the agent.

        Args:
        use_staging: bool, if True, uses a staging area to prefetch data for
            faster training.

        Returns:
        A WrapperReplayBuffer object.
        """
        return circular_replay_buffer.WrappedReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype)

    def _build_target_q_op(self):
        """Build an op used as a target for the Q-value.

        Returns:
        target_q_op: An op calculating the Q-value.
        """
        # Get the maximum Q-value across the actions dimension.
        # replay_next_qt_max = tf.reduce_max(
        #               self._replay_next_target_net_outputs.q_values,
        #     axis=1)
        with tf.name_scope("Calc_Target"):

            replay_next_q_argmax=tf.one_hot(
                tf.argmax(tf.matmul(
                    self.weight_samples, self._replay_next_net_outputs.encoding, transpose_b=True)), self.num_actions, name="argmax_next_q")

            replay_next_qt_max=tf.reduce_sum(
                tf.transpose(tf.matmul(self.mean, self._replay_next_target_net_outputs.encoding,
                                    transpose_b=True)) * replay_next_q_argmax,
                reduction_indices=1,
                name='qt_max')

            qt=tf.matmul(self.mean,
                        self._replay_next_target_net_outputs.encoding,
                        transpose_b=True)

            qs=tf.matmul(self.weight_samples,
                        self._replay_next_target_net_outputs.encoding,
                        transpose_b=True)

            tf.summary.scalar("sample_diff", tf.reduce_mean(qt-qs))

            tf.summary.scalar("mean", tf.reduce_mean(
                qt, axis=1)[0], family="q-values0")
            tf.summary.scalar("mean", tf.reduce_mean(
                qt, axis=1)[1], family="q-values1")

            tf.summary.scalar("sample", tf.reduce_mean(
                qs, axis=1)[0], family="q-values0")
            tf.summary.scalar("sample", tf.reduce_mean(
                qs, axis=1)[1], family="q-values1")

            return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
                1. - tf.cast(self._replay.terminals, tf.float32))

    def _build_train_op(self):
        """Builds a training op.

        Returns:
        train_op: An op performing one step of training from replay data.
        """

        target=tf.stop_gradient(self._build_target_q_op())

        with tf.name_scope("Loss"):

            replay_action_one_hot=tf.one_hot(
                self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
            replay_chosen_q=tf.reduce_sum(
                tf.transpose(tf.matmul(self.mean, self._replay_net_outputs.encoding,
                                    transpose_b=True)) * replay_action_one_hot,
                reduction_indices=1,
                name='replay_chosen_q')

            loss=tf.losses.huber_loss(
                target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)

            if self.summary_writer is not None:
                    tf.summary.scalar('HuberLoss', tf.reduce_mean(loss))
            return self.optimizer.minimize(tf.reduce_mean(loss))


    def _build_sync_op(self):
        """Builds ops for assigning weights from online to target network.

        Returns:
        ops: A list of ops assigning weights from online to target network.
        """
        # Get trainable variables from online and target DQNs
        sync_qt_ops=[]
        trainables_online=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Online')
        trainables_target=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target')
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            # Assign weights from online to target network.
            sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
        return sync_qt_ops

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.

        Args:
        observation: numpy array, the environment's initial observation.

        Returns:
        int, the selected action.
        """
        self._reset_state()
        self._record_observation(observation)

        if not self.eval_mode:
            self._train_step()

        self.action=self._select_action()
        return self.action

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
        reward: float, the reward received from the agent's most recent action.
        observation: numpy array, the most recent observation.

        Returns:
        int, the selected action.
        """
        self._last_observation=self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(
                self._last_observation, self.action, reward, False)
            self._train_step()

        self.action=self._select_action()
        return self.action

    def end_episode(self, reward):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Args:
        reward: float, the last reward from the environment.
        """
        if not self.eval_mode:
            self._store_transition(
                self._observation, self.action, reward, True)

    def _select_action(self):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
        int, the selected action.
        """
        if self.eval_mode:
            return self._sess.run(self._q_argmax, {self.state_ph: self.state})

        return self._sess.run(self._sample_q_argmax, {self.state_ph: self.state})

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
                
                # Train step
                self._sess.run(self._train_op)
                self._sess.run(self.sample_weights_op)

                # Add summary things
                if (self.summary_writer is not None and
                    self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary=self._sess.run(self._merged_summaries)
                    self.summary_writer.add_summary(
                        summary, self.training_steps)

            # Retrain bayes reg
            if self.training_steps % self.target_update_period == 0:
                self._sess.run([self._sync_qt_ops, self.reset_priors_op])
                training_iterations=int(min(
                    self._replay.memory.add_count, self.target_update_period*10)/self._replay.batch_size)

                for _ in range(training_iterations):
                    self._sess.run(self.bayes_reg_op)
                self._sess.run(self.sample_weights_op)


        self.training_steps += 1

    def _record_observation(self, observation):
        """Records an observation and update state.

        Extracts a frame from the observation vector and overwrites the oldest
        frame in the state buffer.

        Args:
            observation: numpy array, an observation from the environment.
        """
        # Set current observation. We do the reshaping to handle environments
        # without frame stacking.
        self._observation=np.reshape(observation, self.observation_shape)
        # Swap out the oldest frame with the current frame.
        self.state=np.roll(self.state, -1, axis=-1)
        self.state[0, ..., -1]=self._observation

    def _store_transition(self, last_observation, action, reward, is_terminal):
        """Stores an experienced transition.

        Executes a tf session and executes replay buffer ops in order to store the
        following tuple in the replay buffer:
            (last_observation, action, reward, is_terminal).

        Pedantically speaking, this does not actually store an entire transition
        since the next state is recorded on the following time step.

        Args:
            last_observation: numpy array, last observation.
            action: int, the action taken.
            reward: float, the reward.
            is_terminal: bool, indicating if the current state is a terminal state.
        """
        self._replay.add(last_observation, action, reward, is_terminal)

    def _reset_state(self):
        """Resets the agent state by filling it with zeros."""
        self.state.fill(0)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Returns a self-contained bundle of the agent's state.

        This is used for checkpointing. It will return a dictionary containing all
        non-TensorFlow objects (to be saved into a file by the caller), and it saves
        all TensorFlow objects into a checkpoint file.

        Args:
            checkpoint_dir: str, directory where TensorFlow objects will be saved.
            iteration_number: int, iteration number to use for naming the checkpoint
            file.

        Returns:
            A dict containing additional Python objects to be checkpointed by the
            experiment. If the checkpoint directory does not exist, returns None.
        """
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        # Call the Tensorflow saver to checkpoint the graph.
        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)
        # Checkpoint the out-of-graph replay buffer.
        self._replay.save(checkpoint_dir, iteration_number)
        bundle_dictionary={}
        bundle_dictionary['state']=self.state
        bundle_dictionary['eval_mode']=self.eval_mode
        bundle_dictionary['training_steps']=self.training_steps
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
            agent's state.

        Args:
            checkpoint_dir: str, path to the checkpoint saved by tf.Save.
            iteration_number: int, checkpoint version, used when restoring replay
            buffer.
            bundle_dictionary: dict, containing additional Python objects owned by
            the agent.

        Returns:
            bool, True if unbundling was successful.
        """
        try:
            # self._replay.load() will throw a NotFoundError if it does not find all
            # the necessary files, in which case we abort the process & return False.
            self._replay.load(checkpoint_dir, iteration_number)
        except tf.errors.NotFoundError:
            return False
        for key in self.__dict__:
            if key in bundle_dictionary:
                self.__dict__[key]=bundle_dictionary[key]
        # Restore the agent's TensorFlow graph.
        self._saver.restore(self._sess,
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True
