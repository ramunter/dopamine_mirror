import collections
import math
import os
import random

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gin.tf

tfd = tfp.distributions

class BNIG():
    def __init__(self,
                 action,
                 state,
                 replay_next_state,
                 replay_buffer,
                 coef_var=1,
                 lr=0.01):
        self.action = action

        # External Tensors
        self._replay = replay_buffer

        self.input_size =  int(state.get_shape()[1])
        self.coef_var = coef_var

        with tf.name_scope("BNIG"):

            # Variables
            self.mean = tf.get_variable(
                str(action)+"/parameters/mean", initializer=self.mean_initializer, trainable=False)
            self.cov = tf.get_variable(
                str(action)+"/parameters/cov", initializer=self.cov_initializer, trainable=False)
            self.a = tf.get_variable(
                str(action)+"/parameters/alpha", initializer=self.alpha_initializer, trainable=False)
            self.b = tf.get_variable(
                str(action)+"/parameters/beta", initializer=self.beta_initializer, trainable=False)

            # Graphs
            self.sample_op = self._sampler_graph(state)
            self.target_sample_op = self._sampler_graph(replay_next_state, n=self._replay.batch_size)

    @property
    def mean_initializer(self):
        return tf.cast(np.zeros((self.input_size)), dtype=tf.float32)

    @property
    def cov_initializer(self):
        return tf.cast(np.eye(self.input_size)*self.coef_var, dtype=tf.float32)

    @property
    def alpha_initializer(self):
        return tf.cast(1, dtype=tf.float32)

    @property
    def beta_initializer(self):
        return tf.cast(1e-3, dtype=tf.float32)

    def _sampler_graph(self, _input, n=1):
        sigma_dist = tfd.InverseGamma(concentration=self.a, rate=self.b)
        sigma = tf.sqrt(sigma_dist.sample(1))
        normal = tfd.Normal(loc=0, scale=1)
        coef = self.mean[:,None] + sigma*tf.linalg.cholesky(self.cov)@normal.sample((self.input_size,n))
        return tf.reduce_sum(_input*tf.transpose(coef), axis=1) + normal.sample(n)*sigma

    def _build_update_op(self, state, target):

        with tf.name_scope("posterior_update"):
            
            # Filter relevant data for action
            boolean_mask = tf.equal(self._replay.actions, self.action)
            num_samples = tf.reduce_sum(tf.cast(boolean_mask, tf.int32))

            state = tf.boolean_mask(state, boolean_mask)
            target = tf.boolean_mask(target, boolean_mask)

            update_ops = self._build_bayes_posterior_update(state, target[:, None], num_samples)

        return update_ops

    def _build_bayes_posterior_update(self, X, y, n):
            inv_cov = tf.transpose(X)@X + tf.linalg.inv(self.cov)
            cov = tf.linalg.inv(inv_cov)
    
            mean = cov@(tf.transpose(X)@y +
                        tf.linalg.inv(self.cov)@self.mean[:,None])
            print_op1 = tf.print("X Max:", tf.reduce_max(X), "Min", tf.reduce_min(X))
            print_op2 = tf.print("Y Max:", tf.reduce_max(y), "Min", tf.reduce_min(y))

            # lr = 0.01
            # mean = (1-lr)*self.mean[:,None] + lr*mean
            # cov = (1-lr)*self.cov + lr*cov

            alpha = self.a + \
                tf.cast(n/2, tf.float32)

            b = self.b + 0.5*\
                tf.squeeze(tf.transpose(y)@y -
                            tf.transpose(mean)@inv_cov@mean + 
                            tf.transpose(self.mean[:,None])@
                            tf.linalg.inv(self.cov)@self.mean[:,None])
            
            tf.summary.scalar(str(self.action), alpha, family="alpha")
            tf.summary.scalar(str(self.action), b, family="beta")
            
            # print_op = tf.print( -
            #                 tf.transpose(mean)@inv_cov@mean + 
            #                 tf.transpose(self.mean[:,None])@
            #                 tf.linalg.inv(self.cov)@self.mean[:,None])
            
            with tf.control_dependencies([print_op1,print_op2]):
                return  [tf.assign(self.mean, tf.squeeze(mean)),
                        tf.assign(self.cov, cov),
                        tf.assign(self.a, alpha),
                        tf.assign(self.b, b)]
