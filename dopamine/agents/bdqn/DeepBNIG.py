import collections
import math
import os
import random

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gin.tf

from dopamine.agents.bdqn.BNIG import BNIG

tfd = tfp.distributions

@gin.configurable
class DeepBNIG(BNIG):
    def __init__(self,
                 action,
                 state,
                 replay_next_state,
                 replay_buffer,
                 coef_var=1,
                 lr=1e-3):
        self.action = action

        # External Tensors
        self._replay = replay_buffer

        self.input_size =  int(state.get_shape()[1])
        self.coef_var = coef_var
        self.mem = 1-lr
        self.scope_name = "DeepBNIG/"+str(action)

        with tf.name_scope(self.scope_name):
            self.normal = tfd.Normal(loc=0, scale=1)

            self._create_model_variables()
 
            # Graphs
            self.sample_op = self._sampler_graph(state)
            self.target_sample_op = self._target_sampler_graph(replay_next_state, n=self._replay.batch_size)

            self.update_normal_vector_op = tf.assign(self.normal_vector, self.normal_sample)

    @property
    def mean_prior(self):
        return super().mean_prior

    @property
    def cov_prior(self):
        return super().cov_prior

    @property
    def alpha_prior(self):
        return super().alpha_prior

    @property
    def beta_prior(self):
        return super().beta_prior

    @property
    def normal_sample(self):
        return self.normal.sample((self.input_size, 1))

    def _create_model_variables(self):
        with tf.variable_scope(self.scope_name+"/parameters/"):
            self.mean = tf.get_variable("mean",  initializer=self.mean_prior, trainable=False)
            self.cov  = tf.get_variable("cov",   initializer=self.cov_prior, trainable=False)
            self.alpha= tf.get_variable("alpha", initializer=self.alpha_prior, trainable=False)
            self.beta = tf.get_variable("beta",  initializer=self.beta_prior, trainable=False)
            
            self.tar_mean  = tf.get_variable("tar_mean",  initializer=self.mean_prior, trainable=False)
            self.tar_cov   = tf.get_variable("tar_cov",   initializer=self.cov_prior, trainable=False)
            self.tar_alpha = tf.get_variable("tar_alpha", initializer=self.alpha_prior, trainable=False)
            self.tar_beta  = tf.get_variable("tar_beta",  initializer=self.beta_prior, trainable=False)

        with tf.variable_scope(self.scope_name+"/parameters/"):
            self.XTX = tf.get_variable("XTX", initializer=tf.cast(np.zeros((self.input_size, self.input_size)), dtype=tf.float32), trainable=False)
            self.XTy = tf.get_variable("XTy", initializer=tf.cast(np.zeros((self.input_size, 1)), dtype=tf.float32), trainable=False)
            self.n   = tf.get_variable("n",   initializer=tf.cast(0, dtype=tf.float32), trainable=False)
            self.yTy = tf.get_variable("yTy", initializer=tf.cast(np.zeros((1,1)), dtype=tf.float32), trainable=False)

        with tf.variable_scope(self.scope_name+"/random/"):
            self.normal_vector = tf.get_variable("normal_vector",  initializer=self.normal_sample, trainable=False)

    def _sampler_graph(self, _input, n=1):
        sigma_dist = tfd.InverseGamma(concentration=self.alpha, rate=self.beta)
        sigma = tf.sqrt(sigma_dist.sample(1))
        coef = self.mean[:,None] + sigma*tf.linalg.cholesky(self.cov)@self.normal_vector
        normal = tfd.Normal(loc=0, scale=1)
        return tf.reduce_sum(_input*tf.transpose(coef), axis=1) + normal.sample(n)*sigma

    def _target_sampler_graph(self, _input, n=1):
        return super()._target_sampler_graph(_input, n)

    def _build_update_op(self, state, target, target_var):
        return super()._build_update_op(state, target, target_var)

    def sync_target(self):
        return super().sync_target()