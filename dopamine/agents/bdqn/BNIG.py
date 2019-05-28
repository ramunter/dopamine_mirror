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
                 coef_var=1e3,
                 lr=1e-4):
        self.action = action

        # External Tensors
        self._replay = replay_buffer

        self.input_size =  int(state.get_shape()[1])
        self.coef_var = coef_var
        self.mem = 1-lr

        with tf.name_scope("BNIG"):

            self._create_model_variables()

            # Graphs
            self.sample_op = self._sampler_graph(state)
            self.target_sample_op = self._sampler_graph(replay_next_state, n=self._replay.batch_size)

    @property
    def mean_prior(self):
        return tf.cast(np.zeros((self.input_size)), dtype=tf.float32)

    @property
    def cov_prior(self):
        return tf.cast(np.eye(self.input_size)*self.coef_var, dtype=tf.float32)

    @property
    def alpha_prior(self):
        return tf.cast(1, dtype=tf.float32)

    @property
    def beta_prior(self):
        return tf.cast(1e-3, dtype=tf.float32)

    def _create_model_variables(self):
        with tf.variable_scope(str(self.action)+"/parameters/"):
            self.mean = tf.get_variable("mean",  initializer=self.mean_prior, trainable=False)
            self.cov  = tf.get_variable("cov",   initializer=self.cov_prior, trainable=False)
            self.alpha= tf.get_variable("alpha", initializer=self.alpha_prior, trainable=False)
            self.beta = tf.get_variable("beta",  initializer=self.beta_prior, trainable=False)

        with tf.variable_scope(str(self.action)+"/data/"):
            self.XTX = tf.get_variable("XTX", initializer=tf.cast(np.zeros((self.input_size, self.input_size)), dtype=tf.float32), trainable=False)
            self.XTy = tf.get_variable("XTy", initializer=tf.cast(np.zeros((self.input_size, 1)), dtype=tf.float32), trainable=False)
            self.n   = tf.get_variable("n",   initializer=tf.cast(0, dtype=tf.float32), trainable=False)
            self.yTy = tf.get_variable("yTy", initializer=tf.cast(np.zeros((1,1)), dtype=tf.float32), trainable=False)


    def _sampler_graph(self, _input, n=1):
        sigma_dist = tfd.InverseGamma(concentration=self.alpha, rate=self.beta)
        sigma = tf.sqrt(sigma_dist.sample(1))
        normal = tfd.Normal(loc=0, scale=1)
        coef = self.mean[:,None] + sigma*tf.linalg.cholesky(self.cov)@normal.sample((self.input_size,n))
        return tf.reduce_sum(_input*tf.transpose(coef), axis=1) + normal.sample(n)*sigma

    def _build_update_op(self, state, target):

        with tf.name_scope("posterior_update"):
            
            target.set_shape([self._replay.batch_size])

            # Filter relevant data for action
            boolean_mask = tf.equal(self._replay.actions, self.action)
            num_samples = tf.reduce_sum(tf.cast(boolean_mask, tf.int32))

            state = tf.boolean_mask(state, boolean_mask)
            target = tf.boolean_mask(target, boolean_mask)
            terminal = tf.boolean_mask((tf.cast(self._replay.terminals, tf.float32)[:,None]), boolean_mask)

            update_ops = self._build_bayes_posterior_update(state, target[:, None], terminal, num_samples)

        return update_ops

    def _build_bayes_posterior_update(self, X, y, terminal, n):

            XTX = self.mem*self.XTX + tf.transpose(X)@X
            XTy = self.mem*self.XTy + tf.transpose(X)@y

            y_t = y*(1.-terminal) + X@self.mean[:,None]*terminal
            
            yTy = self.mem*self.yTy + tf.transpose(y_t)@y_t
            n   = self.mem*self.n   + tf.cast(n, tf.float32)

            inv_cov = XTX + tf.linalg.inv(self.cov_prior)
            temp = tf.linalg.inv(tf.linalg.cholesky(inv_cov))
            cov = tf.transpose(temp)@temp
    
            mean = cov@XTy

            alpha = self.alpha_prior + \
                tf.cast(n/2, tf.float32)

            beta = tf.maximum(self.beta_prior + \
                    0.5*tf.squeeze(yTy -
                        tf.transpose(mean)@inv_cov@mean),
                    1e-6)
            
            tf.summary.scalar(str(self.action), alpha, family="alpha")
            tf.summary.scalar(str(self.action), beta, family="beta")

            return  [tf.assign(self.mean , tf.squeeze(mean)),
                    tf.assign(self.cov  , cov),
                    tf.assign(self.alpha, alpha),
                    tf.assign(self.beta , beta),
                    tf.assign(self.XTX  , XTX),
                    tf.assign(self.XTy  , XTy),
                    tf.assign(self.n    , n),
                    tf.assign(self.yTy  , yTy)]

    def reset_variance_op(self):
        temp = tf.linalg.inv(tf.linalg.cholesky(self.cov))
        invcov = tf.transpose(temp)@temp
        return tf.assign(self.yTy , tf.transpose(self.mean[:,None])@invcov@self.mean[:,None])