from enum import Enum

from numpy.random import choice, binomial
from numpy import arange, zeros, array

import gym
from gym import spaces
from gym.utils import seeding

import tensorflow as tf
import gin.tf

gin.constant('corridor_lib.CORRIDOR_OBSERVATION_SHAPE', (2, 1))
gin.constant('corridor_lib.CORRIDOR_OBSERVATION_DTYPE', tf.float32)
gin.constant('corridor_lib.CORRIDOR_STACK_SIZE', 1)

slim = tf.contrib.slim

# Action "Enum"
LEFT = 0
RIGHT = 1

@gin.configurable
def create_corridor_environment(N=10):
    return Corridor(N)

@gin.configurable
def _basic_discrete_domain_network(num_actions, state,
                                   num_atoms=None):
    """Builds a basic network for discrete domains, rescaling inputs to [-1, 1].

    Args:
      min_vals: float, minimum attainable values (must be same shape as `state`).
      max_vals: float, maximum attainable values (must be same shape as `state`).
      num_actions: int, number of actions.
      state: `tf.Tensor`, the state input.
      num_atoms: int or None, if None will construct a DQN-style network,
        otherwise will construct a Rainbow-style network.

    Returns:
      The Q-values for DQN-style agents or logits for Rainbow-style agents.
    """
    net = tf.cast(state, tf.float32)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 24)
    if num_atoms is None:
        # We are constructing a DQN-style network.
        return slim.fully_connected(net, num_actions, activation_fn=None)
    else:
        # We are constructing a rainbow-style network.
        return slim.fully_connected(net, num_actions * num_atoms,
                                    activation_fn=None)


@gin.configurable
def _bayesian_discrete_domain_network(num_actions, state):
    """Builds a basic network for discrete domains, rescaling inputs to [-1, 1].

    Args:
      min_vals: float, minimum attainable values (must be same shape as `state`).
      max_vals: float, maximum attainable values (must be same shape as `state`).
      num_actions: int, number of actions.
      state: `tf.Tensor`, the state input.
      num_atoms: int or None, if None will construct a DQN-style network,
        otherwise will construct a Rainbow-style network.

    Returns:
      The Q-values for DQN-style agents or logits for Rainbow-style agents.
    """
    net = tf.cast(state, tf.float32)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 24)
    # We are constructing a BDQN-style network.
    return net

@gin.configurable
def corridor_bdqn_network(num_actions, network_type, state):
    """Builds the deep network used to compute the agent's Q-values.

    It rescales the input features to a range that yields improved performance.

    Args:
      num_actions: int, number of actions.
      network_type: namedtuple, collection of expected values to return.
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    net = _bayesian_discrete_domain_network(num_actions, state)
    return network_type(net)


@gin.configurable
def corridor_dqn_network(num_actions, network_type, state):
    """Builds the deep network used to compute the agent's Q-values.

    It rescales the input features to a range that yields improved performance.

    Args:
      num_actions: int, number of actions.
      network_type: namedtuple, collection of expected values to return.
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    q_values = _basic_discrete_domain_network(num_actions, state)
    return network_type(q_values)

class Corridor(gym.Env):
    """
    Corridor environment.  
    This game is allows the reproduction of the example used to explain 
    policy gradients in Sutton on page 323 and the chain example used by 
    Osband. Reaching the goal gives a reward of +1.  

    args:  
        N: The environment consists of a N long corridor where the agent can
           move left or right.  
        K: In a K number of states, the direction traveled when given is the
           opposite of the expected. I.e. action left will cause the agent to
           move right.    
        p: Probability of success when moving right. 1-p probability of moving
           left instead.  

    Code built based on the Chain environment in AI GYM
    """

    def __init__(self, N=3, K=0, p=1):
        self.seed()
        self.N = N

        self.reverse_states = choice(arange(N), size=K, replace=False)
        self.p = p

        self.state = 1  # Start at beginning of the chain
        self.steps = 1
        self.max_steps = N
        self.game_over = False
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.N)

    @property
    def state_output(self):
        return array([self.steps/self.N, self.state/self.N])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        self.steps += 1

        action = self.env_changes_to_actions(action)
        self.transition(action)
        reward = self.reward_calculator(action)

        if self.steps >= self.max_steps:
            self.game_over = True
        else:
            self.game_over = False


        return self.state_output, reward, self.game_over, {}

    def env_changes_to_actions(self, action):

        # If in a reverse state swap action.
        if self.state in self.reverse_states:
            action = 1 - action

        # If trying to move right there is a prob of moving left
        if action == RIGHT:
            action = binomial(1, p=self.p)  # p prob of right

        return action

    def transition(self, action):

        if action == LEFT:
            if self.state != 1:
                self.state -= 1

        elif action == RIGHT and self.state < self.N:  # 'forwards action'
            self.state += 1

    def reward_calculator(self, action):

        if self.state == self.N:
            reward = 1
        elif action == 0:
            reward = 1/(10*self.N)
        else:
            reward = 0

        return reward

    def reset(self):
        self.state = 1
        self.steps = 1
        self.game_over = False
        return self.state_output
