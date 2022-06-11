from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random


ACTIONS = [np.array([0, 1]), np.array([1, 0])]


class CNNBlock(keras.layers.Layer):
    """
    CNNs Block

    Convolutional Neural Networks Block for DQN
    Input: Images of size (80 * 80 * 4)
    Output: Vectors of size (5 * 5 * 64)
    """

    def __init__(self, padding='same', activation='relu', 
                 w_init=keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                 b_init=keras.initializers.Constant(value=0.01)):
        super(CNNBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=32,
                                         kernel_size=(8, 8),
                                         strides=(4, 4),
                                         padding=padding,
                                         activation=activation,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init)
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                               padding=padding)
        self.conv2 = keras.layers.Conv2D(filters=64,
                                         kernel_size=(4, 4),
                                         strides=(2, 2),
                                         padding=padding,
                                         activation=activation,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init)
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                               padding=padding)
        self.conv3 = keras.layers.Conv2D(filters=64,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding=padding,
                                         activation=activation,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init)
        self.maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                               padding=padding)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # x = self.maxpool2(x)
        x = self.conv3(x)
        # x = self.maxpool3(x)
        return x


class MLPBlock(keras.layers.Layer):
    """
    MLPs Block

    Fully Connected MLPs following CNNs Block.
    Mapping the CNNs' output (i.e. state) to corresponding Q-values (i.e. Q(s, a)).
    Input: Vector of shape (5 * 5 * 64)
    Output: Vector of shape (num_action * 1)
    """

    def __init__(self, num_action=2, activation='relu', 
                 w_init=keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                 b_init=keras.initializers.Constant(value=0.01)):
        super(MLPBlock, self).__init__()
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(512, activation=activation,
                                      kernel_initializer=w_init,
                                      bias_initializer=b_init)
        self.fc2 = keras.layers.Dense(num_action,
                                      kernel_initializer=w_init,
                                      bias_initializer=b_init)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class DQN(keras.Model):
    def __init__(self, gamma=0.99):
        super(DQN, self).__init__()
        self.gamma = gamma
        self.net = self._build_dqn_model()
        self.target_net = self._build_dqn_model()

    @staticmethod
    def _build_dqn_model(optimizer=tf.optimizers.Adam,
                         loss_fn=keras.losses.MeanSquaredError,
                         lr=1e-6):
        net = keras.Sequential()
        net.add(CNNBlock())
        net.add(MLPBlock())
        net.compile(optimizer=optimizer(learning_rate=lr),
                    loss=loss_fn(reduction=tf.keras.losses.Reduction.SUM))
        return net

    def policy(self, state):
        input = tf.convert_to_tensor(state, dtype=tf.float32)
        input = tf.expand_dims(input=input, axis=0)
        q_s = self.net(input)
        # Build a best action vector e.g. np.array([0, 1])
        best_action = np.zeros_like(q_s)
        best_action[np.arange(best_action.shape[0]), np.argmax(q_s, axis=1)] = 1
        best_action = np.squeeze(best_action).astype(np.int64)
        # best_action = np.argmax(tf.squeeze(q_s), axis=0)
        return best_action, np.max(q_s)

    def random_policy(self):
        return random.choice(ACTIONS)

    def collect_policy(self, state, epsilon=0.1):
        action, q_max = self.policy(state)
        if np.random.random() < epsilon:
            print('-' * 20 + 'Random Move' + '-' * 20)
            return self.random_policy(), q_max
        else:
            return action, q_max

    def train(self, batch):
        state_samples, next_state_samples, action_samples, \
            reward_samples, terminate_samples = batch

        # q_current = self.net(state_samples).numpy()
        # q_target = np.zeros_like(q_current)
        q_target = self.net(state_samples).numpy()
        q_next = self.target_net(next_state_samples)
        best_q_next = np.amax(q_next, axis=1)
        for i in range(len(batch)):
            if terminate_samples[i]:
                q_target[i][np.argmax(action_samples[i])] = reward_samples[i]
            else:
                q_target[i][np.argmax(action_samples[i])] = reward_samples[i] + \
                    self.gamma * best_q_next[i]
        # train_history = self.net.fit(x=state_samples, y=q_target, verbose=0)
        # loss = train_history.history['loss']
        loss = self.net.train_on_batch(x=state_samples, y=q_target)
        return loss

    def update_target_net(self):
        self.target_net.set_weights(self.net.get_weights())

        

        
