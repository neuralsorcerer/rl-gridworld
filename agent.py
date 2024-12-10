import numpy as np
import random
from collections import deque
import pickle
import os
from config import Config
from logger import logger

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layers = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) / np.sqrt(layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.layers.append(weight)
            self.biases.append(bias)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.z = []
        self.a = [x]

        for i in range(len(self.layers)):
            z = np.dot(self.a[-1], self.layers[i]) + self.biases[i]
            self.z.append(z)
            if i < len(self.layers) - 1:
                a = self.relu(z)
            else:
                a = z
            self.a.append(a)

        return self.a[-1]

    def backward(self, x, y_true, y_pred, learning_rate=Config.LEARNING_RATE):
        d_layers = [None] * len(self.layers)
        d_biases = [None] * len(self.biases)

        loss_derivative = 2 * (y_pred - y_true) / y_true.shape[0]
        delta = loss_derivative

        for i in reversed(range(len(self.layers))):
            a_prev = self.a[i]
            d_layers[i] = np.dot(a_prev.T, delta)
            d_biases[i] = np.sum(delta, axis=0, keepdims=True)

            if i != 0:
                z_prev = self.z[i-1]
                delta = np.dot(delta, self.layers[i].T) * self.relu_derivative(z_prev)

        for i in range(len(self.layers)):
            self.layers[i] -= learning_rate * d_layers[i]
            self.biases[i] -= learning_rate * d_biases[i]


class DQNAgent:
    def __init__(self, state_size, action_size, memory_size=Config.MEMORY_SIZE, batch_size=Config.BATCH_SIZE,
                 gamma=Config.GAMMA, epsilon_start=Config.EPSILON_START, epsilon_min=Config.EPSILON_MIN,
                 epsilon_decay=Config.EPSILON_DECAY, learning_rate=Config.LEARNING_RATE,
                 target_update_freq=Config.TARGET_UPDATE_FREQ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.model = NeuralNetwork(state_size, Config.HIDDEN_LAYERS, action_size)
        self.target_model = NeuralNetwork(state_size, Config.HIDDEN_LAYERS, action_size)
        self.update_target_network()

    def update_target_network(self):
        for i in range(len(self.model.layers)):
            self.target_model.layers[i] = self.model.layers[i].copy()
            self.target_model.biases[i] = self.model.biases[i].copy()
        logger.info("Target network updated.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.forward(state.reshape(1, -1))
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        q_current = self.model.forward(states)
        q_next = self.target_model.forward(next_states)

        q_target = q_current.copy()
        max_q_next = np.amax(q_next, axis=1)
        for i in range(self.batch_size):
            if dones[i]:
                q_target[i][actions[i]] = rewards[i]
            else:
                q_target[i][actions[i]] = rewards[i] + self.gamma * max_q_next[i]

        loss = np.mean((q_current - q_target) ** 2)
        self.model.backward(states, q_target, q_current, learning_rate=self.learning_rate)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'layers': [layer.copy() for layer in self.model.layers],
                'biases': [bias.copy() for bias in self.model.biases]
            }, f)
        logger.info(f"Model saved to {filename}")

    def load_model(self, filename):
        if not os.path.exists(filename):
            logger.info(f"No model file {filename} found.")
            return
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            for i in range(len(self.model.layers)):
                self.model.layers[i] = params['layers'][i].copy()
                self.model.biases[i] = params['biases'][i].copy()
        self.update_target_network()
        logger.info(f"Model loaded from {filename}")
