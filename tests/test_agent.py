import unittest
import numpy as np
from agent import DQNAgent
from config import Config
from environment import GridWorld
from utils import one_hot
import os

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(
            size=(5, 5),
            start=(0, 0),
            goals={(4, 4): 20},
            obstacles=[(1,1), (2,2), (3,3)],
            dynamic_obstacles=False,
            max_steps=50
        )
        self.agent = DQNAgent(
            state_size=25,
            action_size=4
        )

    def test_remember_and_replay(self):
        state = one_hot(0, size=25)
        next_state = one_hot(1, size=25)
        self.agent.remember(state, 0, 1, next_state, False)
        self.assertEqual(len(self.agent.memory), 1)

        for _ in range(Config.BATCH_SIZE):
            self.agent.remember(state, 1, -1, next_state, False)
        self.assertEqual(len(self.agent.memory), 1 + Config.BATCH_SIZE)

        try:
            _ = self.agent.replay()
        except Exception as e:
            self.fail(f"Replay method crashed with exception: {e}")

        self.assertLess(self.agent.epsilon, Config.EPSILON_START)

    def test_choose_action_explore(self):
        self.agent.epsilon = 1.0
        state = one_hot(0, size=25)
        action = self.agent.choose_action(state)
        self.assertIn(action, range(Config.ACTION_SIZE))

    def test_choose_action_exploit(self):
        self.agent.epsilon = 0.0
        state = one_hot(0, size=25)
        self.agent.model.forward = lambda x: np.array([[1, 3, 2, 0]])
        action = self.agent.choose_action(state)
        self.assertEqual(action, 1)

    def test_save_and_load_model(self):
        original_layers = [layer.copy() for layer in self.agent.model.layers]
        original_biases = [bias.copy() for bias in self.agent.model.biases]
        test_filename = 'models/test_dqn_model.pkl'
        self.agent.save_model(test_filename)

        for i in range(len(self.agent.model.layers)):
            self.agent.model.layers[i] = np.zeros_like(self.agent.model.layers[i])
            self.agent.model.biases[i] = np.zeros_like(self.agent.model.biases[i])

        self.agent.load_model(test_filename)
        for i in range(len(self.agent.model.layers)):
            np.testing.assert_array_equal(self.agent.model.layers[i], original_layers[i])
            np.testing.assert_array_equal(self.agent.model.biases[i], original_biases[i])

        if os.path.exists(test_filename):
            os.remove(test_filename)

if __name__ == '__main__':
    unittest.main()
