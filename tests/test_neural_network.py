import unittest
import numpy as np
from agent import NeuralNetwork
from config import Config

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(input_size=25, hidden_layers=Config.HIDDEN_LAYERS, output_size=4)

    def test_forward_output_shape(self):
        x = np.random.randn(64, 25)
        output = self.nn.forward(x)
        self.assertEqual(output.shape, (64, 4))

    def test_backward_no_errors(self):
        x = np.random.randn(64, 25)
        y_true = np.random.randn(64, 4)
        y_pred = self.nn.forward(x)
        try:
            self.nn.backward(x, y_true, y_pred)
        except Exception as e:
            self.fail(f"Backward pass crashed with exception: {e}")

    def test_relu(self):
        x = np.array([[-1, 0, 1], [2, -3, 4]])
        expected = np.array([[0, 0, 1], [2, 0, 4]])
        result = self.nn.relu(x)
        np.testing.assert_array_equal(result, expected)

    def test_relu_derivative(self):
        x = np.array([[-1, 0, 1], [2, -3, 4]])
        expected = np.array([[0, 0, 1], [1, 0, 1]])
        result = self.nn.relu_derivative(x)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
