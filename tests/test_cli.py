import unittest
import argparse
from main import parse_obstacle, validate_positions

class TestCLIUtilities(unittest.TestCase):
    def test_parse_obstacle_valid(self):
        self.assertEqual(parse_obstacle("3 4"), (3, 4))

    def test_parse_obstacle_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_obstacle("3")

    def test_validate_positions_obstacle_checks(self):
        with self.assertRaises(ValueError):
            validate_positions((0,0), {(1,1):10}, (5,5), [(5,5)])
        with self.assertRaises(ValueError):
            validate_positions((0,0), {(1,1):10}, (5,5), [(0,0)])
        with self.assertRaises(ValueError):
            validate_positions((0,0), {(1,1):10}, (5,5), [(1,1)])

if __name__ == "__main__":
    unittest.main()