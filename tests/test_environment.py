import unittest
import random
from environment import GridWorld
from config import Config

class TestGridWorld(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(
            size=(8, 10),
            start=(0, 0),
            goals={(4, 4): 50},
            obstacles=[(1, 1), (2, 2), (3, 4), (6, 1), (6, 2), (7, 0), (7, 2), (7, 3), (7, 6), (7, 8)],
            dynamic_obstacles=False,
            max_steps=50
        )
        self.env.reset()

    def test_reset(self):
        state = self.env.reset()
        self.assertEqual(state, self.env.get_state((0, 0)))
        self.assertEqual(self.env.current_pos, (0, 0))
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.obstacles, Config.INITIAL_OBSTACLES)

    def test_step_up(self):
        self.env.reset()
        next_state, reward, done = self.env.step(0)  # Up from (0, 0) invalid
        self.assertEqual(next_state, self.env.get_state((0, 0)))
        self.assertEqual(reward, Config.INVALID_MOVE_PENALTY)
        self.assertFalse(done)

    def test_step_down(self):
        self.env.reset()
        next_state, reward, done = self.env.step(1)  # Down to (1,0)
        expected_pos = (1, 0)
        self.assertEqual(next_state, self.env.get_state(expected_pos))
        initial_distance = self.env.manhattan_distance((0,0),(4,4))
        new_distance = self.env.manhattan_distance((1,0),(4,4))
        distance_change = initial_distance - new_distance
        expected_reward = distance_change * Config.REWARD_FACTOR
        self.assertEqual(reward, expected_reward)
        self.assertFalse(done)

    def test_step_into_obstacle(self):
        self.env.current_pos = (1, 0)
        next_state, reward, done = self.env.step(3) # Right into (1,1) obstacle
        self.assertEqual(next_state, self.env.get_state((1,1)))
        self.assertEqual(reward, Config.OBSTACLE_COLLISION_PENALTY)
        self.assertTrue(done)

    def test_revisit_state(self):
        self.env.reset()
        # Move to (0,1)
        self.env.current_pos = (0,1)
        self.env.previous_positions.add((0,0))
        # Move left back to (0,0)
        _, reward, done = self.env.step(2)
        # Distance (0,1)->(4,4) = 7
        # Distance (0,0)->(4,4) = 8
        # distance_change = 7-8 = -1
        # revisit penalty = -1
        # total = -1 + (-1) = -2
        self.assertEqual(reward, -2)
        self.assertFalse(done)

    def test_no_change_in_distance(self):
        self.env.reset()
        # Try invalid move up from (0,0)
        _, reward, done = self.env.step(0)
        self.assertEqual(reward, Config.INVALID_MOVE_PENALTY)
        self.assertFalse(done)

    def test_moving_away(self):
        self.env.reset()
        self.env.current_pos = (2,0)
        self.env.previous_positions = {(0,0),(2,0)}
        # Move up: (2,0) to (1,0)
        # Distance(2,0->4,4)=|2-4|+|0-4|=6
        # Distance(1,0->4,4)=|1-4|+|0-4|=7
        # distance_change=6-7=-1
        # No revisit penalty if not visited (1,0) before
        # Actually (1,0) not in previous_positions, so reward = -1
        _, reward, done = self.env.step(0)
        self.assertEqual(reward, -1)
        self.assertFalse(done)

    def test_max_steps_exceeded(self):
        for _ in range(self.env.max_steps):
            _, _, done = self.env.step(random.choice([0,1,2,3]))
            if done:
                break
        self.assertTrue(done)

    def test_dynamic_obstacles_move(self):
        env = GridWorld(
            size=(5, 5),
            start=(0, 0),
            goals={(4, 4): 10},
            obstacles=[(2, 2)],
            dynamic_obstacles=True,
            max_steps=10,
        )
        env.reset()
        random.seed(0)
        initial_obstacles = env.obstacles.copy()
        env.step(1)
        self.assertNotEqual(env.obstacles, initial_obstacles)
        for obs in env.obstacles:
            self.assertTrue(0 <= obs[0] < env.size[0] and 0 <= obs[1] < env.size[1])
            self.assertNotEqual(obs, env.start)
            self.assertNotIn(obs, env.goals)
            self.assertNotEqual(obs, env.current_pos)

if __name__ == "__main__":
    unittest.main()
