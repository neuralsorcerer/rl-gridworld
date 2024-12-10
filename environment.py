from config import Config

class GridWorld:
    def __init__(self, size, start, goals, obstacles, dynamic_obstacles, max_steps):
        self.size = size  # (rows, cols)
        self.start = start
        self.goals = goals
        self.obstacles = obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.max_steps = max_steps
        self.current_pos = self.start
        self.step_count = 0
        self.previous_positions = set([self.start])

    def reset(self):
        self.current_pos = self.start
        self.step_count = 0
        self.previous_positions = set([self.start])
        return self.get_state(self.current_pos)

    def get_position(self, idx):
        x = idx // self.size[1]
        y = idx % self.size[1]
        return (x, y)

    def get_state(self, pos):
        x, y = pos
        return x * self.size[1] + y

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        x, y = self.current_pos
        current_distance = min(self.manhattan_distance(self.current_pos, goal) for goal in self.goals.keys())

        # Determine new position
        if action == 0 and x > 0:  # Up
            new_pos = (x - 1, y)
        elif action == 1 and x < self.size[0] - 1:  # Down
            new_pos = (x + 1, y)
        elif action == 2 and y > 0:  # Left
            new_pos = (x, y - 1)
        elif action == 3 and y < self.size[1] - 1:  # Right
            new_pos = (x, y + 1)
        else:
            # Invalid move, no position change
            new_pos = self.current_pos

        # Check conditions
        if new_pos in self.obstacles:
            reward = Config.OBSTACLE_COLLISION_PENALTY
            done = True
        elif new_pos in self.goals:
            reward = self.goals[new_pos]
            done = True
        elif new_pos == self.current_pos:
            # No movement scenario
            reward = Config.INVALID_MOVE_PENALTY
            done = False
        else:
            # Distance-based reward
            new_distance = min(self.manhattan_distance(new_pos, goal) for goal in self.goals.keys())
            distance_change = current_distance - new_distance
            if new_pos in self.previous_positions:
                reward = distance_change * Config.REWARD_FACTOR + Config.REVISIT_PENALTY
            else:
                reward = distance_change * Config.REWARD_FACTOR
            done = False

        if not done:
            self.previous_positions.add(new_pos)
        self.current_pos = new_pos
        self.step_count += 1

        # Check max steps
        if self.step_count >= self.max_steps:
            done = True

        return self.get_state(new_pos), reward, done

    def render(self, path=None):
        grid = [[' ' for _ in range(self.size[1])] for _ in range(self.size[0])]
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        for goal, reward in self.goals.items():
            grid[goal[0]][goal[1]] = 'G'
        grid[self.start[0]][self.start[1]] = 'S'

        if path:
            for pos in path:
                if pos != self.start and pos not in self.obstacles and pos not in self.goals:
                    grid[pos[0]][pos[1]] = '.'

        for row in grid:
            print("+---" * self.size[1] + "+")
            print("".join([f"| {cell} " for cell in row]) + "|")
        print("+---" * self.size[1] + "+")
        print("\nLegend: S = Start, G = Goal, X = Obstacle, . = Path")
