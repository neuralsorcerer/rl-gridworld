class Config:
    # ------------------------------
    # GridWorld Configuration
    # ------------------------------
    DEFAULT_GRID_SIZE = (8, 10)
    DEFAULT_START_POS = (0, 0)
    DEFAULT_GOALS = {
        (4, 4): 50,
    }
    INITIAL_OBSTACLES = [
        (1, 1), (2, 2), (3, 4), (6, 1), (6, 2), (7, 0), (7, 2), (7, 3), (7, 6), (7, 8)
    ]
    # Set to True to randomly move obstacles after each agent step
    DYNAMIC_OBSTACLES = False
    MAX_STEPS_PER_EPISODE = 50

    # ------------------------------
    # DQN Agent Configuration
    # ------------------------------
    ACTION_SIZE = 4
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 0.001
    TARGET_UPDATE_FREQ = 500

    # ------------------------------
    # Neural Network Architecture
    # ------------------------------
    HIDDEN_LAYERS = [128, 64, 32]

    # ------------------------------
    # Training Configuration
    # ------------------------------
    EPISODES = 5000
    LOG_INTERVAL = 100

    # ------------------------------
    # Early Stopping Configuration
    # ------------------------------
    CONSECUTIVE_SUCCESS_THRESHOLD = 100
    EARLY_STOP_AVG_REWARD_THRESHOLD = 35

    # ------------------------------
    # Logging Configuration
    # ------------------------------
    LOG_FILE = 'logs/rl_gridworld.log'

    # ------------------------------
    # Reward Configuration
    # ------------------------------
    REWARD_FACTOR = 1.0
    REVISIT_PENALTY = -1
    OBSTACLE_COLLISION_PENALTY = -10
    INVALID_MOVE_PENALTY = -1
