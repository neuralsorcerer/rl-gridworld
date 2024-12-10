import argparse
from environment import GridWorld
from agent import DQNAgent
from config import Config
from utils import train_agent, plot_rewards, plot_losses, plot_epsilons, get_optimal_path, render_optimal_path, animate_path, is_solvable
from logger import logger
import sys
import os

def parse_goal(goal_str):
    try:
        parts = goal_str.split()
        if len(parts) != 3:
            raise ValueError
        x, y, reward = map(int, parts)
        return (x, y), reward
    except ValueError:
        raise argparse.ArgumentTypeError("Goals must be in the format 'x y reward' with integer values.")

def validate_positions(start, goals, grid_size, obstacles):
    rows, cols = grid_size
    if not (0 <= start[0] < rows) or not (0 <= start[1] < cols):
        raise ValueError(f"Start position {start} is out of grid boundaries.")
    if start in obstacles:
        raise ValueError(f"Start position {start} overlaps with an obstacle.")
    for pos in goals.keys():
        if not (0 <= pos[0] < rows) or not (0 <= pos[1] < cols):
            raise ValueError(f"Goal position {pos} is out of grid boundaries.")
        if pos in obstacles:
            raise ValueError(f"Goal position {pos} overlaps with an obstacle.")

def main():
    parser = argparse.ArgumentParser(description='Train DQN on GridWorld.')
    parser.add_argument('--episodes', type=int, default=Config.EPISODES)
    parser.add_argument('--max_steps', type=int, default=Config.MAX_STEPS_PER_EPISODE)
    parser.add_argument('--log_interval', type=int, default=Config.LOG_INTERVAL)
    parser.add_argument('--rows', type=int, default=Config.DEFAULT_GRID_SIZE[0])
    parser.add_argument('--cols', type=int, default=Config.DEFAULT_GRID_SIZE[1])
    parser.add_argument('--start_x', type=int, default=Config.DEFAULT_START_POS[0])
    parser.add_argument('--start_y', type=int, default=Config.DEFAULT_START_POS[1])
    parser.add_argument('--goal', type=parse_goal, action='append')
    parser.add_argument('--save_animation', action='store_true')
    parser.add_argument('--animation_filename', type=str, default='path_animation.gif')

    args = parser.parse_args()

    if not args.goal:
        goals = Config.DEFAULT_GOALS.copy()
    else:
        goals = {position: reward for position, reward in args.goal}

    env = GridWorld(
        size=(args.rows, args.cols),
        start=(args.start_x, args.start_y),
        goals=goals,
        obstacles=Config.INITIAL_OBSTACLES,
        dynamic_obstacles=Config.DYNAMIC_OBSTACLES,
        max_steps=args.max_steps
    )

    agent = DQNAgent(
        state_size=args.rows * args.cols,
        action_size=Config.ACTION_SIZE
    )

    try:
        validate_positions(env.start, env.goals, env.size, env.obstacles)
    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        sys.exit(1)

    if not is_solvable(env):
        logger.error("The environment is not solvable. Please adjust obstacles or goals.")
        sys.exit(1)

    logger.info("Starting training...")
    rewards, losses, epsilons = train_agent(
        env, agent,
        state_size=agent.state_size,
        episodes=args.episodes,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        early_stop_threshold=Config.EARLY_STOP_AVG_REWARD_THRESHOLD,
        consecutive_success_threshold=Config.CONSECUTIVE_SUCCESS_THRESHOLD
    )

    if os.path.exists('models/best_dqn_model.pkl'):
        agent.load_model('models/best_dqn_model.pkl')
    else:
        logger.info("No best model found. Using current weights.")

    path, total_reward = get_optimal_path(env, agent, max_steps=args.max_steps)
    logger.info(f"Optimal Path Reward: {total_reward}")
    print(f"Optimal Path: {path}")
    render_optimal_path(env, path)
    animate_path(env, path, save=args.save_animation, filename=args.animation_filename)
    plot_rewards(rewards, filename='rewards_plot.png')
    plot_losses(losses, filename='losses_plot.png')
    plot_epsilons(epsilons, filename='epsilons_plot.png')

if __name__ == "__main__":
    main()
