import matplotlib.pyplot as plt
import numpy as np
from logger import logger
from config import Config
import matplotlib.animation as animation
from collections import deque


def one_hot(state, size):
    one_hot_state = np.zeros(size)
    one_hot_state[state] = 1.0
    return one_hot_state

def get_state_index(one_hot_state):
    return int(np.argmax(one_hot_state))

def is_solvable(env):
    start = env.start
    goals = set(env.goals.keys())
    obstacles = set(env.obstacles)
    visited = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current in goals:
            return True
        for action in range(Config.ACTION_SIZE):
            x, y = current
            if action == 0 and x > 0:
                new_pos = (x - 1, y)
            elif action == 1 and x < env.size[0] - 1:
                new_pos = (x + 1, y)
            elif action == 2 and y > 0:
                new_pos = (x, y - 1)
            elif action == 3 and y < env.size[1] - 1:
                new_pos = (x, y + 1)
            else:
                continue

            if new_pos not in obstacles and new_pos not in visited:
                visited.add(new_pos)
                queue.append(new_pos)

    return False

def train_agent(env, agent, state_size, episodes=Config.EPISODES, max_steps=Config.MAX_STEPS_PER_EPISODE,
               log_interval=Config.LOG_INTERVAL, early_stop_threshold=Config.EARLY_STOP_AVG_REWARD_THRESHOLD,
               consecutive_success_threshold=Config.CONSECUTIVE_SUCCESS_THRESHOLD):
    rewards = []
    losses = []
    steps_per_episode = []
    epsilons = []
    consecutive_successes = 0
    best_avg_reward = -float('inf')

    for episode in range(1, episodes + 1):
        state_index = env.reset()
        state = one_hot(state_index, size=state_size)
        total_reward = 0
        done = False

        for step in range(1, max_steps + 1):
            action = agent.choose_action(state)
            next_state_index, reward, done = env.step(action)
            next_state = one_hot(next_state_index, size=state_size)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()

            state = next_state
            total_reward += reward

            if loss is not None:
                losses.append(loss)

            if done:
                steps_per_episode.append(step)
                break
        else:
            steps_per_episode.append(max_steps)

        rewards.append(total_reward)
        epsilons.append(agent.epsilon)

        if total_reward >= max(env.goals.values()):
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        if episode % log_interval == 0:
            avg_steps = np.mean(steps_per_episode[-log_interval:])
            avg_reward = np.mean(rewards[-log_interval:])
            avg_loss = np.mean(losses[-log_interval:]) if len(losses) >= log_interval else np.mean(losses) if len(losses) > 0 else 0
            logger.info(f"Episode {episode}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}, "
                        f"Avg Steps: {avg_steps:.2f}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.2f}")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model('models/best_dqn_model.pkl')
                logger.info(f"New best average reward {avg_reward:.2f} achieved. Model saved.")

            if avg_reward >= early_stop_threshold:
                logger.info(f"Early stopping triggered at episode {episode} with average reward {avg_reward:.2f}")
                break

        if consecutive_successes >= consecutive_success_threshold:
            logger.info(f"Early stopping triggered at episode {episode} after {consecutive_successes} consecutive successes.")
            break

    return rewards, losses, epsilons


def plot_rewards(rewards, filename='rewards_plot.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_losses(losses, filename='losses_plot.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Loss per Training Step', color='orange')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_epsilons(epsilons, filename='epsilons_plot.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons, label='Epsilon', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()



def get_optimal_path(env, agent, max_steps=Config.MAX_STEPS_PER_EPISODE):
    state_index = env.reset()
    state = one_hot(state_index, size=agent.state_size)
    done = False
    path_indices = [state_index]
    total_reward = 0

    for _ in range(max_steps):
        q_values = agent.model.forward(state.reshape(1, -1))
        action = np.argmax(q_values[0])
        next_state_index, reward, done = env.step(action)
        path_indices.append(next_state_index)
        state = one_hot(next_state_index, size=agent.state_size)
        total_reward += reward
        if done:
            break

    path_positions = [env.get_position(idx) for idx in path_indices]
    return path_positions, total_reward

def render_optimal_path(env, path):
    env.render(path=path)

def animate_path(env, path, save=False, filename='path_animation.gif'):

    grid = [[' ' for _ in range(env.size[1])] for _ in range(env.size[0])]
    for obs in env.obstacles:
        grid[obs[0]][obs[1]] = 'X'
    for goal, _ in env.goals.items():
        grid[goal[0]][goal[1]] = 'G'
    grid[env.start[0]][env.start[1]] = 'S'

    fig, ax = plt.subplots()
    ax.set_xlim(0, env.size[1])
    ax.set_ylim(0, env.size[0])
    ax.set_xticks(np.arange(0, env.size[1], 1))
    ax.set_yticks(np.arange(0, env.size[0], 1))
    ax.grid(True)
    ax.invert_yaxis()

    obs_x, obs_y = zip(*env.obstacles) if env.obstacles else ([], [])
    goals_x, goals_y = zip(*env.goals.keys()) if env.goals else ([], [])
    start_x, start_y = env.start

    ax.scatter(obs_y, obs_x, marker='s', s=100, color='black', label='Obstacles')
    ax.scatter(goals_y, goals_x, marker='*', s=150, color='gold', label='Goals')
    ax.scatter(start_y, start_x, marker='o', s=100, color='green', label='Start')

    agent_dot, = ax.plot([], [], marker='o', color='blue', markersize=12, label='Agent')
    path_line, = ax.plot([], [], color='blue', linewidth=2, linestyle='--', label='Path')

    ax.legend(loc='upper right')

    def init():
        agent_dot.set_data([], [])
        path_line.set_data([], [])
        return agent_dot, path_line

    def update(frame):
        pos = path[frame]
        x = float(pos[1]) + 0.5
        y = float(pos[0]) + 0.5
        agent_dot.set_data([x], [y])
        current_path = path[:frame+1]
        path_x = [float(p[1]) + 0.5 for p in current_path]
        path_y = [float(p[0]) + 0.5 for p in current_path]
        path_line.set_data(path_x, path_y)
        return agent_dot, path_line

    ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init,
                                  blit=True, repeat=False, interval=300)

    if save:
        ani.save(filename, writer='pillow')
        logger.info(f"Animation saved to {filename}")
    else:
        plt.show()
