import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Wrapper para personalizar as recompensas
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_goal=10, reward_hole=-3, reward_step=-0.50):
        super().__init__(env)
        self.reward_goal = reward_goal
        self.reward_hole = reward_hole
        self.reward_step = reward_step

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        desc = self.unwrapped.desc
        row, col = divmod(obs, desc.shape[1])
        tile = desc[row][col].decode()

        if tile == 'G':
            reward = self.reward_goal
        elif tile == 'H':
            reward = self.reward_hole
        else:
            reward = self.reward_step

        return obs, reward, terminated, truncated, info

is_slippery = True
env_base = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode=None)
env_train = CustomRewardWrapper(env_base)

# Parâmetros do Q-learning
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05
num_episodes = 20000
max_steps = 30

Q = np.zeros((env_train.observation_space.n, env_train.action_space.n))
rewards_per_episode = []

for episode in range(1, num_episodes + 1):
    state, _ = env_train.reset()
    total_reward = 0

    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env_train.action_space.sample()
        else:
            action = np.argmax(Q[state])

        new_state, reward, terminated, truncated, _ = env_train.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
        state = new_state
        total_reward += reward

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

    if episode % 500 == 0:
        print(f"Episódio {episode}: recompensa = {total_reward:.2f}, ε = {epsilon:.3f}")

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.plot(moving_average(rewards_per_episode))
plt.title('Recompensa Média a Cada 100 Episódios')
plt.xlabel('Episódios')
plt.ylabel('Recompensa Média')
plt.grid()
plt.show()

def plot_policy_with_q_values(Q, env):
    import matplotlib.patches as patches
    action_arrows = ['←', '↓', '→', '↑']
    policy = np.argmax(Q, axis=1)
    q_values = np.max(Q, axis=1)
    desc = env.unwrapped.desc.astype(str)
    shape = desc.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Política + Valor Q no Frozen Lake", fontsize=14)
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, shape[1]-0.5)
    ax.set_ylim(-0.5, shape[0]-0.5)
    ax.grid(True)

    color_map = {'S': 'lightgreen', 'F': 'white', 'H': 'lightcoral', 'G': 'gold'}

    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = i * shape[1] + j
            tile = desc[i, j]
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=color_map.get(tile, 'white'),
                                     edgecolor='black')
            ax.add_patch(rect)
            ax.text(j, i - 0.25, tile, ha='center', va='center',
                    fontsize=12, weight='bold')

            if tile not in ['H', 'G']:
                arrow = action_arrows[policy[idx]]
                qv = q_values[idx]
                ax.text(j, i + 0.05, f"{arrow} ({qv:.2f})",
                        ha='center', va='center', fontsize=11)

    plt.gca().invert_yaxis()
    plt.show()

plot_policy_with_q_values(Q, env_train)

env_render = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode="human")
env_visual = CustomRewardWrapper(env_render)

def simulate_agent_visual(env, Q, sleep_time=0.8):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(sleep_time)

    print(f"\n🏁 Episódio finalizado. Recompensa total: {total_reward:.2f}")

simulate_agent_visual(env_visual, Q)

env_train.close()
env_visual.close()
