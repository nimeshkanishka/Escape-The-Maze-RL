# Escape The Maze: A Custom [Gymnasium](https://gymnasium.farama.org/index.html) Environment for Reinforcement Learning

Thanks to [Python Simplified on YouTube](https://www.youtube.com/@PythonSimplified) for the idea. This is essentially an implementation of the example environment in her reinforcement learning explanation [video](https://youtu.be/sgsQZmlJbSY).

## Installation

1. Clone the repository

```bash
git clone https://github.com/nimeshkanishka/Escape-The-Maze-RL.git
```

2. Install the package

```bash
cd Escape-The-Maze-RL
pip install .
```

## Usage

The example below demonstrates how to create, interact with, and visualize the environment.

```python
import gymnasium as gym
import escape_the_maze

env = gym.make("EscapeTheMaze-v0", render_mode="human")

observation, info = env.reset()
episode_length = 0
episode_reward = 0
terminated, truncated = False, False

while not (terminated or truncated)
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_length += 1
    episode_reward += reward

print(f"Episode finished in {episode_length} steps with total reward {episode_reward}.")

env.close()
```

The example below demonstrates how to train and visualize an agent on the environment using [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)'s [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) algorithm.

```python
import gymnasium as gym
import escape_the_maze
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

def train(
    total_timesteps: int = 2_000_000,
    num_parallel_envs: int = 8,
    seed: int | None = None,
    device: str = "auto"
) -> None:
    # Vectorized environment for training
    env = make_vec_env(
        env_id="EscapeTheMaze-v0",
        n_envs=num_parallel_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv
    )

    # Evaluation environment
    eval_env = gym.make("EscapeTheMaze-v0")

    # SB3's DQN algorithm for training
    model = DQN(
        policy="MlpPolicy",
        env=env,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=64,
        target_update_interval=5_000,
        exploration_fraction=0.5,
        tensorboard_log="logs", # directory to save tensorboard logs
        seed=seed,
        device=device
    )
    
    # Perform an evaluation every 20k steps (2500 * 8) and save model checkpoint if reward has improved
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=2500,
        best_model_save_path="models" # directory to save model checkpoints
    )

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        reset_num_timesteps=False,
        progress_bar=True
    )

def eval(
    total_episodes: int,
    render: bool = True,
    device: str = "auto"
) -> None:
    env = gym.make("EscapeTheMaze-v0", render_mode="human" if render else None)

    # Load the saved best model checkpoint
    model = DQN.load(
        path="models/best_model",
        device=device
    )

    episode_length_history = []
    episode_reward_history = []

    for episode in range(total_episodes):
        observation, _ = env.reset()
        episode_length = 0
        episode_reward = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action, _ = model.predict(observation=observation, deterministic=True)

            observation, reward, terminated, truncated, _ = env.step(action)

            episode_length += 1
            episode_reward += reward

        episode_length_history.append(episode_length)
        episode_reward_history.append(episode_reward)
        print(f"Episode: {episode + 1}/{total_episodes} - Length: {episode_length} - Reward: {episode_reward:.2f}")

    print(f"\n{total_episodes} episodes completed!")
    print(f"Episode length: {np.mean(episode_length_history):.2f} +/- {np.std(episode_length_history):.2f}")
    print(f"Episode reward: {np.mean(episode_reward_history):.2f} +/- {np.std(episode_reward_history):.2f}")

if __name__ == "__main__":
    # Train for 2M timesteps
    train()

    # Watch the trained agent play 5 episodes
    eval(total_episodes=5)
```

## Environment Details

### Description

The robot wakes up in a maze. There's an exit but he doesn't know where it is. He must find it as soon as possible but any wrong move could be costly. He will get electrocuted if he touches a wall. He must collect as many gold chunks and diamonds before he leaves the maze. But there is only a limited time to escape the maze. If he does not escape in time... 💀

### Observation Space

The observation is a ```ndarray``` of shape ```(51,)```, where the first 49 elements contain a flattened 7x7 grid around the agent and the last 2 elements are the number of gold chunks and diamonds collected by the agent.

The grid values have the following encoding:

| Value    | Meaning           |
|----------|-------------------|
| ```-1``` | Out of bounds     |
| ```0```  | Empty             |
| ```1```  | Wall              |
| ```2```  | Agent             |
| ```3```  | Gold              |
| ```4```  | Diamond           |
| ```5```  | Exit              |

### Action Space

There are 4 discrete actions:

```0```: Move up

```1```: Move down

```2```: Move left

```3```: Move right

### Rewards

The agent is rewarded for exploring new cells (```+0.02```) and penalized for visiting already visited cells (```-0.01 * number_of_times_visited```; capped at ```-0.05```). Collecting gold chunks and diamonds is rewarded with ```+1.0``` and ```+2.0``` respectively. A penalty of ```-0.5``` is given for getting electrocuted (hitting a wall). Finding the exit before the end of the episode will be rewarded with ```+5.0```, with an extra bonus for each gold chunk (```+3.0```) and diamond (```+5.0```) collected.

### Episode End

Termination: When the agent finds the exit of the maze

Truncation: When the episode length is greater than 200

### Arguments

```python
env = gym.make("EscapeTheMaze-v0", render_mode=None, map_csv=None)
```

- ```render_mode: str | None = None```

  Render mode of the environment.
  
  ```render_mode="human"``` visualizes the environment using [pygame](https://www.pygame.org/wiki/about).

- ```map_csv: str | Path | None = None```

  Path to a CSV file defining the maze layout.

  The CSV should be 2D with the following encoding:

  | Value   | Meaning           |
  |---------|-------------------|
  | ```0``` | Empty cell        |
  | ```1``` | Wall              |
  | ```2``` | Starting position |
  | ```3``` | Gold              |
  | ```4``` | Diamond           |
  | ```5``` | Exit              |

  If ```None```, the default layout will be used.

### Version History

v1: Change observation space and reward function

v0: Initial version