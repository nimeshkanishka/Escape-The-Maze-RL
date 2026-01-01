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
total_steps = 0
total_reward = 0
done = False

while not done:
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_steps += 1
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished in {total_steps} steps with total reward {total_reward}.")

env.close()
```

The example below demonstrates how to train an agent on the environment using [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)'s [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) algorithm.

```python
import gymnasium as gym
import escape_the_maze
from stable_baselines3 import DQN

env = gym.make("EscapeTheMaze-v0")

model = DQN(
    policy="MlpPolicy",
    env=env,
    buffer_size=250_000,
    exploration_fraction=0.5,
    exploration_final_eps=0.02
)

model.learn(total_timesteps=1_000_000)
model.save("escape_the_maze_agent")
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