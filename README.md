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

The robot wakes up in a maze. There is only one exit. He must find it as soon as possible but any wrong move could be costly. Touching a wall is not an option; he will get electrocuted. He must collect as many gold chunks and diamonds before he leaves the maze. But there is only a limited time to escape the maze. If he does not escape in time... 💀

### Observation Space

The observation, a ```ndarray``` with shape ```(3, 3)```, is a 3x3 grid around the agent consisting of the state (```-1```: Out of bounds, ```0```: Empty, ```1```: Wall, ```2```: Agent, ```3```: Gold, ```4```: Diamond, ```5```: Exit) of each cell.

### Action Space

There are 4 discrete actions:

```0```: Move up

```1```: Move down

```2```: Move left

```3```: Move right

### Rewards

The agent is rewarded for collecting gold chunks (```+1.5```) and diamonds (```+2```). A penalty of ```-1``` is given for getting electrocuted (hitting a wall). Aditionally, a small penalty (```-0.25```) is given for each passing timestep. Finding the exit before the end of the episode will be rewarded with ```+10```.

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

v0: Initial version