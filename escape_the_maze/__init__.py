from gymnasium.envs.registration import register

register(
    id="EscapeTheMaze-v0",
    entry_point="escape_the_maze.envs:EscapeTheMazeEnv",
    max_episode_steps=200
)