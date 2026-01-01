from pathlib import Path
from importlib.resources import files
import gymnasium as gym
import numpy as np
import pygame

class EscapeTheMazeEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 5
    }

    def __init__(
        self,
        render_mode: str | None = None,
        map_csv: str | Path | None = None
    ):
        self.render_mode = render_mode

        # Load maze layout from CSV file
        # 0: Empty, 1: Wall, 2: Starting position, 3: Gold, 4: Diamond, 5: Exit
        if map_csv is not None:
            # Get absolute path to user-provided map
            map_path = Path(map_csv).expanduser().resolve()
            if not map_path.exists():
                raise FileNotFoundError(f"Map file '{map_path}' not found.")
        else:
            # Default packaged map
            map_path = files("escape_the_maze.envs").joinpath("assets/maps/level1.csv")
        # Load grid values from CSV file
        self.grid_values = np.loadtxt(map_path, dtype=int, delimiter=",")
        # Ensure the grid is 2D
        if self.grid_values.ndim != 2:
            raise ValueError("Map CSV must be 2D.")
        
        # Number of gold chunks and diamonds in the maze
        num_gold_chunks = np.argwhere(self.grid_values == 3).shape[0]
        num_diamonds = np.argwhere(self.grid_values == 4).shape[0]
        
        # Initialize variables
        self.grid = None
        self.agent_position = None
        self.times_visited = None
        self.is_electrocuted = None
        self.last_action = None
        self.gold_collected = None
        self.diamonds_collected = None
        self.times_electrocuted = None

        # Pygame variables
        self.screen = None
        self.clock = None
        self.cell_size = 32
        self.grid_offset = (100, 50, 100, 100) # top, bottom, left, right
        self.wall_width = 8
        self.screen_size = None
        self.agent_img = None
        self.gold_img = None
        self.diamond_img = None
        self.flag_img = None
        self.gold_icon_img = None
        self.diamond_icon_img = None
        self.spark_icon_img = None
        self.font = None
        
        # Observation space
        # This contains a flattened 7x7 grid around the agent (first 49 elements of the observation vector)
        # and the number of gold chunks and diamonds collected by the agent (last 2 elements)
        # The grid values have the following encoding:
        # -1: Out of bounds, 0: Empty, 1: Wall, 2: Agent, 3: Gold, 4: Diamond, 5: Exit
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=max(num_gold_chunks, num_diamonds, 5),
            shape=(51,),
            dtype=np.int8
        )

        # Action space
        # Contains 4 discrete actions
        # 0: Move up, 1: Move down, 2: Move left, 3: Move right
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the grid
        self.grid = self.grid_values.copy()

        # Set agent position to starting position
        self.agent_position = tuple(np.argwhere(self.grid == 2)[0])
        ay, ax = self.agent_position

        # At the start of the episode, the agent hasnt visited any cells
        self.times_visited = np.zeros(self.grid.shape, dtype=np.uint8)
        # Mark the cell it starts on as visited
        self.times_visited[ay, ax] += 1

        # Reset other variables
        self.is_electrocuted = False
        self.last_action = None
        self.gold_collected = 0
        self.diamonds_collected = 0
        self.times_electrocuted = 0

        self.render()
        
        return self._get_observation(), {}

    def step(self, action):
        if action < 0 or action > 3:
            raise ValueError(
                f"Invalid action ({action}). Action must be an integer in [0, 3]."
            )
        
        # Save the action
        # This is needed for animating the agent hitting walls and getting electrocuted
        self.last_action = action

        reward = 0.0
        terminated = False

        # Offsets of neighboring cells (up, down, left, right)
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Calculate new position of the agent
        new_position = (
            self.agent_position[0] + neighbor_offsets[action][0],
            self.agent_position[1] + neighbor_offsets[action][1]
        )

        ay, ax = self.agent_position
        ny, nx = new_position

        if self.grid[ny, nx] == 1:
            # If the agent is trying to move to a cell with a wall, animate the agent
            # getting electrocuted by calling render while is_electrocuted flag is True
            self.is_electrocuted = True
            self.times_electrocuted += 1
            self.render()
            # Reset the flag and penalize the agent
            self.is_electrocuted = False
            reward -= 0.5

        else:
            # If the agent is moving to another cell, mark the cell it was on as empty
            self.grid[ay, ax] = 0

            # Update agent's position
            self.agent_position = new_position

            # If the agent has moved to a cell it has never been to before,
            # give a small reward to encourage exploration
            if self.times_visited[ny, nx] == 0:
                reward += 0.02
            # Otherwise, penalize the agent for visiting an already visited cell
            # This penalty depends on how many times it has visited the same cell previously
            # It is capped at -0.05 to ensure it is not too large
            else:
                reward -= min(0.01 * self.times_visited[ay, ax], 0.05)
            # Mark the cell as visited
            self.times_visited[ny, nx] += 1

            # If the agent collects a gold chunk, reward the agent
            if self.grid[ny, nx] == 3:
                self.gold_collected += 1
                reward += 1.0
            
            # If the agent collects a diamond, reward the agent
            elif self.grid[ny, nx] == 4:
                self.diamonds_collected += 1
                reward += 2.0

            # If the agent reaches the exit, reward the agent and terminate the episode
            # There is an extra bonus for each gold chunk (+3) and diamond (+5) collected to
            # encourage the agent to explore the maze rather than exiting as fast as possible
            elif self.grid[ny, nx] == 5:
                reward += 5 + 3 * self.gold_collected + 5 * self.diamonds_collected
                terminated = True

            # Mark agent's new position on the grid
            self.grid[ny, nx] = 2
            
        self.render()

        return self._get_observation(), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            self._render_human()

        return

    def _get_observation(self):
        # Initialize local grid with -1 (out of bounds)
        grid = np.full(shape=(7, 7), fill_value=-1, dtype=np.int8)

        h, w = self.grid.shape
        ay, ax = self.agent_position

        # Loop over each cell on a 7x7 grid around the agent
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                # Calculate x and y coordinates of each cell
                y, x = ay + dy, ax + dx

                # If the cell is within bounds, set its value in the grid
                if 0 <= y < h and 0 <= x < w:
                    grid[dy + 3, dx + 3] = self.grid[y, x]

        # The observation vector also contains the number of gold chunks and diamonds collected so far
        loot = np.array(
            [self.gold_collected, self.diamonds_collected],
            dtype=np.int8
        )

        # Return the full observation vector
        return np.concatenate((grid.flatten(), loot), axis=0)

    def _render_human(self):
        h, w = self.grid.shape

        # Initialize pygame once
        if self.screen is None:
            # If grid is too large, disable rendering
            if h > 25 or w > 50:
                print("Warning: Grid is too large for visualization. Disabling rendering.")
                self.render_mode = None
                return
            
            pygame.init()
            
            self.screen_size = (
                self.grid_offset[2] + w * self.cell_size + self.grid_offset[3],
                self.grid_offset[0] + h * self.cell_size + self.grid_offset[1]
            )
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Escape The Maze")

            self.clock = pygame.time.Clock()

            # Load images
            sprites_path = files("escape_the_maze.envs").joinpath("assets/sprites")
            self.agent_img = pygame.image.load(sprites_path.joinpath("agent.png")).convert_alpha()
            self.gold_img = pygame.image.load(sprites_path.joinpath("gold.png")).convert_alpha()
            self.diamond_img = pygame.image.load(sprites_path.joinpath("diamond.png")).convert_alpha()
            self.flag_img = pygame.image.load(sprites_path.joinpath("flag.png")).convert_alpha()
            # Icons for the HUD
            self.gold_icon_img = pygame.image.load(sprites_path.joinpath("gold_icon.png")).convert_alpha()
            self.diamond_icon_img = pygame.image.load(sprites_path.joinpath("diamond_icon.png")).convert_alpha()
            self.spark_icon_img = pygame.image.load(sprites_path.joinpath("spark_icon.png")).convert_alpha()

            # Load the custom font
            font_path = files("escape_the_maze.envs").joinpath("assets/fonts/SourceCodePro-Regular.ttf")
            self.font = pygame.font.Font(font_path, 28)

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Dark blue background
        self.screen.fill((10, 20, 40))

        # Draw HUD
        # Number of gold chunks collected
        self.screen.blit(self.gold_icon_img, (20, 22))
        gold_text = self.font.render(str(self.gold_collected), True, (255, 255, 255))
        self.screen.blit(gold_text, (55, 15))
        # Number of diamonds collected
        self.screen.blit(self.diamond_icon_img, (95, 22))
        diamond_text = self.font.render(str(self.diamonds_collected), True, (255, 255, 255))
        self.screen.blit(diamond_text, (130, 15))
        # Number of times electrocuted
        self.screen.blit(self.spark_icon_img, (self.screen_size[0] - 90, 18))
        electrocutions_text = self.font.render(str(self.times_electrocuted), True, (255, 255, 255))
        self.screen.blit(electrocutions_text, (self.screen_size[0] - 50, 15))

        # Color of the walls based on whether the agent is being electrocuted
        # Yellow if electrocuted, blue otherwise
        wall_color = (210, 225, 0) if self.is_electrocuted else (65, 130, 255)

        # Loop over each cell on the grid
        for y in range(h):
            for x in range(w):
                # Get current cell value
                cell_value = self.grid[y, x]

                # Skip empty cells
                if cell_value == 0:
                    continue

                # Draw walls
                elif cell_value == 1:
                    # Offsets of neighboring cells (up, down, left, right)
                    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    # Offsets from the cell origin (top left of the cell) to the wall origins
                    # (top left of the rectangles) for drawing wall segments joining each of
                    # the neighboring cells (up, down, left, right)
                    left_top_offsets = [
                        (self.cell_size // 2 - self.wall_width // 2, 0),
                        (self.cell_size // 2 - self.wall_width // 2, self.cell_size // 2 - self.wall_width // 2),
                        (0, self.cell_size // 2 - self.wall_width // 2),
                        (self.cell_size // 2 - self.wall_width // 2, self.cell_size // 2 - self.wall_width // 2)
                    ]
                    # Width and height of wall segments (rectangles) joining each of the
                    # neighboring cells (up, down, left, right)
                    width_height = [
                        (self.wall_width, self.cell_size // 2 + self.wall_width // 2),
                        (self.wall_width, self.cell_size // 2 + self.wall_width // 2),
                        (self.cell_size // 2 + self.wall_width // 2, self.wall_width),
                        (self.cell_size // 2 + self.wall_width // 2, self.wall_width)
                    ]

                    # Loop over each of the 4 neighboring cells
                    for i, (dy, dx) in enumerate(neighbor_offsets):
                        # x and y coordinates of neighboring cell
                        ny, nx = y + dy, x + dx

                        # Ensure the neighboring cell is within bounds
                        if 0 <= ny < h and 0 <= nx < w:

                            # If the neighboring cell has a wall, draw a wall segment (rectangle)
                            # joining the border of the current cell towards the neighboring cell
                            # with the center of the current cell
                            if self.grid[ny, nx] == 1:
                                pygame.draw.rect(
                                    self.screen,
                                    wall_color,
                                    pygame.Rect(
                                        self.grid_offset[2] + x * self.cell_size + left_top_offsets[i][0],
                                        self.grid_offset[0] + y * self.cell_size + left_top_offsets[i][1],
                                        width_height[i][0],
                                        width_height[i][1]
                                    )
                                )

                # Draw the agent
                elif cell_value == 2:
                    # x and y coordinates to draw the agent
                    agent_coords = [
                        self.grid_offset[2] + x * self.cell_size,
                        self.grid_offset[0] + y * self.cell_size
                    ]

                    # If the agent is trying to move to a cell with a wall, shift the coordinates
                    if self.is_electrocuted:
                        match self.last_action:
                            case 0:
                                agent_coords[1] -= self.cell_size // 2 - self.wall_width // 2
                            case 1:
                                agent_coords[1] += self.cell_size // 2 - self.wall_width // 2
                            case 2:
                                agent_coords[0] -= self.cell_size // 2 - self.wall_width // 2
                            case 3:
                                agent_coords[0] += self.cell_size // 2 - self.wall_width // 2

                    self.screen.blit(self.agent_img, tuple(agent_coords))

                # Draw gold, diamond and flag sprites
                else:
                    match cell_value:
                        case 3:
                            img = self.gold_img
                        case 4:
                            img = self.diamond_img
                        case 5:
                            img = self.flag_img

                    self.screen.blit(
                        img,
                        (
                            self.grid_offset[2] + x * self.cell_size,
                            self.grid_offset[0] + y * self.cell_size
                        )
                    )

        pygame.display.flip()
        self.clock.tick(self.metadata.get("render_fps", 5))