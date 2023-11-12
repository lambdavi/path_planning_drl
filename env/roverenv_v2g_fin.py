import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gymnasium as gym
import random
import math
from gymnasium import Env, spaces
import time
from env.elements import *

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
    
class RoverEnvV2(Env):
    def __init__(self, obs_type="image", print_path=False):
        super(RoverEnvV2, self).__init__()
        self.obs_type = obs_type
        self.max_targets = 4
        self.frame_iteration = 0
        self.drone_path = []  # Initialize an empty list to store the drone's path
        self.print_path = print_path
        self.img_size = (3, 600, 800)

        if obs_type == 'image':
            # Define the observation space for image-based observations
            self.observation_shape = self.img_size
            self.observation_space = spaces.Box(
                low=np.zeros(self.observation_shape, dtype=np.float16),
                high=np.ones(self.observation_shape, dtype=np.float16),
                dtype=np.float32
            )
        elif obs_type == 'linear':
            # Define the observation space for linear observations
            self.observation_shape = (self.max_targets*2+1,)  # Adjust the shape as needed
            self.observation_space = spaces.Box(
                low=np.zeros(self.observation_shape, dtype=np.float16),
                high=np.ones(self.observation_shape, dtype=np.float16),
                dtype=np.float32
            )
        # Define an action space ranging from 0 to 7
        self.action_space = spaces.Discrete(8,)
                        
        # Create a canvas to render the environment images upon 
        self.canvas = np.ones(self.img_size) * 1
        self.cells_visited = 0
        self.targets_collected = 0
        self.elements = []
        self.visited = set()  # Initialize an empty set to store visited cell coordinates

        print(self.canvas.shape)
        self.y_min = int(self.img_size[1] * 0.1)
        self.x_min = int(self.img_size[2] * 0.1)
        self.y_max = int(self.img_size[1] * 0.9)
        self.x_max = int(self.img_size[2] * 0.9)
    
    def draw_elements_on_canvas(self):
        # Init the canvas 
        self.canvas = np.ones(self.img_size) * 1
        if self.print_path:
            self.draw_drone_path()  # Draw the drone's path on the canvas

        # Draw the heliopter on canvas
        for elem in self.elements:
            if isinstance(elem, Aruco):
                if elem.found == 1:
                    continue
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            self.canvas[:, y:y + elem_shape[1], x:x + elem_shape[0]] = elem.icon.transpose((2, 0, 1))

        text = 'Visited: {} | Targets Collected: {}'.format(self.cells_visited, self.targets_collected)

        # Put the info on canvas 
        self.canvas = cv2.putText(self.canvas, text, (10,20), font,  
                0.8, (0,0,0), 1, cv2.LINE_AA)

    def reset(self, seed=None):
        self.cells_visited = 0
        self.targets_collected = 0
        self.visited.clear()  # Clear the visited cell set
        self.drone_path = []
        # Reset the reward
        self.ep_return  = 0
        self.frame_iteration = 0

            # Determine a place to initialize the drone for image observations
        x = random.randrange(int(self.img_size[1] * 0.05), int(self.img_size[1] * 0.9))
        y = random.randrange(int(self.img_size[2] * 0.05), int(self.img_size[2] * 0.9))

        # Intialise the drone
        self.drone = Drone("drone", self.x_max, self.x_min, self.y_max, self.y_min)
        self.drone.set_position(x,y)

        # Intialise the elements 
        self.elements = [self.drone]

        self._place_walls()
        self._place_targets()

        # Reset the Canvas 
        self.canvas = np.ones(self.img_size) * 1

        # Return observations based on the selected "obs_type"
        self.draw_elements_on_canvas()
        if self.obs_type == 'image':
            return self.canvas, {}
        elif self.obs_type == 'linear':
            return self.calculate_linear_observations(), {}
        
    def calculate_linear_observations(self):
        drone_x, drone_y = self.drone.get_position()
        observations = []
        remaining = 0
        for elem in self.elements:
            if isinstance(elem, Aruco):
                if elem.found == 0:
                    target_x, target_y = elem.get_position()
                    angle = math.atan2(target_y - drone_y, target_x - drone_x)
                    distance = np.sqrt((target_x - drone_x) ** 2 + (target_y - drone_y) ** 2)
                    observations.append(angle)
                    observations.append(distance)
                    remaining+=1
                else:
                    observations.append(0)
                    observations.append(0)        

        return np.array(observations+[remaining], dtype=np.float32)
    
    def _place_walls(self, n=5):
        for i in range(n):
            valid_position = False
            while not valid_position:
                spawned_wall = Wall("wall_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)
                x = random.randrange(self.x_min, self.x_max - spawned_wall.icon_w)
                y = random.randrange(self.y_min, self.y_max - spawned_wall.icon_h)
                spawned_wall.set_position(x, y)
                # Check for overlap with other elements
                overlap = any(self.has_collided(spawned_wall, elem) for elem in self.elements)
                if not overlap:
                    self.elements.append(spawned_wall)
                    valid_position = True
    
    def _place_targets(self, n=4):
        for i in range(n):
            valid_position = False
            while not valid_position:
                spawned_target = Aruco("target_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min)
                x = random.randrange(self.x_min, self.x_max - spawned_target.icon_w)
                y = random.randrange(self.y_min, self.y_max - spawned_target.icon_h)
                spawned_target.set_position(x, y)
                # Check for overlap with other elements
                overlap = any(self.has_collided(spawned_target, elem) for elem in self.elements)
                if not overlap:
                    self.elements.append(spawned_target)
                    valid_position = True

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas.transpose((1,2,0)))
            cv2.waitKey(10)
        
        elif mode == "rgb_array":
            return self.canvas
    
    def close(self):
        cv2.destroyAllWindows()
    
    def get_action_meanings(self):
        return {0: "Right", 1: "Up", 2: "Left", 3: "Down", 4: "Diag_UR", 5: "Diag_UL", 6: "Diag_DL", 7: "Diag_DR"}
    
    def step(self, action):
        # Flag that marks the termination of an episode
        done = False
        self.frame_iteration += 1
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        reward = 0

        step_size = 10
        # apply the action to the drone
        if action == 0:
            self.drone.move(0,step_size)
        elif action == 1:
            self.drone.move(-step_size,0)
        elif action == 2:
            self.drone.move(0,-step_size)
        elif action == 3:
            self.drone.move(step_size,0)
        elif action == 4:
            self.drone.move(-step_size,step_size)
        elif action == 5:
            self.drone.move(-step_size,-step_size)
        elif action == 6:
            self.drone.move(step_size,-step_size)
        elif action == 7:
            self.drone.move(step_size,step_size)


        # Calculate the drone's current cell coordinates
        current_cell = (self.drone.x // self.drone.icon_w, self.drone.y // self.drone.icon_h)
        if self.print_path:
            self.drone_path.append(current_cell)  # Append the current cell to the drone's path

        if current_cell not in self.visited:
            # The drone has visited a new cell
            self.visited.add(current_cell)
            reward += 0.01  # Assign a reward for visiting a new cell
            self.cells_visited+=1

        # For elements in the Ev
        for elem in self.elements:
            if isinstance(elem, Wall):
                # If the drone has collided.
                if self.has_collided(self.drone, elem):
                    # Conclude the episode and remove the drone from the Env.
                    done = True
                    reward -= 2
                    self.elements.remove(self.drone)
                    break
            elif isinstance(elem, Aruco):
                # If the fuel tank has collided with the drone.
                if self.has_collided(self.drone, elem) and elem.found == 0:
                    # Remove the fuel tank from the env.
                    elem.found=1
                    self.targets_collected +=1
                    reward += 1
                else:
                    target_x, target_y = elem.get_position()
                    distance = np.sqrt((target_x - self.drone.x) ** 2 + (target_y - self.drone.y) ** 2)
                    # Define a proximity reward function (you can adjust the scaling factor)
                    reward += 0.1 / distance  # Adjust the scaling factor as needed
        
        if self.frame_iteration > 1000:
            done = True
            reward -= 0.5
        
        # Increment the episodic return
        self.ep_return += 1

        # Update observations based on the selected "obs_type"
        self.draw_elements_on_canvas()

        if self.obs_type == 'image':
            observations = self.canvas
        elif self.obs_type == 'linear':
            observations = self.calculate_linear_observations()

        return observations, reward, done, False, {}
    
    def draw_drone_path(self):
        for cell in self.drone_path:
            x, y = cell
            x_pos = x * self.drone.icon_w
            y_pos = y * self.drone.icon_h
            self.canvas[:, y_pos:y_pos + self.drone.icon_h, x_pos:x_pos + self.drone.icon_w] = 0.5  # Change the color to draw the path

    def has_collided(self, elem1, elem2):
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True

        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True

        if x_col and y_col:
            return True

        return False
                