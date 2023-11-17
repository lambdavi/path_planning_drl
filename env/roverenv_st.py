import numpy as np 
import cv2 
import random
import math
from gymnasium import Env, spaces
from env.elements import *
import matplotlib.pyplot as plt
from time import sleep
font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
    
class RoverEnvST(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, obs_type="image", print_path=False):
        super(RoverEnvST, self).__init__()
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
                dtype=np.float16
            )
        elif obs_type == 'linear':
            # Define the observation space for linear observations
            self.observation_shape = (43,)  # Adjust the shape as needed
            self.observation_space = spaces.Box(
                low=np.zeros(self.observation_shape, dtype=np.float16),
                high=np.ones(self.observation_shape, dtype=np.float16),
                dtype=np.float16
            )
        # Define an action space ranging from 0 to 7
        self.action_space = spaces.Discrete(8,)
                        
        # Create a canvas to render the environment images upon 
        self.canvas = np.ones(self.img_size) * 1
        self.cells_visited = 0
        self.targets_collected = 0
        self.elements = []
        self.visited = set()  # Initialize an empty set to store visited cell coordinates
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

        self.text = 'Visited: {} | Targets Collected: {}'.format(self.cells_visited, self.targets_collected)
        # Put the info on canvas 
        self.canvas = cv2.putText(self.canvas, self.text, (10,20), font,  0.8, (255,255,255), 2, cv2.LINE_AA)

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
        #print(f"Drone spawned in {x},{y}")
        # Intialise the elements 
        self.elements = [self.drone]
        
        self._place_walls()
        self._place_targets(n=self.max_targets)

        # Reset the Canvas 
        self.canvas = np.ones(self.img_size) * 1
        plt.close()

        # Return observations based on the selected "obs_type"
        self.draw_elements_on_canvas()
        if self.obs_type == 'image':
            return self.canvas, {}
        elif self.obs_type == 'linear':
            return self.calculate_linear_observations(), {}
        
    def calculate_linear_observations(self):
        drone_x, drone_y = self.drone.get_position()
        observations = []

        max_distance = max(self.img_size[1], self.img_size[2])  # Adjust this based on your environment size
        closest_elem = []
        local_max_dist = 99999999
        for elem in self.elements:
            if isinstance(elem, Aruco) and elem.found == 0:
                target_x, target_y = elem.get_position()
                distance = np.sqrt((target_x - drone_x) ** 2 + (target_y - drone_y) ** 2)
                if distance < local_max_dist:
                    local_max_dist = distance
                    direction = math.degrees(math.atan2(target_y - drone_y, target_x - drone_x))
                    closest_elem = [[np.cos(np.radians(direction)), np.sin(np.radians(direction))], distance/max_distance]
        # Encode direction into observations
        observations.extend(closest_elem[0])

        # Normalize and encode distance into observations
        observations.append(closest_elem[1])
        step_size=5
        for elem in self.elements:
            if isinstance(elem, Wall):
                target_x, target_y = elem.get_position()
                direction = math.atan2(target_y - drone_y, target_x - drone_x)
                # OBS for each wall will be [0/1]*n_walls where each cell will be 0 if going in that direction would result in an impact
                
                if self.has_collided_points([target_x, target_y], [drone_x, drone_y+step_size]):
                    observations.append(1)
                else:
                    observations.append(0)
                if self.has_collided_points([target_x, target_y], [drone_x-step_size, drone_y]):
                    observations.append(1)
                else:
                    observations.append(0)
                if self.has_collided_points([target_x, target_y], [drone_x, drone_y-step_size]):
                    observations.append(1)
                else:
                    observations.append(0)
                if self.has_collided_points([target_x, target_y], [drone_x+step_size, drone_y]):
                    observations.append(1)
                else:
                    observations.append(0)
                if self.has_collided_points([target_x, target_y], [drone_x-step_size, drone_y+step_size]):
                    observations.append(1)
                else:
                    observations.append(0)
                if self.has_collided_points([target_x, target_y], [drone_x-step_size, drone_y-step_size]):
                    observations.append(1)
                else:
                    observations.append(0)
                if self.has_collided_points([target_x, target_y], [drone_x+step_size, drone_y-step_size]):
                    observations.append(1)
                else:
                    observations.append(0)
                if self.has_collided_points([target_x, target_y], [drone_x+step_size, drone_y+step_size]):
                    observations.append(1)
                else:
                    observations.append(0)

        #print(len(observations))
        return np.array(observations, dtype=np.float32)
    
    """    def calculate_linear_observations(self):
        drone_x, drone_y = self.drone.get_position()
        observations = []

        max_distance = max(self.img_size[1], self.img_size[2])  # Adjust this based on your environment size
        closest_elem = []
        local_max_dist = 99999999
        for elem in self.elements:
            if isinstance(elem, Aruco) and elem.found == 0:
                target_x, target_y = elem.get_position()
                distance = np.sqrt((target_x - drone_x) ** 2 + (target_y - drone_y) ** 2)
                if distance < local_max_dist:
                    local_max_dist = distance
                    direction = math.degrees(math.atan2(target_y - drone_y, target_x - drone_x))
                    closest_elem = [[np.cos(np.radians(direction)), np.sin(np.radians(direction))], distance/max_distance]
        # Encode direction into observations
        observations.extend(closest_elem[0])

        # Normalize and encode distance into observations
        observations.append(closest_elem[1])

        for elem in self.elements:
            if isinstance(elem, Wall):
                target_x, target_y = elem.get_position()
                direction = math.degrees(math.atan2(target_y - drone_y, target_x - drone_x))
                distance = np.sqrt((target_x - drone_x) ** 2 + (target_y - drone_y) ** 2)

                # Encode direction into observations
                observations.extend([np.cos(np.radians(direction)), np.sin(np.radians(direction))])

                # Normalize and encode distance into observations
                observations.append(distance / max_distance)
        #print(len(observations))
        return np.array(observations, dtype=np.float32)"""



            
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
            #cv2.startWindowThread()
            cv2.imshow("Game", self.canvas.transpose((1,2,0)))
            """plt.ion()
            plt.xlabel(self.text)
            plt.imshow(self.canvas.transpose((1,2,0)))
            plt.pause(0.05)
            plt.clf()"""
            cv2.waitKey(1)
            #cv2.destroyAllWindows()
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

        step_size = 5

        # Store current drone position before taking action
        prev_drone_x, prev_drone_y = self.drone.get_position()

        # Apply the action to the drone
        if action == 0:
            self.drone.move(0, step_size)
        elif action == 1:
            self.drone.move(-step_size, 0)
        elif action == 2:
            self.drone.move(0, -step_size)
        elif action == 3:
            self.drone.move(step_size, 0)
        elif action == 4:
            self.drone.move(-step_size, step_size)
        elif action == 5:
            self.drone.move(-step_size, -step_size)
        elif action == 6:
            self.drone.move(step_size, -step_size)
        elif action == 7:
            self.drone.move(step_size, step_size)
        else:
            print("ERROR")

        # Calculate the drone's current cell coordinates after taking action
        current_cell = (self.drone.x // self.drone.icon_w, self.drone.y // self.drone.icon_h)

        if current_cell not in self.visited:
            # The drone has visited a new cell
            self.visited.add(current_cell)
            reward += 0.01  # Assign a reward for visiting a new cell
            self.cells_visited += 1
        else:
            reward -= 0.005

        # Check for collisions with walls
        for elem in self.elements:
            if isinstance(elem, Wall) and self.has_collided(self.drone, elem):
                done = True
                reward = -2  # Negative reward for crash
                self.elements.remove(self.drone)
                break

        # Check for collisions with Aruco targets
        target_x = 0
        target_y=0
        local_closest_elem = None
        local_best_dist = 999999
        for elem in self.elements:
            if isinstance(elem, Aruco):
                target_x, target_y = elem.get_position()
                curr_dist = np.sqrt((self.drone.x - target_x) ** 2 + (self.drone.y - target_y) ** 2)
                if curr_dist < local_best_dist:
                    local_closest_elem = [target_x, target_y]
                if self.has_collided(self.drone, elem) and elem.found == 0:
                    elem.found = 1
                    self.targets_collected += 1
                    reward += 2  # Reward for target collection
                    self._place_targets(1)  # Generate new target
        
        if not done:        
            # Calculate the distance moved by the drone
            prev_dist = np.sqrt((prev_drone_x - local_closest_elem[0]) ** 2 + (prev_drone_y - local_closest_elem[1]) ** 2)
            curr_dist = np.sqrt((self.drone.x - local_closest_elem[0]) ** 2 + (self.drone.y - local_closest_elem[1]) ** 2)
            if curr_dist < prev_dist:
                # Reward for getting closer to the target
                reward += 0.1 / (curr_dist if curr_dist != 0 else 1)

        # Negative reward for reaching step limit
        if self.frame_iteration > 1500:
            done = True
            reward -= 0.5

        # Reward to incentivize exploration of new cells
        reward += 0.001

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

    def has_collided_points(self, coord1, coord2):
        x_col = False
        y_col = False

        if 2 * abs(coord1[0] - coord2[0]) <= (32 + 32):
            x_col = True

        if 2 * abs(coord1[1] - coord2[1]) <= (32 + 32):
            y_col = True

        if x_col and y_col:
            return True

        return False
                