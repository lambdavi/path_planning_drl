# IMPORTS
import numpy as np
import pygame
from collections import namedtuple
import random

# GLOBAL PARAMS
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE = (0, 0, 255)
BLACK = (0,0,0)
GREEN = (0, 255, 0)
YELLOW = (255,255,0)

BLOCK_SIZE = 20
SPEED = 60

# MAIN CLASS FOR ENVIRONMENT
class RoverEnv():
    def __init__(self, w=640, h=480, n_obstacles=5, n_targets=4, obs_space="linear", render_mode="human") -> None:
        self.obs_space = obs_space
        self.w = w 
        self.h = h

        self.score = 0
        self.targets_collected=0
        self.n_obstacles=n_obstacles
        self.n_targets = n_targets

        if obs_space != "linear":
            self.matrix=None
        
        self.targets=[]
        self.obstacles = []
        self.agent_location = None
        self.bonus=False
        self.max_iter=w*h/(BLOCK_SIZE*BLOCK_SIZE)

        self.action_to_direction = {
            0: Point(BLOCK_SIZE, 0), # right
            1: Point(0, BLOCK_SIZE), # up
            2: Point(-BLOCK_SIZE, 0), # left
            3: Point(0, -BLOCK_SIZE), # down
            4: Point(BLOCK_SIZE, BLOCK_SIZE), # diag upright
            5: Point(-BLOCK_SIZE, BLOCK_SIZE), # diag upleft
            6: Point(-BLOCK_SIZE, -BLOCK_SIZE), #diag downleft
            7: Point(BLOCK_SIZE, -BLOCK_SIZE), # diag downright
        }

        # Pygame initialization
        self.render_mode = render_mode
        pygame.init()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('RoverEnv')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Choose the agent's location uniformly at random
        agent_x = np.random.randint(0, self.w-1)//BLOCK_SIZE*BLOCK_SIZE
        agent_y = np.random.randint(0, self.h-1)//BLOCK_SIZE*BLOCK_SIZE
        self.agent_location = Point(agent_x, agent_y)
        self.score = 0
        self.bonus=False
        self.targets_collected=0
        self.visited=[self.agent_location]
        # TODO: change targets if visited to better learn exploration
        self.obstacles = []
        self.targets = []
        self._place_targets()
        self._place_obstacles()
        # keep track of frame iteration
        self.frame_iteration = 0
        self._update_ui()
        if self.obs_space != "linear":
            self.matrix = np.transpose(pygame.surfarray.array3d(self.display)[::BLOCK_SIZE, ::BLOCK_SIZE], (2,0,1))
    
    def _place_targets(self, n=4):
        while n>0:
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            if Point(x,y)!=self.agent_location:    
                self.targets.append(Point(x, y))
                n-=1

    def _place_obstacles(self, n=20):
        while n>0:
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            if Point(x,y)!=self.agent_location and (Point(x,y) not in self.targets):    
                self.obstacles.append(Point(x, y))
                n-=1
    
    
    def get_obs(self):
        """
        We calculate the (local) state from the game. The state is composed by 11 values.
        """
        # Get points nearby
        point_l = Point(self.agent_location.x - 20, self.agent_location.y)
        point_r = Point(self.agent_location.x + 20, self.agent_location.y)
        point_u = Point(self.agent_location.x, self.agent_location.y - 20)
        point_d = Point(self.agent_location.x, self.agent_location.y + 20)
        point_ul = Point(self.agent_location.x - 20, self.agent_location.y - 20)
        point_ur = Point(self.agent_location.x + 20, self.agent_location.y - 20)
        point_dl = Point(self.agent_location.x - 20, self.agent_location.y + 20)
        point_dr = Point(self.agent_location.x - 20, self.agent_location.y + 20)
        
        neighbors = [
            # Collision ahead
            self.collided(point_u),
            self.collided(point_d),
            self.collided(point_r),
            self.collided(point_l),
            self.collided(point_ul),
            self.collided(point_ur),
            self.collided(point_dl),
            self.collided(point_dr),
            # Visited neighbours
            self.is_visited(point_u),
            self.is_visited(point_d),
            self.is_visited(point_r),
            self.is_visited(point_l),
            self.is_visited(point_ul),
            self.is_visited(point_ur),
            self.is_visited(point_dl),
            self.is_visited(point_dr),
            # Visited neighbours
            self.is_target(point_u),
            self.is_target(point_d),
            self.is_target(point_r),
            self.is_target(point_l),
            self.is_target(point_ul),
            self.is_target(point_ur),
            self.is_target(point_dl),
            self.is_target(point_dr),

    
        ] # Size = 16
        # TODO: find only next target and give informations about how to find it 
        target_location = self._target_location()
        for t in target_location:
            neighbors += t

        return np.array(neighbors, dtype=np.float32)

    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.frame_iteration += 1
        self.reward = 0
        done = False

        direction = self.action_to_direction[action]
        self.agent_location = Point(self.agent_location.x + direction.x, self.agent_location.y + direction.y)
    
        if self.is_target(self.agent_location):
            self.targets_collected += 1
            self.reward = 100/self.frame_iteration
        elif self.collided() or self.frame_iteration >= self.max_iter:
            done = True
            self.reward = -40
            return self.reward, done, self.score, self.targets_collected
        
        if not self.is_visited(self.agent_location):
            self.visited.append(self.agent_location)
            self.score += 1
            self.reward += 0.1
        else:
            self.reward -= 0.05

        if self._finished() and not self.bonus:
            self.reward=100
            self.bonus=True
            print("Good exploration!")
            done = True
        
        self._update_ui()
        if self.obs_space != "linear":
            self.matrix = np.transpose(pygame.surfarray.array3d(self.display)[::BLOCK_SIZE, ::BLOCK_SIZE], (2,0,1))

        self.clock.tick(SPEED)
        return self.reward, done, self.score, self.targets_collected
    
    def _finished(self):
        for pt in self.targets:
            if pt not in self.visited:
                return False
        return True
    
    def _target_location(self):
        ret = []
        # Food location
        for pt in self.targets:
            if pt not in self.visited:
                ret.append([pt.x < self.agent_location.x, # food left 0
                pt.x > self.agent_location.x, # food right 1
                pt.y < self.agent_location.y, # food up 2
                pt.y > self.agent_location.y]) # food down 3)
            else: 
                ret.append([0,0,0,0])
        return ret

    def collided(self, point=None):
        """
            Check for collision. if 'point' is None it just checks if head collided, 
            otherwise we can use it to know if danger is nearby passing a point different from the head.
        """
        if point is None:
            point = self.agent_location

        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.obstacles:
            return True
        
        return False
    
    def is_visited(self, point=None):
        """
            Check for collision. if 'point' is None it just checks if head collided, 
            otherwise we can use it to know if danger is nearby passing a point different from the head.
        """
        if point is None:
            point = self.agent_location
        # hits boundary
        if point in self.visited:
            return True
        return False
    
    def is_target(self, point=None):
        """
            Check for collision. if 'point' is None it just checks if head collided, 
            otherwise we can use it to know if danger is nearby passing a point different from the head.
        """
        if point is None:
            point = self.agent_location
        # hits boundary
        if point in self.targets and point not in self.visited:
            return True
        return False
    
    def _update_ui(self):
        self.display.fill(BLACK)

        # Print targets
        for point in self.targets:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Print visited
        for point in self.visited:
            pygame.draw.rect(self.display, YELLOW, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Print obstacles
        for point in self.obstacles:
            pygame.draw.rect(self.display, RED, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Print agent
        pt = self.agent_location
        pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        
        text = self.font.render("Visited: " + str(self.score), True, WHITE)
        text2 = self.font.render("Target collected: " + str(self.targets_collected), True, WHITE)

        self.display.blit(text, [0, 0])
        self.display.blit(text2, [300, 0])

        pygame.display.flip()
    