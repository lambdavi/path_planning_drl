from roverenv_v2g import RoverEnvV2
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


"""
# LATER WE WILL USE STABLEBASELINES
# It will check your custom environment and output additional warnings if needed
model = DQN("CnnPolicy", env, verbose=1, policy_kwargs=dict(normalize_images=False))
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_roverenv")
"""

env = RoverEnvV2()

obs = env.reset()


while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    # Render the game
    env.render()
    
    if done == True:
        break

env.close()