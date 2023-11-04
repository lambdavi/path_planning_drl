from roverenv_v2g_fin import RoverEnvV2
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnvWrapper

"""
# LATER WE WILL USE STABLEBASELINES
# It will check your custom environment and output additional warnings if needed
model = DQN("CnnPolicy", env, verbose=1, policy_kwargs=dict(normalize_images=False))
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_roverenv")
"""

vec_env = RoverEnvV2(obs_type='linear')

model = A2C.load("models/e5ts0197/model")

obs = vec_env.reset()[0]
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info, _ = vec_env.step(action)
    vec_env.render()
    if dones:
        obs = vec_env.reset()[0]


"""while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    # Render the game
    env.render()
    
    if done == True:
        break

env.close()"""