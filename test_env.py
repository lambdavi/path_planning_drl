from env.roverenv_easy import RoverEnvV2
from env.roverenv_st import RoverEnvST
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnvWrapper
import imageio
import numpy as np
from time import sleep

from stable_baselines3.common.monitor import Monitor

# LATER WE WILL USE STABLEBASELINES
# It will check your custom environment and output additional warnings if needed

env = Monitor(RoverEnvST(obs_type='linear'))
#env = RoverEnvV2(obs_type='linear')
model = PPO(env=env, 
            policy="MlpPolicy", 
            n_steps=1024, 
            n_epochs=4, 
            gamma = 0.999,
            gae_lambda = 0.98,
            ent_coef = 0.01,
            policy_kwargs=dict(normalize_images=False), 
            verbose=0)
#model = DQN(policy="MlpPolicy", env=env, verbose=0)

#model.learn(total_timesteps=100000, log_interval=4)
#model.save("dqn_roverenv")
model = PPO.load("models/best/new_ppo", env)
mean_reward, std_reward = evaluate_policy(model, env=env, n_eval_episodes=10, deterministic=True)
print(mean_reward, std_reward)
# Enjoy trained agent
obs = env.reset()[0]
for i in range(1000):
    #print(i)
    action, _states = model.predict(obs)
    #print("Action", env.get_action_meanings()[action.item()])
    obs, rewards, dones, _, info = env.step(action)
    #print(f"OBS: {obs}, Reward: {rewards}, Dones: {dones}")
    if dones:
        obs = env.reset()[0]
    env.render()
exit(1)
"""images = []
obs = model.env.reset()
img = model.env.render()
for i in range(350):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    img = model.env.render()

imageio.mimsave("lander_a2c.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
"""
obs = vec_env.reset()
print(obs[0].shape)
while True:
    action, _states = model.predict(obs[0], deterministic=True)
    obs, rewards, dones, info, _ = vec_env.step(action)
    obs = [obs]
    vec_env.render()
    print(dones)
    sleep(0.1)
    if dones:
        obs = vec_env.reset()


"""while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    # Render the game
    env.render()
    
    if done == True:
        break

env.close()"""