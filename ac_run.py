from roverenv_v2g import RoverEnvV2
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from agents.actor_critic import Agent
env = RoverEnvV2()
agent = Agent(3,n_actions=8)
N_GAMES = 100
load_checkpoint = False
for i in range(N_GAMES):
    observation = env.reset()[0]
    score=0
    done = False
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, _, _ = env.step(action)
        score += reward
        if not load_checkpoint:
            agent.learn(observation, reward, observation_, done)
        observation = observation_
        #env.render()
    print(f"Episode: {i}, Reward: {score}")
    # Render the game

env.close()