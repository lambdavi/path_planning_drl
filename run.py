from env.roverenv_v2g_fin import RoverEnvV2
from env.roverenv_easy import RoverEnvV2 as RoverEnvV2E
from stable_baselines3 import DQN, A2C, PPO
from agents.actor_critic import Agent
from agents.LQAgent import LinearDQN_Agent
import numpy as np
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from argparse import ArgumentParser

# ARGUMENT PARSER
parser = ArgumentParser()
parser.add_argument('--algo', type=str, default='dqn')
parser.add_argument('--obs', type=str, default='image')
parser.add_argument('--sb', action='store_true')
parser.add_argument('--easy', action='store_true')

args = parser.parse_args()

if args.easy:
    print("Easy Env Loaded!")
    env = RoverEnvV2E(obs_type=args.obs)
else:
    env = RoverEnvV2(obs_type=args.obs)
N_GAMES = 1000
load_checkpoint = False
score_history = []
LOG_ON = True

print(args)
if args.sb:
    log_dir = "tmp/"
    env = Monitor(env, log_dir)
    policy = "CnnPolicy" if args.obs == "image" else "MlpPolicy"
    print(policy)
    if args.algo == "dqn":
        model = DQN(env=env, policy=policy, exploration_fraction=0.05, policy_kwargs=dict(normalize_images=False), tensorboard_log=log_dir, verbose=0, buffer_size=10000)
    elif args.algo == "a2c":
        model = A2C(env=env, policy=policy, n_steps=32, policy_kwargs=dict(normalize_images=False), tensorboard_log=log_dir, verbose=0)
    else:
        model = PPO(env=env, policy=policy, policy_kwargs=dict(normalize_images=False), tensorboard_log=log_dir, verbose=0)
        
    # Train the agent
    if LOG_ON:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="rover_projects",
            monitor_gym=True,
            sync_tensorboard=True,
            # Track hyperparameters and run metadata
            config={
                "visited": env.unwrapped.cells_visited,
                "collected": env.unwrapped.targets_collected,
                "epochs": N_GAMES,
            })
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, save_path=log_dir, name_prefix="ddq_"
    )
    model.learn(
        total_timesteps=10000000 if args.obs=="linear" else 100000,
        callback=[
            checkpoint_callback,
            WandbCallback(
                gradient_save_freq=10000,
                model_save_path=f"models/{run.id}",
                model_save_freq=10000,
                log="all",
                verbose=0,
            ),
        ],
        log_interval=4,
        progress_bar=True
    )
else:
    if args.algo == "dqn":
        if args.obs == "linear":
            agent = LinearDQN_Agent(8)
        else:
            pass # TODO: DQN CNN
    else:
        if args.obs == "image":
            agent = Agent(obs_space=3,n_actions=8, obs_type = args.obs)

    for i in range(N_GAMES):
        observation = env.reset()[0]
        score=0
        done = False
        while not done:
            if args.algo == "dqn":
                action = agent.choose_action(observation, env.frame_iteration)
            observation_, reward, done, _, _ = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
            #env.render()
        score_history.append(score)
        avg_rew = np.mean(score_history[-20:])
        print(f"Episode: {i}, Instant_reward: {score}, Avg Reward: {avg_rew}, Visited: {env.cells_visited}, Targets Collected: {env.targets_collected}")
        if LOG_ON:
                wandb.log({"Instant_reward": score, "Avg_reward": avg_rew, "Collected": env.targets_collected, "Visited":env.cells_visited})
        # Render the game

    #env.close()