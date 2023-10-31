from agents.general import GeneralAgent
import random
import torch
from models.DQ import LinearDQN, ImageDQN, ImageDQN_Mobilenet
import os
import numpy as np
class LinearDQN_Agent(GeneralAgent):
    def __init__(self, n_actions=8, lr=0.001, bs=1000, train=True, load_path="", sched=False) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.bs = bs
        self.eps = None
        self.gamma = 0.9
        self.model = LinearDQN(40, 256, 128, n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.HuberLoss()
        self.train = train
        self.scheduler = None
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        if sched:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=0.5, patience=30, verbose=True)
        if not train:
            self.load(load_path)
            self.model.eval()
        print(f"Selected device: {self.device}")
    
        self.model = self.model.to(self.device)
        
    def get_action(self, state, episode):
        # random moves: tradeoff between exploration / exploitation
        self.eps = 80 - episode # random function
        final_move = 0

        if not self.train:
            self.model.eval()

        if random.randint(0, 200) < self.eps and self.train:
            # increasing n_games we don't get moves anymore
            final_move = random.randint(0, self.n_actions-1)
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            prediction = self.model(state0)
            final_move = torch.argmax(prediction).item() 
        
        return final_move
    
    def train_step(self, observation, action, reward, observation_, done):
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long)
        observation_ = torch.tensor(observation_, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(observation.shape) == 1:
            observation = torch.unsqueeze(observation, 0)
            observation_ = torch.unsqueeze(observation_, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        if not self.train:
            self.model.eval()
        # 1. Predicted Q values with current state
        pred = self.model(observation)

        # 2. Q_new = R + gamma * max(next_predicted Q value) -> only if not gameover
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(observation_[idx]))

            target[idx][torch.max(action).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # Q_new and Q
        loss.backward()
        self.optimizer.step()

    def load(self, file_name='model.pt'):
        model_folder_path = "./best_model"
        if not os.path.exists(model_folder_path):
            print("Error in loading")
            exit(1)
        
        file_name = os.path.join(model_folder_path, file_name)
        self.model.load_state_dict(torch.load(file_name))

class ImageDQNAgent(GeneralAgent):
    def __init__(self, obs_shape=(3, 32, 16), n_actions=8, lr=0.001, bs=1000, train=True, load_path="", sched=False) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.bs = bs
        self.eps = None
        self.gamma = 0.9
        self.model = ImageDQN_Mobilenet(n_actions)  # Update input shape to match image dimensions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.HuberLoss()
        self.train = train
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Selected device: {self.device}")
        self.scheduler = None
        if sched:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=0.5, patience=30, verbose=True)
        if not train:
            self.load(load_path)
            self.model.eval()

        self.model = self.model.to(self.device)

    def get_action(self, state, episode):
        self.eps = 80 - episode  # random function
        final_move = 0

        if not self.train:
            self.model.eval()

        if random.randint(0, 200) < self.eps and self.train:
            final_move = random.randint(0, self.n_actions - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            prediction = self.model(state0.unsqueeze(0))  # Adjust for single-image input
            final_move = torch.argmax(prediction).item()

        return final_move
    
    def train_short_memory(self, observation, action, reward, observation_, done):   
        if len(observation.shape)!=3:        
            self.train_step(np.expand_dims(observation,axis=0), action, reward, np.expand_dims(observation_,axis=0), done)
        else:
            self.train_step(observation, action, reward, observation_, done)


    def train_step(self, observation, action, reward, observation_, done):
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        observation_ = torch.tensor(observation_, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        if len(reward.shape) == 0 or len(observation.shape)==3:
            # (1, x) 
            observation = observation.unsqueeze(0)
            observation_ = observation_.unsqueeze(0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            done = np.expand_dims(done, 0)
        if not self.train:
            self.model.eval()

        # 1. Predicted Q values with current state
        pred = self.model(observation)  # Adjust for single-image input

        # 2. Q_new = R + gamma * max(next_predicted Q value) -> only if not gameover
        target = pred.clone()
        
        for idx in range(observation.shape[0]):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(observation_[idx].unsqueeze(0)))  # Adjust for single-image input

            target[0][torch.max(action).item()] = Q_new  # Use [0] to access the first element

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)  # Q_new and Q
        loss.backward()
        self.optimizer.step()

    def load(self, file_name='model_i.pt'):
        model_folder_path = "./best_model"
        if not os.path.exists(model_folder_path):
            print("Error in loading")
            exit(1)

        file_name = os.path.join(model_folder_path, file_name)
        self.model.load_state_dict(torch.load(file_name))