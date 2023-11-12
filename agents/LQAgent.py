from agents.general import GeneralAgent
import random
import torch
from networks.DQ import LinearDQN
import os
class LinearDQN_Agent(GeneralAgent):
    def __init__(self, n_actions=8, lr=0.001, bs=1000, train=True, load_path="", sched=False) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.bs = bs
        self.eps = None
        self.gamma = 0.9
        self.model = LinearDQN(9, 256, 128, n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.HuberLoss()
        self.train = train
        self.scheduler = None
        if sched:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=0.5, patience=30, verbose=True)
        if not train:
            self.load(load_path)
            self.model.eval()
        
    def choose_action(self, state, episode):
        # random moves: tradeoff between exploration / exploitation
        self.eps = 80 - episode # random function
        final_move = 0

        if not self.train:
            self.model.eval()

        if random.randint(0, 200) < self.eps and self.train:
            # increasing n_games we don't get moves anymore
            final_move = random.randint(0, self.n_actions-1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            final_move = torch.argmax(prediction).item() 
        
        return final_move
    
    def train_step(self, observation, action, reward, observation_, done):
        observation = torch.tensor(observation, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        observation_ = torch.tensor(observation_, dtype=torch.float)
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