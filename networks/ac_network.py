import os
from torch import nn
from torchvision import transforms
class ActorCriticNetworkCNN(nn.Module):
    def __init__(self, n_actions, obs_space, name='actor_critic', chpt_dir='tmp/actor_critic'):
        super(ActorCriticNetworkCNN, self).__init__()

        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        # Define a simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(obs_space, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.transforms = transforms.Compose([
            transforms.Resize((224,224), antialias=None)
        ])
        # Define fully connected layers for value and policy networks
        self.val_fc = nn.Linear(93312,1)  # Adjust the input size (128 * 5 * 5) according to your observation space
        self.pol_fc = nn.Linear(93312, n_actions)

    def forward(self, x):
        x = x.float()  # Ensure input is a float tensor
        x = self.transforms(x)
        x = self.features(x)
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        v = self.val_fc(x)
        p = nn.functional.softmax(self.pol_fc(x), dim=1)

        return v, p

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions, obs_space, fc1_dims=1024, fc2_dims=512, name='actor_critic', chpt_dir='tmp/actor_critic') -> None:
        super(ActorCriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')
        self.fc1 = nn.Linear(obs_space, self.fc1_dims)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.val_out = nn.Linear(self.fc2_dims, 1)
        self.pol_out = nn.Linear(self.fc2_dims, self.n_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        v = self.val_out(x)
        p = nn.functional.softmax(self.pol_out(x), dim=1)
        return v, p