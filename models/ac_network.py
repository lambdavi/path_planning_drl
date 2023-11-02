import os
from torch import nn
from torchvision import transforms
class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions, obs_space, name='actor_critic', chpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()

        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        # Define a simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(obs_space, 8, kernel_size=5, stride=1),
            nn.ReLU(),
        )
        self.transforms = transforms.Compose([
            transforms.Resize((224,224))
        ])
        # Define fully connected layers for value and policy networks
        self.val_fc = nn.Linear(387200,1)  # Adjust the input size (128 * 5 * 5) according to your observation space
        self.pol_fc = nn.Linear(387200, n_actions)

    def forward(self, x):
        x = x.float()  # Ensure input is a float tensor
        x = self.transforms(x)
        x = self.features(x)
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        v = self.val_fc(x)
        p = nn.functional.softmax(self.pol_fc(x), dim=1)

        return v, p
