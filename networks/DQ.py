from torch import nn, save, zeros
from torch.nn import functional as F
import os
from torchvision import models, transforms
class LinearDQN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size2)        
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pt'):
        model_folder_path = "./best_model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        save(self.state_dict(), file_name)

class ImageDQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ImageDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 28 * 20, 128)  # Adapt the input size based on your input shape
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name='model.pt'):
        model_folder_path = "./best_model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        save(self.state_dict(), file_name)
    

class ImageDQN_Mobilenet(nn.Module):
    def __init__(self, output_size):
        super(ImageDQN_Mobilenet, self).__init__()
        # Load a pretrained MobileNetV2 model
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Modify the last classification layer to match the output_size
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, output_size)

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224), antialias=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        # Pass the input through the MobileNet model
        x = self.transforms(x)
        x = self.mobilenet(x)
        return x
    

    def save(self, file_name='model_m.pt'):
        model_folder_path = "./best_model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        save(self.state_dict(), file_name)

class ImageDQN_RESNET(nn.Module):
    def __init__(self, output_size):
        super(ImageDQN_RESNET, self).__init__()
        # Load a pretrained MobileNetV2 model
        self.res18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the last classification layer to match the output_size
        num_features = self.res18.fc.in_features
        self.res18.fc = nn.Linear(num_features, output_size)

        self.transforms = transforms.Compose([  # Convert to PIL Image (if not already)
            transforms.Resize((224, 224)),  # Resize to a specific size
            transforms.ToTensor(),  # Convert to a PyTorch tensor
        ])

    def forward(self, x):
        # Pass the input through the MobileNet model
        x = self.transforms(x)
        x = self.res18(x)
        return x
    

    def save(self, file_name='model_r.pt'):
        model_folder_path = "./best_model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        save(self.state_dict(), file_name)