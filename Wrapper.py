#write your own data loader, parser, network and loss function for this phase.

from email.mime import image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class NerfDataset(Dataset):
    def __init__(self, data_dir, transform_path, transform=None):
        self.data_dir = data_dir
        self.transform_path = transform_path
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        with open(self.transform_path, 'r') as f:
            transform_data = json.load(f)
            self.frames = transform_data['frames']
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # grab the corresponding transform for the image
        # Search for same imagepath in frames dictionary 
        for frame in self.frames:
            if frame['file_path'] == img_path:
                transform = frame['transform_matrix']
                break


        
        
        
        return image, transform
class NeRFModel(nn.Module):
    def __init__(self):
        super(NeRFModel, self).__init__()
        # TODO : Define the architecture of the NeRF model (e.g., MLP with positional encoding)
        self.fc1 = nn.Linear(3, 256)  # Input: ray direction (x, y, z)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3)    # Output: RGB color

    def forward(self, rays):
        # TODO : Implement the forward pass of the NeRF model
        x = F.relu(self.fc1(rays))
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))  # Output color in range [0, 1]
        return output

# Fov present in transform JSON in radians, key is "camera_angle_x"
def focal_length_from_fov(fov, image_width):
    # Calculate focal length from field of view (FOV) and image width
    focal_length = (image_width / 2) / np.tan((fov) / 2)
    return focal_length   


def image2cam(u,v, focal_length):
    # Convert image to camera coordinates (Intrinsic parameters)
    x_shifted= u - image_width/ 2
    y_shifted= v - image_height/ 2

    dir_x = x_shifted / focal_length
    dir_y = -y_shifted / focal_length
    dir_z = -1.0 
    dir = torch.tensor([dir_x, dir_y, dir_z])

    return dir

def cam2world(cam_coords, extrinsic_matrix):
    # Convert camera coordinates to world coordinates (Extrinsic parameters)
    # TODO : Implement the conversion from camera coordinates to world coordinates using the extrinsic matrix
    rotation = extrinsic_matrix[:3, :3]
    translation = extrinsic_matrix[:3, 3]

    ray_o= translation
    ray_d = torch.matmul(rotation, cam_coords)
    return ray_o, ray_d


def generate_batch(image, transform):
    # Generate a batch of rays and corresponding pixel colors
    # TODO : Implement the generation of rays and pixel colors from the image and transform
    rays = []
    colors = []

    # Random sample batch of pixels from the image
    batch_size = 100
    indices = torch.randperm(image.numel())[:batch_size]
    rows, cols = torch.unravel_index(indices, image.shape)

    # sample rays for each pixel and get corresponding colors
    for r, c in zip(rows, cols):
        ray = image2cam(image[r, c], focal_length=1.0)  # Assuming focal length is 1.0 for simplicity
        ray_o, ray_d = cam2world(ray, transform)
        rays.append((ray_o, ray_d))
        colors.append(image[r, c])
    return torch.stack([r[0] for r in rays]), torch.stack([r[1] for r in rays]), torch.stack(colors)

def train(model, dataloader, optimizer, criterion):
    model.train()
    for images, transforms in dataloader:
        optimizer.zero_grad()
        rays, colors = generate_batch(images, transforms)

        t_samples = 64  # Number of samples along each ray

        for t in range(t_samples):
            rays_t = rays + t * 0.01  # Sample points along the ray (assuming step size of 0.01)

        outputs = model(rays)
        loss = criterion(outputs, colors)
        loss.backward()
        optimizer.step()

