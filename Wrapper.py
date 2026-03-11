#write your own data loader, parser, network and loss function for this phase.
from matplotlib import image

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
import matplotlib.pyplot as plt

class NerfDataset(Dataset):
    def __init__(self, data_dir, transform_path, transform=None):
        self.data_dir = data_dir
        self.transform_path = transform_path
        
        with open(self.transform_path, 'r') as f:
            transform_data = json.load(f)
            self.frames = transform_data['frames']
            self.fov = transform_data['camera_angle_x']  # FOV in radians

        self.transform_map = {}
        for frame in self.frames:
  
            basename = os.path.basename(frame['file_path']) 
            self.transform_map[basename] = frame['transform_matrix']

        
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.image_paths = []
        for f in all_files:
            
            basename = os.path.splitext(f)[0]
            if basename in self.transform_map:
                self.image_paths.append(os.path.join(data_dir, f))

    
        sample_image = Image.open(self.image_paths[0])
        self.image_width, self.image_height = sample_image.size
        self.focal_length = self.focal_length_from_fov(self.fov, self.image_width)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract "r_0" from the absolute path
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Instant O(1) lookup! No more for loops!
        transform = self.transform_map[basename]
        
        image = torch.from_numpy(np.array(image) / 255.0).float() 
        transform = torch.tensor(transform, dtype=torch.float32)
        
        return image, transform
    
    
    # Fov present in transform JSON in radians, key is "camera_angle_x"
    def focal_length_from_fov(self,fov, image_width):
        # Calculate focal length from field of view (FOV) and image width
        focal_length = (image_width / 2) / np.tan((fov) / 2)
        return focal_length   


    def image2cam(self,u,v):
        # Convert image to camera coordinates (Intrinsic parameters)
        x_shifted= u - self.image_width/ 2
        y_shifted= v - self.image_height/ 2

        dir_x = x_shifted / self.focal_length
        dir_y = -y_shifted / self.focal_length
        dir_z = torch.full_like(dir_x, -1.0)  # Assuming the camera looks along the negative z-axis
        dir = torch.stack([dir_x, dir_y, dir_z], dim=-1)

        return dir

    def cam2world(self,cam_coords, extrinsic_matrix):
        # Convert camera coordinates to world coordinates (Extrinsic parameters)
        # TODO : Implement the conversion from camera coordinates to world coordinates using the extrinsic matrix
        rotation = extrinsic_matrix[:3, :3]
        translation = extrinsic_matrix[:3, 3]

        
        ray_d = torch.matmul(cam_coords, rotation.T )
        ray_o= translation.expand_as(ray_d)
        return ray_o, ray_d
        
class NeRFModel(nn.Module):
    def __init__(self):
        super(NeRFModel, self).__init__()
        # TODO : Define the architecture of the NeRF model (e.g., MLP with positional encoding)
        self.fc1 = nn.Linear(63, 256)  # Input: ray direction (x, y, z)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(319, 256) # add skip connection here, add pose.
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 257)
        self.fc9 = nn.Linear(283, 256)# no relu , add 3D.
        self.fc10 = nn.Linear(256, 128)
        self.fc11 = nn.Linear(128, 3)    # Output: RGB color


    def positional_encoding(self, x, num_freqs=10):
        # TODO : Implement positional encoding for the input coordinates
        pe = [x]
        for i in range(num_freqs):
            for fn in [torch.sin, torch.cos]:
                pe.append(fn((2.0 ** i) * x))
        return torch.cat(pe, dim=-1)
    
    def forward(self, pts, view_dir):
        rays_63= self.positional_encoding(pts)  # Apply positional encoding to the input pts
        x = F.relu(self.fc1(rays_63))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(torch.cat([rays_63, self.fc4(x)], dim=-1)) # skip connection, adds PE
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        sigma = F.relu(x[..., 0])  # Extract density value (first element of the output)
        view_dir_20 = self.positional_encoding(view_dir, num_freqs=4)  # Apply positional encoding to the input view_dir
        x = self.fc9((torch.cat([ x[..., 1:257], view_dir_20], dim=-1))) # skip connection, adds PE and 3D and density
        x = F.relu(self.fc10(x))
        
        output = torch.sigmoid(self.fc11(x))  # Output color in range [0, 1]
        return output,sigma



def volume_rendering(model, ray_o, ray_d, near, far):
    # Implement volume rendering to compute the final pixel color from the sampled points along the ray
    # TODO : Implement volume rendering using the NeRF model's output (e.g., using alpha compositing)
    t = torch.linspace(near, far, steps=100).unsqueeze(0).unsqueeze(-1)  # Number of samples along each ray
    ray_d_math = ray_d.unsqueeze(1)  # Expand ray_d to match the shape of t
    ray_o_math = ray_o.unsqueeze(1)  # Expand ray_o to match the shape of t
    rays_t_math = ray_o_math + t * ray_d_math  # Sample points along the ray (assuming step size of 0.01)

    ray_d_expanded= ray_d.unsqueeze(1).expand(-1, 100, -1)  # Expand ray_d to match the batch size

    rgb,sigma = model(rays_t_math, ray_d_expanded)  # Get RGB and density from the model

    # Compute alpha values from density
    alpha = 1.0 - torch.exp(-sigma * 0.01)  # Assuming step size of 0.01

    # Compute weights for alpha compositing
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]

    # Compute final pixel color using alpha compositing
    pixel_color = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)

    return pixel_color

def loss_function(predicted_color, true_color):
    # Implement a loss function to compare the predicted pixel color with the true pixel color
    # TODO : Implement a suitable loss function (e.g., mean squared error)
    error = predicted_color - true_color
    mse_loss = torch.mean(error ** 2)
    return mse_loss

def train(model, dataloader,nerfdata, val_dataset, optimizer,epochs,  near, far):

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for images, transforms in dataloader:
            optimizer.zero_grad()
            images= images.squeeze(0) # remove batch dimension
            transforms = transforms.squeeze(0) # remove batch dimension
            num_of_pixels = images.shape[0] * images.shape[1]

            batch_size = 100
            indices = torch.randperm(num_of_pixels)[:batch_size]

            image_flatten= images.view(-1,3)
            colors = image_flatten[indices]

            r, c = torch.unravel_index(indices, images.shape[:2])

            
            ray = nerfdata.image2cam(u=c, v=r)  # Assuming focal length is 1.0 for simplicity
            ray_o, ray_d = nerfdata.cam2world(ray, transforms)
        

            
            
            pixel_color = volume_rendering(model, ray_o, ray_d, near, far)
            loss = loss_function(pixel_color, colors)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")


if __name__ == "__main__":
    # 1. Initialize your dataset and dataloader ONCE
    nerf_dataset = NerfDataset(data_dir='/home/adipat/Documents/Spring_26/CV/p2/nerf/nerf_synthetic/chair/train', transform_path='/home/adipat/Documents/Spring_26/CV/p2/nerf/nerf_synthetic/chair/transforms_train.json')
    val_dataset = NerfDataset(data_dir='/home/adipat/Documents/Spring_26/CV/p2/nerf/nerf_synthetic/chair/val', transform_path='/home/adipat/Documents/Spring_26/CV/p2/nerf/nerf_synthetic/chair/transforms_val.json')
    nerf_dataloader = DataLoader(nerf_dataset, batch_size=1, shuffle=True)

    # 2. Initialize your model and optimizer
    nerf_model = NeRFModel()
    # Adam is standard for NeRF, learning rate usually starts around 5e-4
    optimizer = optim.Adam(nerf_model.parameters(), lr=5e-4) 

    # 3. Start training!
    train(
        model=nerf_model, 
        dataloader=nerf_dataloader, 
        nerfdata=nerf_dataset, # Pass the dataset so you can call image2cam
        val_dataset=val_dataset, # Pass the validation dataset
        optimizer=optimizer, 
        epochs=100, 
        near=2.0,  # NeRF Lego dataset bounds
        far=6.0    # NeRF Lego dataset bounds
    )       