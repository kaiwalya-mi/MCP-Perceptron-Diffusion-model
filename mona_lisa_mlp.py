#!/usr/bin/env python3
"""
MCP Perceptron Diffusion Model
==============================
This script implements a research paper on neural networks that learn to recreate
famous paintings by mapping coordinates to colors through a diffusion-like process.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import psutil

# Check if we're running in a notebook
IN_NOTEBOOK = 'ipykernel' in sys.modules

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_hidden_layers=4, output_dim=1):
        super(MLP, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def download_painting(painting_name):
    """Download a painting if it doesn't exist locally."""
    painting_file = f"{painting_name}.jpg"
    
    if os.path.exists(painting_file):
        print(f"Using existing file: {painting_file}")
        return painting_file
    
    # URLs for different paintings
    urls = {
        "mona_lisa": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
        "starry_night": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "girl_with_pearl_earring": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg"
    }
    
    if painting_name not in urls:
        print(f"Unknown painting: {painting_name}")
        print(f"Available paintings: {', '.join(urls.keys())}")
        sys.exit(1)
    
    print(f"Downloading {painting_name}...")
    response = requests.get(urls[painting_name])
    with open(painting_file, 'wb') as f:
        f.write(response.content)
    
    print(f"Downloaded {painting_file}")
    return painting_file

def prepare_data(image_path, target_size=256):
    """Prepare the training data from an image."""
    # Load and resize the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((target_size, target_size))
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(
        np.linspace(0, 1, target_size),
        np.linspace(0, 1, target_size)
    )
    
    # Stack coordinates
    coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
    
    # Create target values (pixel intensities)
    targets = img_array.flatten().reshape(-1, 1)
    
    # Convert to PyTorch tensors
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return coords_tensor, targets_tensor, img_array

def create_dataloader(coords, targets, batch_size=1024):
    """Create a DataLoader for training."""
    dataset = TensorDataset(coords, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, num_epochs=5000, learning_rate=0.001, 
                original_img=None, painting_name="painting", save_interval=100):
    """Train the MLP model and visualize progress."""
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Move model to device
    model = model.to(device)
    
    # Lists to store metrics
    losses = []
    ssim_values = []
    psnr_values = []
    
    # Create a figure for visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # For creating GIF
    frames_dir = f"frames_{painting_name}"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get image dimensions
    img_size = int(np.sqrt(original_img.shape[0] * original_img.shape[1]))
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for coords, targets in dataloader:
            # Move data to device
            coords, targets = coords.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(coords)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Generate the current image
        with torch.no_grad():
            # Create a grid of coordinates
            y_coords, x_coords = np.meshgrid(
                np.linspace(0, 1, img_size),
                np.linspace(0, 1, img_size)
            )
            coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
            coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
            
            # Generate predictions
            predictions = model(coords_tensor).cpu().numpy().reshape(img_size, img_size)
        
        # Calculate SSIM and PSNR
        current_ssim = ssim(original_img, predictions, data_range=1.0)
        current_psnr = psnr(original_img, predictions, data_range=1.0)
        
        ssim_values.append(current_ssim)
        psnr_values.append(current_psnr)
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, SSIM: {current_ssim:.4f}, PSNR: {current_psnr:.2f}, Time: {elapsed:.2f}s")
        
        # Visualize progress
        if (epoch + 1) % save_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
            # Clear axes
            for ax in axes:
                ax.clear()
            
            # Plot original image
            axes[0].imshow(original_img, cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            # Plot current recreation
            axes[1].imshow(predictions, cmap='gray')
            axes[1].set_title(f"Epoch {epoch+1}")
            axes[1].axis('off')
            
            # Plot metrics
            axes[2].plot(losses, label='Loss')
            axes[2].plot(ssim_values, label='SSIM')
            axes[2].plot(psnr_values, label='PSNR/100')
            axes[2].set_title("Metrics")
            axes[2].legend()
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(f"{frames_dir}/epoch_{epoch+1:04d}.png")
            
            # Save the final comparison
            if epoch == num_epochs - 1:
                plt.savefig(f"output/{painting_name}_final_comparison.png")
            
            # Display in notebook if applicable
            if IN_NOTEBOOK:
                plt.draw()
                plt.pause(0.001)
    
    # Save the trained model
    torch.save(model.state_dict(), f"output/{painting_name}_mlp_model.pth")
    
    # Create a GIF of the training progress
    create_gif(frames_dir, f"output/{painting_name}_training.gif")
    
    return model, losses, ssim_values, psnr_values

def create_gif(frames_dir, output_path, duration=100):
    """Create a GIF from a directory of frames."""
    print(f"Creating GIF at {output_path}...")
    
    # Get all frame files and sort them
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    # Load all frames
    images = [Image.open(f) for f in frames]
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    print(f"GIF created successfully!")
    
    # Try to open the GIF
    try:
        if sys.platform == 'darwin':  # macOS
            os.system(f"open {output_path}")
        elif sys.platform == 'win32':  # Windows
            os.system(f"start {output_path}")
        elif sys.platform == 'linux':  # Linux
            os.system(f"xdg-open {output_path}")
    except:
        print(f"Could not automatically open the GIF. Please find it at: {output_path}")

def main(painting_name="mona_lisa", num_epochs=None):
    """Main function to run the neural painting recreation."""
    # Download or use existing painting
    painting_file = download_painting(painting_name)
    
    # Prepare data
    # Determine target size based on available memory
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    if available_memory < 2:
        target_size = 128
        print(f"Limited memory detected ({available_memory:.1f} GB). Using smaller image size: {target_size}x{target_size}")
    else:
        target_size = 256
        print(f"Sufficient memory detected ({available_memory:.1f} GB). Using image size: {target_size}x{target_size}")
    
    coords, targets, original_img = prepare_data(painting_file, target_size=target_size)
    
    # Create dataloader
    batch_size = min(1024, len(coords))
    dataloader = create_dataloader(coords, targets, batch_size=batch_size)
    
    # Determine number of epochs based on available resources
    if num_epochs is None:
        # Adjust epochs based on CPU count and memory
        cpu_count = psutil.cpu_count(logical=True)
        if cpu_count <= 2 or available_memory < 2:
            num_epochs = 1000
            print(f"Limited resources detected. Training for {num_epochs} epochs.")
        elif cpu_count <= 4 or available_memory < 4:
            num_epochs = 2000
            print(f"Moderate resources detected. Training for {num_epochs} epochs.")
        else:
            num_epochs = 5000
            print(f"Good resources detected. Training for {num_epochs} epochs.")
    
    # Create and train the model
    model = MLP(input_dim=2, hidden_dim=128, num_hidden_layers=4, output_dim=1)
    print(f"Training MLP with {sum(p.numel() for p in model.parameters())} parameters...")
    
    # Determine save interval based on epochs
    save_interval = max(1, num_epochs // 50)
    
    # Train the model
    model, losses, ssim_values, psnr_values = train_model(
        model, dataloader, num_epochs=num_epochs, 
        original_img=original_img.reshape(target_size, target_size),
        painting_name=painting_name,
        save_interval=save_interval
    )
    
    print(f"Training complete! Final metrics:")
    print(f"Loss: {losses[-1]:.6f}")
    print(f"SSIM: {ssim_values[-1]:.4f}")
    print(f"PSNR: {psnr_values[-1]:.2f} dB")
    
    print(f"\nOutput files:")
    print(f"- Model: output/{painting_name}_mlp_model.pth")
    print(f"- Training GIF: output/{painting_name}_training.gif")
    print(f"- Final comparison: output/{painting_name}_final_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Perceptron Diffusion Model")
    parser.add_argument("painting", nargs="?", default="mona_lisa", 
                        help="Painting to recreate (mona_lisa, starry_night, girl_with_pearl_earring)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: auto-determined based on resources)")
    
    args = parser.parse_args()
    main(painting_name=args.painting, num_epochs=args.epochs)