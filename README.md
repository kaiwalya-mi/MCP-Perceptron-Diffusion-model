# 🎨 MCP Perceptron Diffusion Model(ML Paper implementation)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Research](https://img.shields.io/badge/Research-Implementation-brightgreen.svg)

**ML Research Paper Implementation: Watch a neural network learn to recreate famous paintings in real-time!**




<p align=center>

https://github.com/user-attachments/assets/93abbf52-e092-47a5-8fbd-0e3147f6ae4d
</p>


<img width="563" alt="Screenshot 2025-03-16 at 9 54 58 PM" src="https://github.com/user-attachments/assets/32232b49-1831-4101-be04-805fb891fb0a" />



## 📑 Research Overview

This project implements a Multi-layer Perceptron (MLP) based diffusion model that learns continuous representations of images. The model maps 2D coordinates to color values, effectively recreating famous paintings pixel by pixel through a diffusion-like process.

Unlike traditional image processing techniques that work with discrete pixel data, our approach learns a continuous function that can generate any point in the image from its coordinates. This creates a fascinating visualization of how neural networks learn to approximate complex patterns over time.

The system shows real-time training progress, allowing you to witness the neural network's journey from random noise to a recognizable masterpiece. This approach is similar to techniques used in advanced computer graphics and neural radiance fields (NeRF).

## 🚀 Quick Start (Just 2 Steps!)

### Using pip:
```bash
# Step 1: Install requirements
pip install -r requirements.txt

# Step 2: Run the interactive menu
python run.py
```

### Using uv (recommended for reproducibility):
```bash
# Step 1: Install uv if you don't have it
curl -fsSL https://astral.sh/uv/install.sh | sh

# Step 2: Install dependencies from lock file
uv pip install -r requirements.lock

# Step 3: Run the interactive menu
python run.py
```

That's it! The program will automatically:
- Check your system capabilities
- Download the painting
- Train the neural network
- Show real-time visualization
- Open the final GIF when done

## 💡 What's Happening?

This project demonstrates how neural networks can learn to represent images in a fascinating way:

1. **Coordinates → Colors**: The neural network learns to map (x,y) coordinates to colors
2. **No Image Data**: Unlike traditional image processing, the network only sees coordinates
3. **Continuous Function**: The network learns a continuous function that represents the painting
4. **Live Learning**: You can watch the network improve in real-time as it trains

## 🧠 How It Works (Simple Explanation)

Imagine you're trying to recreate a painting, but instead of looking at the whole image, you can only look at one pixel at a time:

1. You look at a position (x,y) on the canvas
2. You try to guess what color should be at that position
3. You check if your guess was right
4. You adjust your guessing strategy to do better next time
5. Repeat thousands of times until you can accurately predict any pixel's color

That's exactly what our neural network (called a Multilayer Perceptron or MLP) is doing!

## 🖼️ Available Paintings

The project includes three famous masterpieces:

- **Mona Lisa** by Leonardo da Vinci
- **The Starry Night** by Vincent van Gogh
- **Girl with a Pearl Earring** by Johannes Vermeer

## 📊 Project Structure

```
neural-painting-recreation/
├── mona_lisa_mlp.py    # Main code for neural network training
├── run.py              # Interactive menu to run the project
├── requirements.txt    # Dependencies
├── requirements.lock   # Locked dependencies (for uv)
├── output/             # Generated images and models
│   ├── *_training.gif  # Training progression GIFs
│   ├── *_final_comparison.png # Final results
│   └── *_mlp_model.pth # Trained models
└── README.md           # This file
```

## 🔍 Technical Details (For Those Interested)

- **Architecture**: 4-layer MLP with 128 neurons per layer
- **Input**: 2 neurons (x,y coordinates)
- **Output**: 1 neuron (grayscale pixel value)
- **Training**: Optimized for CPU, automatically adapts to your system
- **Metrics**: Loss, SSIM (structural similarity), and PSNR (peak signal-to-noise ratio)

## 📝 Why This Matters

This project demonstrates several important concepts in machine learning:

1. **Function Approximation**: Neural networks can learn complex functions
2. **Coordinate-Based Networks**: A new paradigm in image representation
3. **Implicit Neural Representations**: Representing data continuously rather than discretely
4. **Visualization**: The importance of seeing the learning process

These concepts are used in cutting-edge research areas like NeRF (Neural Radiance Fields) for 3D scene reconstruction and neural implicit representations for graphics.

## 🔧 Troubleshooting

If you encounter any issues:

- **Dependency errors**: Try using uv with the lock file for exact versions
- **Memory issues**: The code will automatically adapt to your system
- **Display issues**: Make sure you have a graphical environment for visualization
- **Performance**: Training is CPU-intensive; expect 5-10 minutes per painting

## 📚 Research References

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- [Coordinate-Based Neural Representations for Graphics and Vision](https://www.cs.cmu.edu/~aayushb/SIREN/)

---

Created with ❤️ using PyTorch
