#!/usr/bin/env python3
"""
MCP Perceptron Diffusion Model - Run Script
==========================================
This script provides a simple interface to run the MCP Perceptron Diffusion model.
Just run this script and watch as a neural network learns to recreate a famous painting!
"""

import os
import sys
import subprocess
import platform

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header():
    """Print a fancy header."""
    clear_screen()
    print("\n" + "=" * 60)
    print("🎨  MCP PERCEPTRON DIFFUSION MODEL  🎨")
    print("=" * 60)
    print("\nWatch a neural network learn to recreate famous paintings!")
    print("This project demonstrates how neural networks can learn continuous")
    print("representations of images using only coordinates as input.\n")

def check_dependencies():
    """Check if all dependencies are installed."""
    try:
        import torch
        import numpy
        import matplotlib
        import PIL
        import requests
        import skimage
        import psutil
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install all dependencies first:")
        print("pip install -r requirements.txt")
        return False

def show_menu():
    """Show the menu of available paintings."""
    print("\nChoose a painting to recreate:")
    print("1. Mona Lisa (Leonardo da Vinci)")
    print("2. The Starry Night (Vincent van Gogh)")
    print("3. Girl with a Pearl Earring (Johannes Vermeer)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    return choice

def main():
    """Main function."""
    print_header()
    
    if not check_dependencies():
        return
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            subprocess.run([sys.executable, "mona_lisa_mlp.py", "mona_lisa"])
        elif choice == '2':
            subprocess.run([sys.executable, "mona_lisa_mlp.py", "starry_night"])
        elif choice == '3':
            subprocess.run([sys.executable, "mona_lisa_mlp.py", "girl_with_pearl_earring"])
        elif choice == '4':
            print("\nThank you for exploring the MCP Perceptron Diffusion Model!")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()