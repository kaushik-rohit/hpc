import torch
from torchvision import io
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def is_cuda_available():
    """Check if CUDA is available on the system."""
    return torch.cuda.is_available()

print("Is CUDA available?", is_cuda_available())

path_img = Path("./puppy.jpg")

if not path_img.exists():
    raise FileNotFoundError(f"The image file {path_img} does not exist.")

img = io.read_image("./puppy.jpg")

print(img.shape)

def show_image(x, figsize=(5, 5)):
    """Display an image tensor."""
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape) == 3: x = x.permute(1, 2, 0)  # CxHxW to HxWxC
    plt.imshow(x.cpu())
    plt.axis('off')
    plt.show()

show_image(img)