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

def show_image(x, figsize=(5, 5), **kwargs):
    """Display an image tensor."""
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape) == 3: x = x.permute(1, 2, 0)  # CxHxW to HxWxC
    plt.imshow(x.cpu(), **kwargs)
    plt.axis('off')
    plt.show()


def run_kernel(f, times,*args):
    for i in range(times): f(i, *args)

def rgb2grey_k(i, x, out, n):
    out[i] = 0.2989 * x[i] + 0.5870 * x[i + n] + 0.1140 * x[i + 2*n]

def rgb2grey(x):
    c, h, w = x.shape
    n = h * w
    x = x.flatten()
    res = torch.empty((h * w,), device=x.device)
    run_kernel(rgb2grey_k, h * w, x, res, n)
    return res.view(h, w)

# img_g = rgb2grey(img.cuda())
# show_image(img_g, cmap='gray')


##### Python block kernel version #####

def blk_kernel(f, blocks, threads, *args):
    for i in range(blocks):
        for j in range(threads): f(i, j, threads, *args)

def rgb2grey_blk_k(b, t, threads, x, out, n):
    i = b * threads + t
    if i < n:
        out[i] = 0.2989 * x[i] + 0.5870 * x[i + n] + 0.1140 * x[i + 2*n]

def rgb2grey_blk(x):
    c, h, w = x.shape
    n = h * w
    x = x.flatten()
    res = torch.empty((h * w,), device=x.device)
    threads = 4000
    blocks = (n + threads - 1) // threads
    blk_kernel(rgb2grey_blk_k, blocks, threads, x, res, n)
    return res.view(h, w)

img_g_blk = rgb2grey_blk(img.contiguous().cuda())
show_image(img_g_blk, cmap='gray')