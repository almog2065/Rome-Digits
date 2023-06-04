import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from PIL import Image
import random
from scipy.ndimage import zoom

if __name__ == '__main__':
    letters = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    # divide
    div_directory = 'data/val/'
    for letter in letters:
        directory = 'data/train/' + letter
        items = os.listdir(directory)
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=int(0.2 * len(items)))
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # # Plot the image
                # image = plt.imread(file_path)
                #
                # image_pil = Image.fromarray((image * 255).astype(np.uint8))
                #
                # # Save the image as a PNG file
                # image_pil.save(os.path.join(div_directory + letter, filename))

                os.rename(file_path, os.path.join(div_directory + letter, filename))
