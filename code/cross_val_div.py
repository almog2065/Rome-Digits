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
from sklearn.model_selection import KFold


def noisy_aug(image):
    number_of_noisy_points = int(random.uniform(0, 0.002) * image.shape[0] * image.shape[1])
    for i in range(number_of_noisy_points):
        pixel_0 = random.randint(9, image.shape[0] - 9)
        pixel_1 = random.randint(9, image.shape[1] - 9)
        for j in range(9):
            image[pixel_0][pixel_1] = random.uniform(0, 0.5)
            image[pixel_0][pixel_1 + 1] = random.uniform(0, 0.5)
            image[pixel_0][pixel_1 - 1] = random.uniform(0, 0.5)
            image[pixel_0 + 1][pixel_1] = random.uniform(0, 0.5)
            image[pixel_0 + 1][pixel_1 + 1] = random.uniform(0, 0.5)
            image[pixel_0 + 1][pixel_1 - 1] = random.uniform(0, 0.5)
            image[pixel_0 - 1][pixel_1] = random.uniform(0, 0.5)
            image[pixel_0 - 1][pixel_1 - 1] = random.uniform(0, 0.5)
            image[pixel_0 - 1][pixel_1 + 1] = random.uniform(0, 0.5)
    return image


def lines_aug(image):
    number_of_ver_lines = int(random.randint(0, 3))
    if number_of_ver_lines == 0:
        number_of_hor_lines = int(random.randint(1, 3))
    else:
        number_of_hor_lines = int(random.randint(0, 3))

    for i in range(number_of_ver_lines):
        number_of_pixels_line = int(random.randint(1, int(image.shape[1])))
        start_pixel_0 = random.randint(9, image.shape[0] - 9)
        start_pixel_1 = random.randint(9, image.shape[1] - 9)
        for j in range(number_of_pixels_line):
            # skip
            if random.randint(0, 2) == 1:
                continue
            pixel_0 = start_pixel_0
            pixel_1 = start_pixel_1 + 3 * j
            if pixel_1 >= image.shape[1] - 9:
                continue
            for j in range(9):
                image[pixel_0][pixel_1] = random.uniform(0, 0.5)
                image[pixel_0][pixel_1 + 1] = random.uniform(0, 0.5)
                image[pixel_0][pixel_1 - 1] = random.uniform(0, 0.5)
                image[pixel_0 + 1][pixel_1] = random.uniform(0, 0.5)
                image[pixel_0 + 1][pixel_1 + 1] = random.uniform(0, 0.5)
                image[pixel_0 + 1][pixel_1 - 1] = random.uniform(0, 0.5)
                image[pixel_0 - 1][pixel_1] = random.uniform(0, 0.5)
                image[pixel_0 - 1][pixel_1 - 1] = random.uniform(0, 0.5)
                image[pixel_0 - 1][pixel_1 + 1] = random.uniform(0, 0.5)

    for i in range(number_of_hor_lines):
        number_of_pixels_line = int(random.randint(1, int(image.shape[0])))
        start_pixel_0 = random.randint(9, image.shape[0] - 9)
        start_pixel_1 = random.randint(9, image.shape[1] - 9)
        for j in range(number_of_pixels_line):
            # skip
            if random.randint(0, 2) == 1:
                continue
            pixel_0 = start_pixel_0 + 3 * j
            pixel_1 = start_pixel_1
            if pixel_0 >= image.shape[0] - 9:
                continue
            for j in range(9):
                image[pixel_0][pixel_1] = random.uniform(0, 0.5)
                image[pixel_0][pixel_1 + 1] = random.uniform(0, 0.5)
                image[pixel_0][pixel_1 - 1] = random.uniform(0, 0.5)
                image[pixel_0 + 1][pixel_1] = random.uniform(0, 0.5)
                image[pixel_0 + 1][pixel_1 + 1] = random.uniform(0, 0.5)
                image[pixel_0 + 1][pixel_1 - 1] = random.uniform(0, 0.5)
                image[pixel_0 - 1][pixel_1] = random.uniform(0, 0.5)
                image[pixel_0 - 1][pixel_1 - 1] = random.uniform(0, 0.5)
                image[pixel_0 - 1][pixel_1 + 1] = random.uniform(0, 0.5)
    return image


def rotate_image(image_array):
    image_array = (image_array * 255).astype(np.uint8)
    degree = random.randint(-30, 30)
    image = Image.fromarray(image_array)
    rotated_image = image.rotate(degree, expand=True, fillcolor=255)
    return np.array(rotated_image)


def zoom_in(image):
    zoom_factor = random.uniform(1, 1.5)
    # Calculate new dimensions based on the zoom factor
    new_height = int(image.shape[0] * zoom_factor)
    new_width = int(image.shape[1] * zoom_factor)

    # Resize the image using zoom function
    zoomed_image = zoom(image, (zoom_factor, zoom_factor), order=1)

    # Crop or pad the zoomed image to match the original shape
    y_start = (zoomed_image.shape[0] - image.shape[0]) // 2
    y_end = y_start + image.shape[0]
    x_start = (zoomed_image.shape[1] - image.shape[1]) // 2
    x_end = x_start + image.shape[1]

    if zoomed_image.shape[0] - y_start > image.shape[0] or zoomed_image.shape[1] - x_start > image.shape[1]:
        cropped_image = zoomed_image[y_start:y_end, x_start:x_end]
    else:
        cropped_image = np.pad(zoomed_image, ((y_start, image.shape[0] - zoomed_image.shape[0] - y_start),
                                              (x_start, image.shape[1] - zoomed_image.shape[1] - x_start)),
                               mode='constant')

    return cropped_image


def zoom_out(image):
    zoom_factor = random.uniform(1.0, 3.0)
    # zoom_fator = zoom1
    # Calculate new dimensions based on the zoom factor
    new_height = int(image.shape[0] / zoom_factor)
    new_width = int(image.shape[1] / zoom_factor)

    # Resize the image using zoom function
    zoomed_image = zoom(image, (1 / zoom_factor, 1 / zoom_factor), order=1)

    # Calculate the difference in dimensions
    diff_height = new_height - image.shape[0]
    diff_width = new_width - image.shape[1]

    # Crop the zoomed image to match the original shape
    y_start = diff_height // 2
    y_end = y_start + image.shape[0]
    x_start = diff_width // 2
    x_end = x_start + image.shape[1]

    # cropped_image = zoomed_image[y_start:y_end, x_start:x_end]
    cropped_image = zoomed_image

    pad_im_height = image.shape[0] - cropped_image.shape[0]
    pad_im_width = image.shape[1] - cropped_image.shape[1]

    for i in range(pad_im_width):
        num_rows = cropped_image.shape[0]

        # Create a column vector of ones
        ones_column = np.ones((num_rows, 1))

        if i % 2 == 0:
            # Concatenate the ones column with the original array
            cropped_image = np.hstack((cropped_image, ones_column))
        else:
            cropped_image = np.hstack((ones_column, cropped_image))

    for i in range(pad_im_height):
        num_cols = cropped_image.shape[1]

        # Create a column vector of ones
        ones_row = np.ones((1, num_cols))

        if i % 2 == 0:
            # Concatenate the ones column with the original array
            cropped_image = np.vstack((cropped_image, ones_row))
        else:
            cropped_image = np.vstack((ones_row, cropped_image))

    return cropped_image


if __name__ == '__main__':
    letters = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
    k = 5  # Replace with the desired number of folds
    kf = KFold(n_splits=k, shuffle=True)
    for letter in letters:
        # Directory to check
        directory = "data/train/" + letter
        items = os.listdir(directory)
        for j, (train_index, val_index) in enumerate(kf.split(items)):
            kf_train_dir = f"data{j}/train"
            kf_val_dir = f"data{j}/val"
            if not os.path.exists(kf_train_dir):
                os.mkdir(kf_train_dir)
            if not os.path.exists(kf_val_dir):
                os.mkdir(kf_val_dir)
            kf_train_dir = f"data{j}/train/{letter}"
            kf_val_dir = f"data{j}/val/{letter}"
            if not os.path.exists(kf_train_dir):
                os.mkdir(kf_train_dir)
            if not os.path.exists(kf_val_dir):
                os.mkdir(kf_val_dir)

            train_items = [items[i] for i in train_index]
            for i, train_file in enumerate(train_items):
                if train_file.endswith(".png"):
                    file_path = os.path.join(directory, train_file)
                    image = plt.imread(file_path)

                    image_pil = Image.fromarray((image * 255).astype(np.uint8))

                    # Save the image as a PNG file
                    image_pil.save(kf_train_dir + f"/im_{i}.png")


            val_items = [items[i] for i in val_index]
            for i, val_file in enumerate(val_items):
                if val_file.endswith(".png"):
                    file_path = os.path.join(directory, val_file)
                    image = plt.imread(file_path)

                    image_pil = Image.fromarray((image * 255).astype(np.uint8))

                    # Save the image as a PNG file
                    image_pil.save(kf_val_dir + f"/im_{i}.png")




