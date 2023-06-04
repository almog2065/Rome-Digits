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
    return np.array(rotated_image)/255


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
    for l in letters:
        # Directory to check
        # directory = "data/train/" + l
        # aug_directory = "aug_data/train/" + l
        directory = "data/train/" + l
        aug_directory = "data/train/" + l

        # 1 - noisy + zoom in 
        items = os.listdir(directory)
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=100)
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # Plot the image
                image = plt.imread(file_path)
                noisy_im = noisy_aug(image)
                zoom_in_im = zoom_in(noisy_im)

                image_pil = Image.fromarray((zoom_in_im * 255).astype(np.uint8))

                # Save the image as a PNG file
                image_pil.save(aug_directory + f"/mix_1_{i}.png")

        # 2 - lines aug + noisy
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=100)
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # Plot the image
                image = plt.imread(file_path)
                noisy_im = lines_aug(image)
                noisy_2_im = noisy_aug(noisy_im)

                image_pil = Image.fromarray((noisy_2_im * 255).astype(np.uint8))

                # Save the image as a PNG file
                image_pil.save(aug_directory + f"/mix_2_{i}.png")

        # 3- rotate aug and zoom in
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=100)
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # Plot the image
                image = plt.imread(file_path)
                rotate_im = rotate_image(image)
                zoom_in_im = zoom_in(rotate_im)

                image_pil = Image.fromarray((zoom_in_im * 255).astype(np.uint8))

                # Save the image as a PNG file
                image_pil.save(aug_directory + f"/mix_3_{i}.png")

        # 4- nosiy lines and zoom in aug
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=100)
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # Plot the image
                image = plt.imread(file_path)
                noisy_lines = lines_aug(image)
                zoom_in_im = zoom_in(noisy_lines)

                image_pil = Image.fromarray((zoom_in_im * 255).astype(np.uint8))

                # Save the image as a PNG file
                image_pil.save(aug_directory + f"/mix_4_{i}.png")

        # 5 - noisy aug and zoom out aug
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=100)
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # Plot the image
                image = plt.imread(file_path)
                noisy_im = noisy_aug(image)
                zoom_out_im = zoom_out(noisy_im)

                image_pil = Image.fromarray((zoom_out_im * 255).astype(np.uint8))

                # Save the image as a PNG file
                image_pil.save(aug_directory + f"/mix_5_{i}.png")

        # mix6 - noisy  & rotate aug
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=100)
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # Plot the image
                image = plt.imread(file_path)
                noisy_im = noisy_aug(image)
                rotated_im = rotate_image(noisy_im)

                image_pil = Image.fromarray((rotated_im * 255).astype(np.uint8))

                # Save the image as a PNG file
                image_pil.save(aug_directory + f"/mix_6_{i}.png")


        # mix7 - rotate & lines 
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=100)
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # Plot the image
                image = plt.imread(file_path)
                rotated_im = rotate_image(image)
                lines_im = lines_aug(rotated_im)


                image_pil = Image.fromarray((lines_im * 255).astype(np.uint8))

                # Save the image as a PNG file
                image_pil.save(aug_directory + f"/mix_7_{i}.png")


        # mix8 - zoom out and rotate aug
        # Randomly select 100 items from the list
        random_items = random.sample(items, k=100)
        for i, filename in enumerate(random_items):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)

                # Plot the image
                image = plt.imread(file_path)
                zoom_out_im = zoom_out(image)
                rotated_im = rotate_image(zoom_out_im)

                image_pil = Image.fromarray((rotated_im * 255).astype(np.uint8))

                # Save the image as a PNG file
                image_pil.save(aug_directory + f"/mix_8_{i}.png")
