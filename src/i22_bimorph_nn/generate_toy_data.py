################################
# Data Generation
################################


import os

import numpy as np

# import torch
# import torch.share
# from torchvision import transforms

os.system("clear")


def elliptical_gaussian(x, y, X_0, Y_0, SIGMA_X, SIGMA_Y, A, THETA):
    """Generates an individual 2D Guassian image with the given parameters"""
    a = (np.cos(THETA) ** 2) / (2 * SIGMA_X**2) + (np.sin(THETA) ** 2) / (
        2 * SIGMA_Y**2
    )
    b = -(np.sin(2 * THETA)) / (4 * SIGMA_X**2) + (np.sin(2 * THETA)) / (4 * SIGMA_Y**2)
    c = (np.sin(THETA) ** 2) / (2 * SIGMA_X**2) + (np.cos(THETA) ** 2) / (
        2 * SIGMA_Y**2
    )
    g = A * np.exp(
        -(a * ((x - X_0) ** 2) + 2 * b * (x - X_0) * (y - Y_0) + c * ((y - Y_0) ** 2))
    )

    return g


def generate_gaussian2(X_0, Y_0, SIGMA_X, SIGMA_Y, A, THETA, DATA_SIZE):
    """Creates the 2D gaussian time sequence"""

    # Create the grid
    x, y = np.meshgrid(np.arange(-128, 128, 1), np.arange(-128, 128, 1))

    # Prepare numpy arrays for export.
    images_out = np.empty(shape=(DATA_SIZE, 1, 256, 256), dtype=float)
    volt_out = np.empty(shape=(DATA_SIZE, 6), dtype=float)

    # Create time series. Start small and deviate overtime.
    # Non-linear, reproducible 'mirror deviation'.
    for item in range(DATA_SIZE):
        if item % 2 == 1:
            A += item * np.cos(item) / 4
        else:
            A += item * np.cos(2 * item) / 4
        if item <= 7 and item % 3 == 0:
            SIGMA_X += item / 1.3
        elif item <= 5 and item % 2 == 0:
            SIGMA_Y += item
            A += item * np.cos(item) / 4
        elif item <= 9 and item % 2 == 1:
            SIGMA_Y += item / 2
            A += item * np.sin(item) / 4
        else:
            SIGMA_X += item / 8
            SIGMA_Y += item / 8
        if item >= 6:
            A -= 2.5
        else:
            THETA -= 8.9 * (item * np.sin(item) ** 2)
            SIGMA_X += (50 - SIGMA_X) * 0.1
            SIGMA_Y += (50 - SIGMA_Y) * 0.1

        # Export images and variables
        images_out[item, 0, :, :] = elliptical_gaussian(
            x, y, X_0, Y_0, SIGMA_X, SIGMA_Y, A, THETA
        )
        volt_out[item:] = [
            [X_0, Y_0, SIGMA_X, SIGMA_Y, A, THETA],
        ]

        # Add some noise
        images_out[item, 0, :, :] += (
            np.random.random(np.shape(images_out[item, 0, :, :])) * item / 20
        )

    # Reverse image order, want unfocused to focused.
    images_out = np.flip(images_out, 0)
    volt_out = np.flip(volt_out, 0)

    next_images_out = np.float32(
        np.array([images_out[i + 3] for i in range(len(images_out) - 3)])
    )

    # Sliding window sequence into batches of three images.
    images_out = np.float32(
        np.array([images_out[i : i + 3] for i in range(len(images_out) - 3)])
    )

    # 'next step' for each image batch.
    next_volt = np.float32(
        np.array([volt_out[i + 3] for i in range(len(volt_out) - 3)])
    )

    # Channel information at each step.
    volts_out = np.float32(
        np.array([volt_out[i : i + 3] for i in range(len(volt_out) - 3)])
    )

    # # Normalise the images and 'voltages'.
    # norm_img = transforms.Normalize(mean=torch.mean(image), std=torch.std(image))

    # # Potentially using different techniques here could be erroneous?
    # row_mean = next_volt.mean(dim=1, keepdim=True)
    # row_std = next_volt.std(dim=1, keepdim=True)
    # norm_next_volt = (next_volt - row_mean) / row_std

    # row_mean2 = volts_out.mean(dim=1, keepdim=True)
    # row_std2 = volts_out.std(dim=1, keepdim=True)
    # norm_volts_out = (volts_out - row_mean2) / (row_std2 + 1e-10)

    # norm_images_out = norm_img(image)

    return images_out, next_images_out, next_volt, volts_out
