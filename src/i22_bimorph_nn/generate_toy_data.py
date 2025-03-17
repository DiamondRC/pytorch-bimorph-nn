################################
# Data Generation
################################


import os
import random

import h5py
import numpy as np
import torch
import torch.share
from torch import tensor
from torchvision import transforms

# import matplotlib.pyplot as plt

os.system("clear")


def elliptical_gaussian(x, y, x_0, y_0, sigma_x, sigma_y, amp, theta):
    """Generates an individual 2D Guassian image with the given parameters"""
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = amp * np.exp(
        -(a * ((x - x_0) ** 2) + 2 * b * (x - x_0) * (y - y_0) + c * ((y - y_0) ** 2))
    )

    return g


def generate_gaussian2(x_0, y_0, sigma_x, sigma_y, amp, theta, SEQUENCE_LENGTH):
    """Creates the 2D gaussian time sequence"""

    # Create the grid
    x, y = np.meshgrid(np.arange(-112, 112, 1), np.arange(-112, 112, 1))

    # Prepare numpy arrays for export.
    image_sequence = np.empty(shape=(SEQUENCE_LENGTH, 1, 224, 224), dtype=float)
    voltage_sequence = np.empty(shape=(SEQUENCE_LENGTH, 6), dtype=float)

    # Create time series. Start small and deviate overtime.
    # Non-linear, reproducible 'mirror deviation'.
    for item in range(SEQUENCE_LENGTH):
        if item % 2 == 1:
            amp += item * np.cos(item) / 4
        else:
            amp += item * np.cos(2 * item) / 4
        if item <= 7 and item % 3 == 0:
            sigma_x += item / 1.3
        elif item <= 5 and item % 2 == 0:
            sigma_y += item
            amp += item * np.cos(item) / 4
        elif item <= 9 and item % 2 == 1:
            sigma_y += item / 2
            amp += item * np.sin(item) / 4
        else:
            sigma_x += item / 8
            sigma_y += item / 8
        if item >= 6:
            amp -= 2.5
        else:
            theta -= 8.9 * (item * np.sin(item) ** 2)
            sigma_x += (50 - sigma_x) * 0.1
            sigma_y += (50 - sigma_y) * 0.1

        # Export images and variables
        image_sequence[-item - 1, 0, :, :] = elliptical_gaussian(
            x, y, x_0, y_0, sigma_x, sigma_y, amp, theta
        )
        voltage_sequence[-item - 1 :] = [
            [x_0, y_0, sigma_x, sigma_y, amp, theta],
        ]

    image_sequence = tensor(image_sequence)
    voltage_sequence = tensor(voltage_sequence)

    # Normalise the images and 'voltages'.
    norm_img = transforms.Normalize(
        mean=torch.mean(image_sequence), std=torch.std(image_sequence)
    )
    norm_image_sequence = norm_img(tensor(image_sequence))

    volt_mean = voltage_sequence.mean(dim=1, keepdim=True)
    volt_std = voltage_sequence.std(dim=1, keepdim=True)
    norm_voltage_sequence = (voltage_sequence - volt_mean) / (volt_std + 1e-10)

    # Return outputs
    return norm_image_sequence, norm_voltage_sequence


# Uncomment below to check validity of sequence

# return norm_image_sequence, image_sequence, norm_voltage_sequence, voltage_sequence

# import matplotlib.pyplot as plt
# norm_image_sequence, image_sequence, norm_voltage_sequence, voltage_sequence = (
#     generate_gaussian2(
#         x_0=0, y_0=0, sigma_x=10, sigma_y=10, amp=20, theta=60, SEQUENCE_LENGTH=10
#     )
# )
# for j in range(10):
#     plt.subplot(2, 10, j + 1)
#     plt.imshow(norm_image_sequence[j, 0], cmap="hot", interpolation="nearest")

#     plt.subplot(2, 10, j + 11)
#     plt.imshow(
#         image_sequence[j, 0],
#         cmap="hot",
#         interpolation="nearest",
#         # vmin=np.min(next_images_out[j]),
#         # vmax=np.max(next_images_out[j]),
#     )
# plt.show()

# print(norm_voltage_sequence)
# print(voltage_sequence)


def generate_seed():
    X_0 = random.uniform(-80, 80)
    Y_0 = random.uniform(-80, 80)
    SIGMA_X = random.uniform(8, 12)
    SIGMA_Y = SIGMA_X
    A = random.uniform(17, 19)
    THETA = random.uniform(20, 160)
    return X_0, Y_0, SIGMA_X, SIGMA_Y, A, THETA


NUM_SEQUENCES = 1000
SEQUENCE_LENGTH = 10

with h5py.File("gaussian_2d_sequences.hdf5", "w") as f:
    image_dataset = f.create_dataset(
        "gaussian_seq", (NUM_SEQUENCES, SEQUENCE_LENGTH, 1, 224, 224), dtype="f"
    )
    voltage_dataset = f.create_dataset(
        "voltage_seq", (NUM_SEQUENCES, SEQUENCE_LENGTH, 6), dtype="f"
    )

    for i in range(0, NUM_SEQUENCES):
        image_sequence, voltage_sequence = generate_gaussian2(
            *generate_seed(), SEQUENCE_LENGTH
        )
        image_dataset[i] = image_sequence
        voltage_dataset[i] = voltage_sequence
