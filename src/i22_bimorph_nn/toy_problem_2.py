#######################################################################
# Toy problem 2 - 2D gaussain optimisation
#######################################################################

# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php
# https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
# https://gist.github.com/nvladimus/fc88abcece9c3e0dc9212c2adc93bfe7

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Linear, MSELoss
from torch.optim import SGD

os.system("clear")


def elliptical_gaussian(x_y: tuple, x0, y0, sigma_x, sigma_y, A, offset, theta):
    # Elliptical
    x, y = x_y

    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + A * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )

    return g


def generate_gaussian2(data_size, debug):
    """Function to fit, returns 2D gaussian function as 1D array"""

    # Start with some appropriate parameters
    x0 = random.uniform(-2, 2)
    y0 = random.uniform(-2, 2)
    sigma_x = random.uniform(8, 12)
    sigma_y = sigma_x
    A = random.uniform(17, 19)
    offset = random.uniform(-3, 3)
    theta = random.uniform(50, 70)

    if debug:
        print("RUN PARAMS")
        print(f"x0 = {x0}")
        print(f"y0 = {y0}")
        print(f"sigma_x = {sigma_x}")
        print(f"sigma_y = {sigma_y}")
        print(f"A = {A}")
        print(f"offset = {offset}")
        print(f"theta = {theta}")
        print("=" * 20)

    # Create the grid
    x, y = np.meshgrid(np.arange(-35, 35, 1), np.arange(-35, 35, 1))

    # Prepare numpy arrays for export
    images_out = np.empty(shape=(70, 70, data_size))
    params_out = np.empty(shape=(2, 1, data_size))
    full_out = np.empty(shape=(7, 1, data_size))

    # Create gaussian sequence
    for item in range(data_size):
        print(f"item: {item}")
        if item % 2 == 1:
            x0 += item / 40 * np.cos(item)
            y0 -= item / 20
            A += item * np.cos(item) / 4
        else:
            x0 -= item / 20 * np.sin(item)
            y0 += item / 40
        if item <= 7 and item % 3 == 0:
            sigma_x += item / 1.3
        elif item <= 5 and item % 2 == 0:
            sigma_y += item
            A += item * np.cos(item) / 4
        elif item <= 9 and item % 2 == 1:
            sigma_y += item / 2
            A += item * np.sin(item) / 4
        else:
            sigma_x += item / 8
            sigma_y += item / 8
        if item >= 6:
            A -= 2.5
        if sigma_x >= sigma_y:
            offset += np.sin(item * np.cos(item))
        else:
            offset += np.sin(item) / 10
            theta -= 0.1

        # Export images and variables
        images_out[:, :, item] = elliptical_gaussian(
            (x, y), x0, y0, sigma_x, sigma_y, A, offset, theta
        )
        params_out[:, :, item] = [
            [2 * np.sqrt(2 * np.log(2)) * sigma_x],
            [2 * np.sqrt(2 * np.log(2)) * sigma_y],
        ]
        full_out[:, :, item] = [
            [sigma_x],
            [sigma_y],
            [x0],
            [y0],
            [A],
            [offset],
            [theta],
        ]

        # Add some noise
        images_out[:, :, item] += (
            np.random.random(np.shape(images_out[:, :, item])) * item / 20
        )

        if debug:
            print("REAL")
            print(
                f"xcenter: {x0}, ycenter: {y0}, sigma_x: {sigma_x}, \
sigma_y: {sigma_y}, A: {A}, offset: {offset}"
            )
            print(f"FWHM_x (gen_gaussian2) = {2 * np.sqrt(2 * np.log(2)) * sigma_x}")
            print(f"FWHM_y (gen_gaussian2) = {2 * np.sqrt(2 * np.log(2)) * sigma_y}")

            # plt.imshow(np.array(z).reshape(50,50), cmap="hot",
            # interpolation="nearest")
            plt.subplot(2, data_size // 2, item + 1)
            plt.imshow(images_out[:, :, item], cmap="hot", interpolation="nearest")
    if debug:
        plt.show()

    # Reverse order of datasets
    images_out = np.flip(images_out, 2)
    params_out = np.flip(params_out, 2)
    full_out = np.flip(full_out, 2)

    return images_out, params_out, full_out


######################################################
# Model
######################################################


# define the model
class Optimise_FWHM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(in_features=3, out_features=6)

    def forward(self, x):
        return self.layer(x)


model = Optimise_FWHM()

# define the loss function
critereon = MSELoss()
# define the optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# define the number of epochs and the data set size
epochs = 2000
data_size = 10


# create our training loop
for epoch in range(epochs):
    X, y = generate_gaussian2(data_size)
    X = Variable(Tensor(np.array(X)))
    y = Variable(Tensor(np.array(y)))
    epoch_loss = 0
    y_pred = model(X)
    loss = critereon(y_pred, y)
    epoch_loss = loss.data
    if epoch % 1000 == 99:
        print(f"Epoch: {epoch} Loss: {epoch_loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# # Grab a single piece of test data and pass through the NN.
# # Compare the result to the forumla to determine model accuracy.
# model.eval()
# test_data = generate_gaussian2(1)
# prediction = model(Variable(Tensor(test_data[0][0])))

# print(test_data)
# print(prediction)
