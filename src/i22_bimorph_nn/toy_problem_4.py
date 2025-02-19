#######################################################################
# Toy problem 4 - LSTM Model on Toy Problem
#######################################################################

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.share
from torch import Tensor
from torch.autograd import Variable
from torch.nn import MSELoss

os.system("clear")

################################
# Data Generation
################################


def elliptical_gaussian(x, y, x0, y0, sigma_x, sigma_y, A, offset, theta):
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


def generate_gaussian2(x0, y0, sigma_x, sigma_y, A, offset, theta, data_size):
    """Function to fit, returns 2D gaussian function"""

    # Create the grid
    x, y = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))

    # Prepare numpy arrays for export
    images_out = np.empty(shape=(data_size, 1, 200, 200))
    volt_out = np.empty(shape=(data_size, 7))

    # Deviate gaussian
    for item in range(data_size):
        # Non-linear, reproducible 'mirror deviation'
        # Model will learn this equation.
        if item % 2 == 1:
            x0 += (item / 40 * np.cos(item)) * 10
            y0 -= (item / 20) * 10
            A += item * np.cos(item) / 4
        else:
            x0 -= (item / 10 * np.sin(item)) * 10
            y0 += (item / 20) * 10
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
            offset += 10 * np.sin(item * np.cos(item))
        else:
            offset += np.sin(item) / 10
            theta -= 8.9 * (item * np.sin(item) ** 2)
            sigma_x += (50 - sigma_x) * 0.1
            sigma_y += (50 - sigma_y) * 0.1

        # Export images and variables
        images_out[item, 0, :, :] = elliptical_gaussian(
            x, y, x0, y0, sigma_x, sigma_y, A, offset, theta
        )
        volt_out[item:] = [
            [x0, y0, sigma_x, sigma_y, A, offset, theta],
        ]

        # Add some noise
        images_out[item, 0, :, :] += (
            np.random.random(np.shape(images_out[item, 0, :, :])) * item / 20
        )

    # Reverse order of datasets
    images_out = np.flip(images_out, 0)
    volt_out = np.flip(volt_out, 0)

    # Collect data into batches
    next_images_out = np.array([images_out[i + 3] for i in range(len(images_out) - 3)])
    images_out = np.array([images_out[i : i + 3] for i in range(len(images_out) - 3)])
    next_volt = np.array([volt_out[i + 3] for i in range(len(volt_out) - 3)])

    return images_out, next_images_out, next_volt


################################
# Model Setup
################################


class Focusing_Sequence(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2, padding=1
            ),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(7744, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
        )

        self.sequence = torch.nn.LSTM(
            input_size=128,
            hidden_size=int((2 / 3 * 128) + 44),
            num_layers=1,
            batch_first=True,
        )

        self.fully_connected = torch.nn.Sequential(torch.nn.Linear(129, 7))

    def forward(self, image):
        batch_size, sequence_length = image.shape[:2]
        image_features = []

        for t in range(batch_size):
            image_batch = image[t, :]
            image_features.append(self.image_conv(image_batch))

        image_features = torch.stack(image_features, dim=0)
        LSTM_out, h_n = self.sequence(image_features)

        out = self.fully_connected(LSTM_out[:, -1])

        return out


x0 = random.uniform(-10, 10)
y0 = random.uniform(-10, 10)
sigma_x = random.uniform(14, 16)
sigma_y = sigma_x
A = random.uniform(17, 19)
offset = random.uniform(-30, 30)
theta = random.uniform(20, 160)
data_size = 10

# Generate focusing sequence
images_out, next_images_out, next_volt = generate_gaussian2(
    x0, y0, sigma_x, sigma_y, A, offset, theta, data_size
)

model = Focusing_Sequence()

# Define loss, optimiser and run parameters.
critereon = MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)

epochs = 50
data_size = 10

losses = []


################################
# Training
################################

for epoch in range(epochs):
    # Seed for focusing sequence
    x0 = random.uniform(-2, 2)
    y0 = random.uniform(-2, 2)
    sigma_x = random.uniform(8, 12)
    sigma_y = sigma_x
    A = random.uniform(17, 19)
    offset = random.uniform(-3, 3)
    theta = random.uniform(50, 70)
    data_size = 10

    optimizer.zero_grad()

    # Generate focusing sequence
    images_out, next_image_out, next_volt = generate_gaussian2(
        x0, y0, sigma_x, sigma_y, A, offset, theta, data_size
    )

    image = Variable(Tensor(images_out.copy()))
    next_volt = Variable(Tensor(next_volt.copy()))

    # Calculate loss, backpropagate etc
    epoch_loss = 0
    model_pred = model(image)

    loss = critereon(model_pred, next_volt)

    loss.backward()
    optimizer.step()
    epoch_loss = loss.data

    # Collect loss for plotting
    losses.append(loss.detach().numpy())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Loss: {epoch_loss}")


################################
# Testing
################################

model.eval()

# Display loss
plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.show()

# Seed for testing
x0 = random.uniform(-10, 10)
y0 = random.uniform(-10, 10)
sigma_x = random.uniform(14, 16)
sigma_y = sigma_x
A = random.uniform(17, 19)
offset = random.uniform(-30, 30)
theta = random.uniform(20, 160)
data_size = 10

# Generate focusing sequence
images_out, next_images_out, next_volt = generate_gaussian2(
    x0, y0, sigma_x, sigma_y, A, offset, theta, data_size
)

image = Variable(Tensor(images_out.copy()))
next_image = Variable(Tensor(next_image_out.copy()))

# Model prediction
prediction = model(image)

prediction_copy = prediction.detach().numpy().copy()
x, y = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))

for i in range(7):
    images_out = elliptical_gaussian(x, y, *prediction_copy[i])
    plt.subplot(1, 7, i + 1)
    plt.imshow(images_out, cmap="hot", interpolation="nearest")
plt.show()
