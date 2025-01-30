#######################################################################
# Toy problem 3 - Gaussian Sequence prediction
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


def generate_gaussian2(x0, y0, sigma_x, sigma_y, A, offset, theta, data_size, debug):
    """Function to fit, returns 2D gaussian function"""

    # Create the grid
    x, y = np.meshgrid(np.arange(-50, 50, 1), np.arange(-50, 50, 1))

    # Prepare numpy arrays for export
    images_out = np.empty(shape=(data_size, 1, 100, 100))
    params_out = np.empty(shape=(data_size, 1, 7))

    # Deviate gaussian
    for item in range(data_size):
        # Non-linear, reproducible 'mirror deviation'
        # Model will learn this equation.
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
        images_out[item, 0, :, :] = elliptical_gaussian(
            x, y, x0, y0, sigma_x, sigma_y, A, offset, theta
        )
        params_out[item, :, :] = [
            [x0, y0, sigma_x, sigma_y, A, offset, theta],
        ]

        # Add some noise
        images_out[item, 0, :, :] += (
            np.random.random(np.shape(images_out[item, 0, :, :])) * item / 20
        )

        if debug:
            plt.subplot(2, data_size // 2, item + 1)
            plt.imshow(images_out[item, 0, :, :], cmap="hot", interpolation="nearest")
    if debug:
        plt.show()

    # Reverse order of datasets
    images_out = np.flip(images_out, 0)
    params_out = np.flip(params_out, 0)

    return images_out, params_out


################################
# Model Setup
################################


class Focusing_Sequence(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat1 = torch.nn.Flatten(start_dim=0, end_dim=2)
        self.fc1 = torch.nn.Linear(in_features=21, out_features=250)
        self.fc2 = torch.nn.Linear(in_features=250, out_features=250)
        self.fc3 = torch.nn.Linear(in_features=250, out_features=250)
        self.fc4 = torch.nn.Linear(in_features=250, out_features=7)

    def forward(self, image, params):
        x = self.flat1(params)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc4(x)
        return out


model = Focusing_Sequence()

# Define loss, optimiser and run parameters.
critereon = MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 5000
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
    images_out, params_out = generate_gaussian2(
        x0, y0, sigma_x, sigma_y, A, offset, theta, data_size, debug=False
    )

    slice = random.randrange(0, 7)
    next_images_out = images_out[slice + 3, 0, :, :]
    next_params_out = params_out[slice + 3, 0, :]
    images_out = images_out[slice : slice + 3, :, :, :]
    params_out = params_out[slice : slice + 3, :, :]

    # Pass sequence through the model
    image = Variable(Tensor(images_out.copy()))
    params = Variable(Tensor(params_out.copy()))
    next_images_out = Variable(Tensor(next_images_out.copy()))
    next_params_out = Variable(Tensor(next_params_out.copy()))
    epoch_loss = 0

    model_pred = model(image, params)

    # Calculate loss, backpropagate etc
    loss = critereon(model_pred, next_params_out)

    loss.backward()
    optimizer.step()
    epoch_loss = loss.data

    losses.append(loss.detach().numpy())

    if epoch % 1000 == 99:
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

# Testing sequence seed
x0 = random.uniform(-2, 2)
y0 = random.uniform(-2, 2)
sigma_x = random.uniform(8, 12)
sigma_y = sigma_x
A = random.uniform(17, 19)
offset = random.uniform(-3, 3)
theta = random.uniform(50, 70)
data_size = 10
slice = random.randrange(0, 6)

# Generate focusing sequence
images_out, params_out = generate_gaussian2(
    x0, y0, sigma_x, sigma_y, A, offset, theta, data_size, debug=False
)

# Pass testing sequence through the model
next_images_out = images_out[slice + 3, 0, :, :]
next_params_out = params_out[slice + 3, 0, :]
images_out_org = images_out
params_out_org = params_out
images_out = images_out[slice : slice + 3, :, :, :]
params_out = params_out[slice : slice + 3, :, :]

image = Variable(Tensor(images_out.copy()))
prev_params = params_out
params = Variable(Tensor(prev_params.copy()))

# Recursively pass images back into the model
model_sequence = []
for item in range(len(images_out_org) - slice):
    prediction = model(image, params)
    prediction_org = prediction.detach().clone()

    predicted_image = generate_gaussian2(
        prediction[0].detach().numpy(),
        prediction[1].detach().numpy(),
        prediction[2].detach().numpy(),
        prediction[3].detach().numpy(),
        prediction[4].detach().numpy(),
        prediction[5].detach().numpy(),
        prediction[6].detach().numpy(),
        data_size,
        debug=False,
    )[0][-1, 0, :, :]

    print()
    print("=" * 50)
    print("         TEST PARAMS vs MODEL PARAMS")
    print(f"Sequence no: {item + slice}")
    print(f"x0:      {params_out_org[item + slice, 0, 0]} vs {prediction_org[0]}")
    print(f"y0:      {params_out_org[item + slice, 0, 1]} vs {prediction_org[1]}")
    print(f"sigma_x: {params_out_org[item + slice, 0, 2]} vs {prediction_org[2]}")
    print(f"sigma_y: {params_out_org[item + slice, 0, 3]} vs {prediction_org[3]}")
    print(f"A:       {params_out_org[item + slice, 0, 4]} vs {prediction_org[4]}")
    print(f"offset:  {params_out_org[item + slice, 0, 5]} vs {prediction_org[5]}")
    print(f"theta:   {params_out_org[item + slice, 0, 6]} vs {prediction_org[6]}")
    print("=" * 50)

    # Ensure you've got exactly three images to pass into the model
    params = params.detach().numpy()
    prediction_org = prediction_org.detach().numpy()
    tmp_for_append = np.empty(shape=(1, 1, 7))
    tmp_for_append[0, 0, :] = prediction_org
    params = np.concatenate((params, tmp_for_append), axis=0)
    model_sequence.append(predicted_image)

    params = np.delete(params, 0, 0)
    params = Variable(Tensor(params.copy()))


for i in range(1, len(model_sequence)):
    plt.subplot(3, len(model_sequence), i)
    plt.imshow(
        model_sequence[i - 1],
        cmap="hot",
        interpolation="nearest",
        vmin=np.min(images_out_org[(i - 1) + slice]),
        vmax=np.max(images_out_org[(i - 1) + slice]),
    )
    plt.title("Model Out")
    for j in range(1, len(model_sequence)):
        plt.subplot(3, len(model_sequence), len(model_sequence) + j)
        plt.imshow(
            images_out_org[(j - 1) + slice, 0, :, :],
            cmap="hot",
            interpolation="nearest",
            vmin=np.min(images_out_org[(j - 1) + slice]),
            vmax=np.max(images_out_org[(j - 1) + slice]),
        )
        plt.title("Expected Out")
        for k in range(1, len(model_sequence)):
            plt.subplot(3, len(model_sequence), 2 * len(model_sequence) + k)
            plt.imshow(
                images_out_org[(k - 1) + slice, 0, :, :] - model_sequence[k - 1],
                cmap="hot",
                interpolation="nearest",
                vmin=np.min(images_out_org[(k - 1) + slice]),
                vmax=np.max(images_out_org[(k - 1) + slice]),
            )
            plt.title("Diff")

plt.show()
