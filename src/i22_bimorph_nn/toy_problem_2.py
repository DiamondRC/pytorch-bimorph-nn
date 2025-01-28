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
import torch.share
from torch import Tensor
from torch.autograd import Variable
from torch.nn import MSELoss
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


def generate_gaussian2(x0, y0, sigma_x, sigma_y, A, offset, theta, data_size, debug):
    """Function to fit, returns 2D gaussian function as 1D array"""

    # Create the grid
    x, y = np.meshgrid(np.arange(-35, 35, 1), np.arange(-35, 35, 1))

    # Prepare numpy arrays for export
    images_out = np.empty(shape=(data_size, 1, 70, 70))
    params_out = np.empty(shape=(data_size, 1, 2))
    channels_out = np.empty(shape=(data_size, 1, 5))
    truth_out = np.empty(shape=(1, 1, 7))

    # Fill with initial values
    truth_out[0, :, :] = [
        [sigma_x, sigma_y, x0, y0, A, offset, theta],
    ]

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
            (x, y), x0, y0, sigma_x, sigma_y, A, offset, theta
        )
        params_out[item, :, :] = [
            [sigma_x, sigma_y],
        ]

        channels_out[item, :, :] = [
            [x0, y0, A, offset, theta],
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
    channels_out = np.flip(channels_out, 0)
    truth_out = np.flip(channels_out, 0)

    return images_out, params_out, channels_out, truth_out


######################################################
# Model
######################################################


# define the model
class Optimise_FWHM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels = C_in, e.g. 1 for greyscale, size out = 61
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(10, 10))
        # Add non-linearity then pool
        self.relu = torch.nn.ReLU()
        # 2x2 w/ stride 2 halves each dim, 6x6 -> 3x3 etc.
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Now flatten the result to match parameters we want to learn
        # start/end dim are the tensor dimensions to squish
        self.flat1 = torch.nn.Flatten(start_dim=2, end_dim=3)

        # Compress result to match the input 'channels'
        self.fc1 = torch.nn.Linear(in_features=100, out_features=7)
        # Fuse 'channel' params with image
        self.fc2 = torch.nn.Linear(in_features=14, out_features=7)

        # Combine all images in the batch.
        self.flat2 = torch.nn.Flatten(start_dim=0, end_dim=2)
        # Return
        self.fc3 = torch.nn.Linear(in_features=70, out_features=7)

    def forward(self, image, params, channels):
        img1 = self.conv(image)
        # print(img1.size())
        # [10, 1, 61, 61]
        img2 = self.relu(img1)

        img3 = self.pool(img2)
        # [10, 1, 30, 30]
        # print(img3.size())

        img4 = self.conv(img3)
        # print(img4.size())
        # [10, 1, 21, 21]
        img5 = self.relu(img4)

        img6 = self.pool(img5)
        # [10, 1, 10, 10]
        # print(img6.size())

        img7 = self.flat1(img6)
        # [10, 1, 100]
        # print(img7.size())

        # print()
        img8 = self.fc1(img7)
        img8 = self.relu(img8)
        # [10, 1, 7]
        # print(f"img8: {img8.size()}")

        all_params = torch.cat((params, channels), 2)
        # [10, 1, 7]
        # print(f"all_params: {all_params.size()}")

        fuse_img_n_params = torch.cat((img8, all_params), 2)
        # [10, 1, 14]
        # print(fuse_img_n_params.size())

        combined_params = self.fc2(fuse_img_n_params)
        combined_params = self.relu(combined_params)
        # [10, 1, 7]
        # print(combined_params.size())

        combined_imgs = self.flat2(combined_params)
        # [70]
        # print(combined_imgs.size())

        out = self.fc3(combined_imgs)
        # out = self.relu(out)

        # print(out)

        return out


model = Optimise_FWHM()

# define the loss function
critereon = MSELoss(reduction="sum")
# define the optimizer
optimizer = SGD(model.parameters(), lr=0.00001)

# define the number of epochs and the data set size
epochs = 20000
data_size = 10


# create our training loop
for epoch in range(epochs):
    # Start with some appropriate parameters
    x0 = random.uniform(-2, 2)
    y0 = random.uniform(-2, 2)
    sigma_x = random.uniform(8, 12)
    sigma_y = sigma_x
    A = random.uniform(17, 19)
    offset = random.uniform(-3, 3)
    theta = random.uniform(50, 70)
    data_size = 10

    images_out, params_out, channels_out, truth_out = generate_gaussian2(
        x0, y0, sigma_x, sigma_y, A, offset, theta, data_size, debug=False
    )
    image = Variable(Tensor(images_out.copy()))
    params = Variable(Tensor(params_out.copy()))
    channels = Variable(Tensor(channels_out.copy()))
    epoch_loss = 0

    # Pass image and 'channels' into model
    # to get predicted next channel config.
    y_pred = model(image, params, channels)

    # Create synthetic image from model to determine preformance.
    image_pred = generate_gaussian2(
        x0=y_pred[0].detach().numpy(),
        y0=y_pred[1].detach().numpy(),
        sigma_x=y_pred[2].detach().numpy(),
        sigma_y=y_pred[3].detach().numpy(),
        A=y_pred[4].detach().numpy(),
        offset=y_pred[5].detach().numpy(),
        theta=y_pred[6].detach().numpy(),
        data_size=1,
        debug=False,
    )[0]

    # Check the image predicted by the model
    # against the known good image.
    image_pred = Variable(Tensor(image_pred.copy()), requires_grad=True)
    loss = critereon(image_pred[:][0], image[:][0])
    epoch_loss = loss.data

    # Visualise
    if epoch % 1000 == 99:
        print(f"Epoch: {epoch} Loss: {epoch_loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Now want to test model with unseen data.
# Generate some new params and give them to the model,
# compare against the analytic value.
# Visually display both for easy comparison.

print("-" * 45)

x0 = random.uniform(-2, 2)
y0 = random.uniform(-2, 2)
sigma_x = random.uniform(8, 12)
sigma_y = sigma_x
A = random.uniform(17, 19)
offset = random.uniform(-3, 3)
theta = random.uniform(50, 70)
data_size = 10

print()
print("=" * 45)
print("TEST PARAMS:")
print(f"x0 = {x0}")
print(f"y0 = {y0}")
print(f"sigma_x = {sigma_x}")
print(f"sigma_y = {sigma_y}")
print(f"A = {A}")
print(f"offset = {offset}")
print(f"theta = {theta}")
print(f"FWHM_x (Real) = {2 * np.sqrt(2 * np.log(2)) * sigma_x}")
print(f"FWHM_y (Real) = {2 * np.sqrt(2 * np.log(2)) * sigma_y}")
print()

model.eval()
images_out, params_out, channels_out, truth_out = generate_gaussian2(
    x0, y0, sigma_x, sigma_y, A, offset, theta, data_size, debug=True
)

image = Variable(Tensor(images_out.copy()))
params = Variable(Tensor(params_out.copy()))
channels = Variable(Tensor(channels_out.copy()))
prediction = model(image, params, channels)

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
)


print("MODEL PARAMS:")
print(f"x0 = {prediction[0]}")
print(f"y0 = {prediction[1]}")
print(f"sigma_x = {prediction[2]}")
print(f"sigma_y = {prediction[3]}")
print(f"A = {prediction[4]}")
print(f"offset = {prediction[5]}")
print(f"theta = {prediction[6]}")
print(f"FWHM_x (Model) = {2 * np.sqrt(2 * np.log(2)) * prediction[2]}")
print(f"FWHM_y (Model) = {2 * np.sqrt(2 * np.log(2)) * prediction[3]}")
print("=" * 45)
print()

plt.subplot(1, 3, 1)
plt.imshow(
    images_out[-1][0],
    cmap="hot",
    interpolation="nearest",
    vmin=np.min(images_out[-1][0]),
    vmax=np.max(images_out[-1][0]),
)
plt.title("Expected")

plt.subplot(1, 3, 2)
plt.imshow(
    predicted_image[0][0, 0, :, :],
    cmap="hot",
    interpolation="nearest",
    vmin=np.min(images_out[-1][0]),
    vmax=np.max(images_out[-1][0]),
)
plt.title("Model")

plt.subplot(1, 3, 3)
plt.imshow(
    images_out[-1][0] - predicted_image[0][0, 0, :, :],
    cmap="hot",
    interpolation="nearest",
    vmin=np.min(images_out[-1][0]),
    vmax=np.max(images_out[-1][0]),
)
plt.title("Diff")

plt.show()
