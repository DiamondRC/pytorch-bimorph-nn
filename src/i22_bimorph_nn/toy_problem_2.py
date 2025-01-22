#######################################################################
# Toy problem 2 - 2D gaussain optimisation
#######################################################################

# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php
# https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
# https://gist.github.com/nvladimus/fc88abcece9c3e0dc9212c2adc93bfe7

import os

import numpy as np
import scipy.optimize as opt
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Linear, MSELoss
from torch.optim import SGD

os.system("clear")


# Want to create datasets of 2D Guassians for training.
# Create model which attempts to minmise the FWHM,
# for some given Gaussian parameters.
# Use the Loss function to optimise for this.


# Architecture:
# Generate 2D gaussians.
# Calculate their FWHM.
# Tell the model the guassian parameters and the FWHM value.
# Have minimum beamsize of 150x180.
# Model learns which values give the smallest FWHM,
# e.g. if the model is given A and xo, it will find the best yo.


def elliptical_gaussian(x_y: tuple, x0, y0, sigma_x, sigma_y, A, offset, theta):
    # Elliptical Gaussian equation w/ rotation.
    # Add deformation down the line.

    # Packed for curve_fit
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

    return g.ravel()


def calculate_FWHM(xyz_arr, theta):
    # Split up the data
    x = xyz_arr[:, 0]
    y = xyz_arr[:, 1]
    z = xyz_arr[:, 2]

    # Guess parameters: x0, y0, sigma_x, simga_y, A, offset, theta.
    # Cheating with theta to prevent flipping x and y.
    initial_guess = (1, 1, 40, 40, 60, 2, theta)

    # Fit the data
    popt, pcov = opt.curve_fit(elliptical_gaussian, (x, y), z, initial_guess)
    # Return properties of the data
    xcenter, ycenter, sigma_x, sigma_y, A, offset = (
        popt[0],
        popt[1],
        popt[2],
        popt[3],
        popt[4],
        popt[5],
    )

    print(f"xcenter: {xcenter}, ycenter: {ycenter}, A: {A}, offset: {offset}")

    FWHM_x = 2 * np.sqrt(2 * np.log(2)) * sigma_x
    FWHM_y = 2 * np.sqrt(2 * np.log(2)) * sigma_y

    # print(f"FWHM_x = {FWHM_x}")
    # print(f"FWHM_y = {FWHM_y}")

    return (FWHM_x, FWHM_y)


def generate_gaussian2(data_size):
    """Function to fit, returns 2D gaussian function as 1D array"""

    inputs_FWHM = []
    loss_FWHM = []

    # Loop data_size times to generate the data
    for _ in range(data_size):
        # Create independant variables
        x = np.arange(-50, 50, 1)
        y = np.arange(-50, 50, 1)

        # Create the grid
        x, y = np.meshgrid(x, y)

        x0 = -1.05
        y0 = 1.67
        sigma_x = 54
        sigma_y = 43
        A = 88
        offset = 0
        theta = 67

        z = elliptical_gaussian((x, y), x0, y0, sigma_x, sigma_y, A, offset, theta)

        # Add some noise
        # z += np.random.random(z.shape) / 3
        z += np.random.random(z.shape)

        # print(f"FWHM_x (generate_gaussian2) = {2 * np.sqrt(2 * np.log(2)) * sigma_x}")
        # print(f"FWHM_y (generate_gaussian2) = {2 * np.sqrt(2 * np.log(2)) * sigma_y}")

        # Combine everything
        xyz_data = np.array([x.ravel(), y.ravel(), z]).T

        # Corresponding y value using the function
        # inputs = xyz_data, x0, y0, sigma_x, sigma_y, A, offset, theta
        inputs = xyz_data
        loss = calculate_FWHM(xyz_data, theta)

        # Append the values to our input and labels lists
        inputs_FWHM.append([inputs])
        loss_FWHM.append([loss])

    return inputs_FWHM, loss_FWHM


######################################################
# Model
######################################################

# Analytic 2D gaussian generated.
# Feed in function parameters as well as grid output.
# Model will learn about the FWHM.
#
# Can compare the learnt FWHM with the analytic
#
# Then will give the model some new guassian + info.
# Model should then return new parameters to generate
# a gaussian with a tighter FWHM.


# define the model
class Optimise_FWHM(torch.nn.Module):
    def __init__(self):
        # 6 parameters to define the gaussian.
        # 2D grid of intensity values.

        # Want out 2 values, the FWHM for both x and y.
        # super().__init__()
        # self.fc1 = Linear(in_features=7, out_features=6)
        # self.fc2 = Linear(in_features=6, out_features=5)
        # self.fc3 = Linear(in_features=5, out_features=2)

        # super().__init__()
        # self.layer = torch.nn.Sequential(
        #    torch.nn.Linear(in_features=3, out_features=2), torch.nn.ReLU()
        # )
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
    # y = torch.unsqueeze(y, dim=2)
    epoch_loss = 0
    y_pred = model(X)
    loss = critereon(y_pred, y)
    epoch_loss = loss.data
    if epoch % 1000 == 99:
        print(f"Epoch: {epoch} Loss: {epoch_loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Grab a single piece of test data and pass through the NN.
# Compare the result to the forumla to determine model accuracy.
model.eval()
test_data = generate_gaussian2(1)
prediction = model(Variable(Tensor(test_data[0][0])))

print(test_data)
print(prediction)
