import os

import numpy as np
import scipy.optimize as opt

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

    return g.ravel()


def calculate_FWHM(xyz_arr, theta):
    # Split up the data
    x = xyz_arr[:, 0]
    y = xyz_arr[:, 1]
    z = xyz_arr[:, 2]
    # Guess parameters: x0, y0, sigma_x, simga_y, A, offset, theta.
    # Cheating with theta do prevent flipping x and y.
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

    print(f"FWHM_x = {FWHM_x}")
    print(f"FWHM_y = {FWHM_y}")

    return (FWHM_x, FWHM_y)


def generate_gaussian2(data_size):
    """Function to fit, returns 2D gaussian function as 1D array"""

    inputs_FWHM = []
    loss_FWHM = []

    # Loop data_size times to generate the data
    for _ in range(data_size):
        # Generate x between 0 and 1000
        # x = np.random.randint(2000) / 1000

        # Create independant variables
        x = np.arange(-25, 25, 0.1)
        y = np.arange(-25, 25, 0.1)

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

        print(f"FWHM_x (generate_gaussian2) = {2 * np.sqrt(2 * np.log(2)) * sigma_x}")
        print(f"FWHM_y (generate_gaussian2) = {2 * np.sqrt(2 * np.log(2)) * sigma_y}")

        # Combine everything
        xyz_data = np.array([x.ravel(), y.ravel(), z]).T

        # Corresponding y value using the function
        inputs = xyz_data
        loss = calculate_FWHM(xyz_data, theta)

        # Append the values to our input and labels lists
        inputs_FWHM.append([inputs])
        loss_FWHM.append([loss])

    return inputs_FWHM, loss_FWHM


data_size = 10
print(generate_gaussian2(data_size))
