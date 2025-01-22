import os

import numpy as np
import scipy.optimize as opt

os.system("clear")


def generate_gaussian2(x_y: tuple, x0, y0, sigma_x, sigma_y, A, offset, theta=67):
    """Function to fit, returns 2D gaussian function as 1D array"""
    x, y = x_y
    x0 = float(x0)
    y0 = float(y0)

    # # 2D circular
    # g = offset + A * np.exp(
    #     -(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2))
    # )

    # Elliptical
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

    print(f"FWHM_x (generate_gaussian2) = {2 * np.sqrt(2 * np.log(2)) * sigma_x}")
    print(f"FWHM_y (generate_gaussian2) = {2 * np.sqrt(2 * np.log(2)) * sigma_y}")

    return g.ravel()


# Create independant variables
x = np.arange(-100, 100, 0.1)
y = np.arange(-100, 100, 0.1)

# Create the grid
x, y = np.meshgrid(x, y)

# Make a Gaussian
# z = generate_gaussian2((x, y), 0, 0.4, -1.07, 0.8, 1, 0)
print("called generate_gaussian2")
z = generate_gaussian2((x, y), -1.05, 1.67, 54, 43, 88, 0)
# Add some noise
# z += np.random.random(z.shape) / 3
z += np.random.random(z.shape)

# Combine everything
print("combined")
xyz_data = np.array([x.ravel(), y.ravel(), z]).T


def calculate_FWHM(xyz_arr):
    # Split up the data
    x = xyz_arr[:, 0]
    y = xyz_arr[:, 1]
    z = xyz_arr[:, 2]
    # Guess parameters: xpos, ypos, sigma_x, simga_y, A, offset.
    initial_guess = (1, 1, 1, 1, 5, 2)

    # Fit the data
    popt, pcov = opt.curve_fit(generate_gaussian2, (x, y), z, initial_guess)

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

    # Use to calculate the FWHM
    # FWHM_x = np.abs(4 * sigma_x * np.sqrt(-0.05 * np.log(0.5)))
    # FWHM_y = np.abs(4 * sigma_y * np.sqrt(-0.05 * np.log(0.5)))

    FWHM_x = 2 * np.sqrt(2 * np.log(2)) * sigma_x
    FWHM_y = 2 * np.sqrt(2 * np.log(2)) * sigma_y

    print(f"FWHM_x = {FWHM_x}")
    print(f"FWHM_y = {FWHM_y}")

    return (FWHM_x, FWHM_y)


calculate_FWHM(xyz_data)


# Have way to create gaussians and calculate their FWHM.
# Want model which takes gaussians, their parameters and FWHM's.
# Model should then learn to minimise the FWHM by returning
# suitable parameters.

# Need to make the guassians deformed.
