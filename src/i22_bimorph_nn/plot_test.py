import os
import random

import matplotlib.pyplot as plt
import numpy as np

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

    # return g.flatten()
    return g


# def calculate_FWHM(x, y, z):
#     # Guess parameters: x0, y0, sigma_x, simga_y, A, offset, theta.
#     initial_guess = (1, 1, 40, 40, 60, 2, 60)

#     # Fit the data
#     popt, pcov = opt.curve_fit(elliptical_gaussian,
# (x, y), z, initial_guess, maxfev=3200)
#     # Return properties of the data
#     xcenter, ycenter, sigma_x, sigma_y, A, offset = (
#         popt[0],
#         popt[1],
#         popt[2],
#         popt[3],
#         popt[4],
#         popt[5],
#     )
#     print("="*30)
#     print("FAKE")
#     print(f"xcenter: {xcenter}, ycenter: {ycenter}, sigma_x: {sigma_x}, \
# sigma_y: {sigma_y}, A: {A}, offset: {offset}")

#     FWHM_x = 2 * np.sqrt(2 * np.log(2)) * sigma_x
#     FWHM_y = 2 * np.sqrt(2 * np.log(2)) * sigma_y

#     print(f"FWHM_x = {FWHM_x}")
#     print(f"FWHM_y = {FWHM_y}")

#     return (FWHM_x, FWHM_y)


def generate_gaussian2(data_size, debug):
    """Function to fit, returns 2D gaussian function as 1D array"""

    # x0 = -1.05
    # y0 = 1.67
    # sigma_x = 10
    # sigma_y = 10
    # A = 20
    # offset = 0
    # theta = 67

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

    # Create independant variables
    x = np.arange(-35, 35, 1)
    y = np.arange(-35, 35, 1)

    # Create the grid
    x, y = np.meshgrid(x, y)

    # images_out = np.empty(shape=(70,70,data_size))
    # params_out = np.empty(shape=(2,1,data_size))
    # Want to export array with ten items in reverse order
    # Each item contains np.array of size (70,70)
    # Also want to give model parameters of 'beam':
    # sigma_x
    # sigma_y
    # e.g. (2,1)

    # Loop data_size times to generate the data
    for item in range(data_size):
        # x0 += random.uniform(-1/2,1/2)
        # y0 += random.uniform(-1/2,1/2)
        # sigma_x += random.uniform(item/4,item)
        # sigma_y += random.uniform(item/4,item)
        # A += item*np.cos(item)/4
        # offset += random.uniform(-1,1)
        # theta += random.uniform(-5,5)

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

        z = elliptical_gaussian((x, y), x0, y0, sigma_x, sigma_y, A, offset, theta)

        # Add some noise
        # z += np.random.random(z.shape) / 3
        z += np.random.random(z.shape) * item / 20

        # calculate_FWHM(x, y, z)

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
            plt.subplot(2, 5, item + 1)
            plt.imshow(z, cmap="hot", interpolation="nearest")
    if debug:
        plt.show()

    return images_out, params_out


data_size = 10
images_out, params_out = generate_gaussian2(data_size, debug=True)


print(np.shape(images_out))
print(np.shape(params_out))

print(params_out[:, :, 3])
print()

# print(array_out[:][:][11])
