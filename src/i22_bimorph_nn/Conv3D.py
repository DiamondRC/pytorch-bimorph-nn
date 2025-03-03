#######################################################################
# Toy problem 4 - LSTM Model on Toy Problem
#######################################################################

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.share
from torch import tensor
from torch.autograd import Variable
from torchvision import transforms

os.system("clear")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

################################
# Data Generation
################################


def elliptical_gaussian(x, y, x0, y0, sigma_x, sigma_y, A, theta):
    """Generates an individual 2D Guassian image with the given parameters"""
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = A * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )

    return g


def generate_gaussian2(x0, y0, sigma_x, sigma_y, A, theta, data_size):
    """Creates the 2D gaussian time sequence"""

    # Create the grid
    x, y = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))

    # Prepare numpy arrays for export.
    images_out = np.empty(shape=(data_size, 1, 200, 200), dtype=float)
    volt_out = np.empty(shape=(data_size, 6), dtype=float)

    # Create time series. Start small and deviate overtime.
    # Non-linear, reproducible 'mirror deviation'.
    for item in range(data_size):
        if item % 2 == 1:
            A += item * np.cos(item) / 4
        else:
            A += item * np.cos(2 * item) / 4
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
        else:
            theta -= 8.9 * (item * np.sin(item) ** 2)
            sigma_x += (50 - sigma_x) * 0.1
            sigma_y += (50 - sigma_y) * 0.1

        # Export images and variables
        images_out[item, 0, :, :] = elliptical_gaussian(
            x, y, x0, y0, sigma_x, sigma_y, A, theta
        )
        volt_out[item:] = [
            [x0, y0, sigma_x, sigma_y, A, theta],
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

    return images_out, next_images_out, next_volt, volts_out


################################
# Model Setup
################################


class Focusing_Sequence(torch.nn.Module):
    """Conv-LSTM Model. Takes image and extracts features,
    then processes them the LSTM to learn how they change over time."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=(1, 3, 3)),
            torch.nn.BatchNorm3d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout3d(p=0.2),
            # torch.nn.AvgPool3d(kernel_size=2, stride=2),
            # torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2),
            #
            torch.nn.Conv3d(32, 32, kernel_size=(1, 3, 3)),
            torch.nn.BatchNorm3d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout3d(p=0.2),
            # torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2),
            #
            torch.nn.Conv3d(32, 64, kernel_size=(1, 3, 3)),
            torch.nn.BatchNorm3d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout3d(p=0.2),
            # torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2),
            # torch.nn.AvgPool3d(kernel_size=2, stride=2),
            #
            torch.nn.Conv3d(64, 128, kernel_size=(1, 3, 3)),
            torch.nn.BatchNorm3d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout3d(p=0.2),
            # torch.nn.AvgPool3d(kernel_size=2, stride=2),
            #
            torch.nn.Conv3d(128, 128, kernel_size=(1, 3, 3)),
            torch.nn.BatchNorm3d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout3d(p=0.2),
            #
            torch.nn.Conv3d(128, 128, kernel_size=(1, 3, 3)),
            torch.nn.BatchNorm3d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout3d(p=0.2),
            #
            torch.nn.Conv3d(128, 256, kernel_size=(1, 3, 3)),
            torch.nn.BatchNorm3d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout3d(p=0.2),
            #
            torch.nn.Conv3d(256, 256, kernel_size=(1, 3, 3)),
            torch.nn.BatchNorm3d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout3d(p=0.2),
            torch.nn.AvgPool3d(kernel_size=2, stride=2),
            #
            torch.nn.Flatten(),
        )
        self.pos = torch.nn.Sequential(
            torch.nn.Linear(2166784, 2),
            torch.nn.LeakyReLU(),
        )
        self.sigmax = torch.nn.Sequential(
            torch.nn.Linear(2166784, 1),
            torch.nn.ReLU(),
        )
        self.sigmay = torch.nn.Sequential(
            torch.nn.Linear(2166784, 1),
            torch.nn.ReLU(),
        )
        self.A = torch.nn.Sequential(
            torch.nn.Linear(2166784, 1),
            torch.nn.ReLU(),
        )
        self.theta = torch.nn.Sequential(
            torch.nn.Linear(2166784, 1),
            torch.nn.ReLU(),
        )

    def forward(self, image, volts_out):
        image = torch.permute(image, (0, 2, 1, 3, 4))
        image = self.conv(image)

        params_out = []

        position = self.pos(image)
        sigmax = self.sigmax(image)
        sigmay = self.sigmay(image)
        amplitude = self.A(image)
        theta = self.theta(image)

        params_out.append(
            torch.cat((position, sigmax, sigmay, amplitude, theta), dim=-1)
        )
        params_out = torch.stack(params_out, dim=0)

        out = torch.squeeze(params_out)

        return out


# Pre-set weights in all model layers.
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(
            # m.weight, mode="fan_in", nonlinearity="leaky_relu"
            m.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(
            m.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# Generate consistent seeds
def generate_seed():
    x0 = random.uniform(-80, 80)
    y0 = random.uniform(-80, 80)
    sigma_x = random.uniform(8, 12)
    sigma_y = sigma_x
    A = random.uniform(17, 19)
    theta = random.uniform(20, 160)
    data_size = 10
    return x0, y0, sigma_x, sigma_y, A, theta, data_size


model = Focusing_Sequence()
model.apply(init_weights)

if torch.cuda.is_available():
    model.to("cuda")

# Define loss, optimiser and run parameters.
criterion = torch.nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

epochs = 1500
data_size = 10

losses = []
layers = []
grads = []


################################
# Training
################################

for epoch in range(epochs):
    # Generate a focusing sequence.
    x0, y0, sigma_x, sigma_y, A, theta, data_size = generate_seed()

    images_out, next_image_out, next_volt, volts_out = generate_gaussian2(
        x0, y0, sigma_x, sigma_y, A, theta, data_size
    )

    image = Variable(tensor(images_out.copy(), device="cuda"))
    next_volt = Variable(tensor(next_volt.copy(), device="cuda"))
    volts_out = Variable(tensor(volts_out.copy(), device="cuda"))

    # Normalise the images and 'voltages'.
    norm_img = transforms.Normalize(mean=torch.mean(image), std=torch.std(image))

    # Potentially using different techniques here could be erroneous?
    row_mean = next_volt.mean(dim=1, keepdim=True)
    row_std = next_volt.std(dim=1, keepdim=True)
    norm_next_volt = (next_volt - row_mean) / row_std

    row_mean2 = volts_out.mean(dim=1, keepdim=True)
    row_std2 = volts_out.std(dim=1, keepdim=True)
    norm_volts_out = (volts_out - row_mean2) / (row_std2 + 1e-10)

    # print("="*40)
    # print(volts_out)
    # print(row_mean2)
    # print(row_std2)
    # print("="*40)

    norm_images_out = norm_img(image)

    # Calculate prediction, loss, backpropagate etc.
    epoch_loss = 0
    model_pred = model(norm_images_out, norm_volts_out)

    optimizer.zero_grad()
    loss = criterion(model_pred, norm_next_volt)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    epoch_loss = loss.data

    # Collect loss on CPU for plotting.
    losses.append(loss.cpu().detach().numpy())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Loss: {epoch_loss}")

    # Collect debug information relating to gradient values.
    if epochs == 0 or epochs % 50 == 0:
        for name, param in model.named_parameters():
            if "weight" in name:  # Only consider weight parameters.
                layers.append(name)
                # Computing L2 norm (Frobenius norm) of the weights.
                grad = torch.norm(param, p=2).item()
                grads.append(grad)


################################
# Testing
################################

model.eval()

# Display loss
plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.savefig("imgs/Loss.png")
plt.close()


for img_name in range(15):
    # Generate focusing sequence
    x0, y0, sigma_x, sigma_y, A, theta, data_size = generate_seed()

    images_out, next_images_out, next_volt, volts_out = generate_gaussian2(
        x0, y0, sigma_x, sigma_y, A, theta, data_size
    )

    # Normalise inputs.
    image = Variable(tensor(images_out.copy(), device="cuda"))
    volts_out = Variable(tensor(volts_out.copy(), device="cuda"))

    norm_img = transforms.Normalize(mean=torch.mean(image), std=torch.std(image))

    row_mean2 = volts_out.mean(dim=1, keepdim=True)
    row_std2 = volts_out.std(dim=1, keepdim=True)
    norm_volts_out = (volts_out - row_mean2) / (row_std2 + 1e-10)

    norm_images_out = norm_img(image)

    # Model prediction
    prediction = model(norm_images_out, norm_volts_out)

    # Denormalise
    prediction = prediction * row_std + row_mean

    # Debug out
    print("=" * 20)
    print(f"x0: {prediction[:, 0].cpu().detach().numpy()}")
    print(f"x0_real: {next_volt[:, 0]}")
    print(f"y0: {prediction[:, 1].cpu().detach().numpy()}")
    print(f"y0_real: {next_volt[:, 1]}")
    print(f"sigma_x: {prediction[:, 2].cpu().detach().numpy()}")
    print(f"sigma_x_real: {next_volt[:, 2]}")
    print(f"sigma_y: {prediction[:, 3].cpu().detach().numpy()}")
    print(f"sigma_y_real: {next_volt[:, 3]}")
    print(f"A: {prediction[:, 4].cpu().detach().numpy()}")
    print(f"A_real: {next_volt[:, 4]}")
    print(f"theta: {prediction[:, 5].cpu().detach().numpy()}")
    print(f"theta_real: {next_volt[:, 5]}")

    # Copy/grid for plotting
    prediction_copy = prediction.cpu().detach().numpy().copy()
    x, y = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))

    # Compare model prediction to data
    for j in range(7):
        plt.subplot(3, 7, j + 1)
        plt.imshow(next_images_out[j, 0], cmap="hot", interpolation="nearest")

        model_images_out = elliptical_gaussian(x, y, *prediction_copy[j])
        plt.subplot(3, 7, j + 8)
        plt.imshow(
            model_images_out,
            cmap="hot",
            interpolation="nearest",
            vmin=np.min(next_images_out[j]),
            vmax=np.max(next_images_out[j]),
        )

        plt.subplot(3, 7, j + 15)
        plt.imshow(
            next_images_out[j, 0] - model_images_out,
            cmap="hot",
            interpolation="nearest",
        )
    plt.savefig(f"imgs/{img_name}.png")

# Plot gradient values for debugging.
for name, param in model.named_parameters():
    if param.grad is not None:
        plt.figure(figsize=(6, 4))
        plt.hist(
            param.grad.cpu().view(-1).detach().numpy(),
            bins=500,
            alpha=0.7,
            color="blue",
        )
        plt.title(f"Gradient Histogram for {name}")
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"imgs/{name}.png")
        plt.close()
