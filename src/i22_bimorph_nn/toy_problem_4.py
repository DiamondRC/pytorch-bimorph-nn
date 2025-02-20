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
    images_out = np.empty(shape=(data_size, 1, 200, 200), dtype=float)
    volt_out = np.empty(shape=(data_size, 7), dtype=float)

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
    next_images_out = np.float32(
        np.array([images_out[i + 3] for i in range(len(images_out) - 3)])
    )
    images_out = np.float32(
        np.array([images_out[i : i + 3] for i in range(len(images_out) - 3)])
    )
    next_volt = np.float32(
        np.array([volt_out[i + 3] for i in range(len(volt_out) - 3)])
    )

    return images_out, next_images_out, next_volt


################################
# Model Setup
################################


class Focusing_Sequence(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5)),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1)),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1)),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(p=0.5),
            torch.nn.LayerNorm(128),
            torch.nn.LeakyReLU(),
        )

        self.sequence = torch.nn.LSTM(
            input_size=128,
            hidden_size=int((2 / 3 * 128) + 44),
            num_layers=7,
            batch_first=True,
            dropout=0.4,
            bidirectional=False,
        )

        self.LSTM_out = (torch.nn.LayerNorm(int((2 / 3 * 128) + 44)),)
        # self.LSTM_out = torch.nn.LayerNorm(32),

        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(129, 7),
            # torch.nn.Linear(32, 7),
            torch.nn.Dropout(p=0.5),
            torch.nn.LayerNorm(7),
            torch.nn.LeakyReLU(),
        )

    def forward(self, image):
        batch_size, sequence_length = image.shape[:2]
        image_features = []

        for t in range(batch_size):
            image_batch = image[t, :]
            image_features.append(self.image_conv(image_batch))

        image_features = torch.stack(image_features, dim=0)
        LSTM_out, h_n = self.sequence(image_features)

        out = self.fully_connected(LSTM_out[:, -1, :])

        return out


# Pre-set weights in all model layers.
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        # torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in",
        # nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(
            m.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


model = Focusing_Sequence()
model.apply(init_weights)

if torch.cuda.is_available():
    model.to("cuda")

# Define loss, optimiser and run parameters.
criterion = torch.nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

epochs = 3000000
data_size = 10

losses = []
layers = []
grads = []


################################
# Training
################################

for epoch in range(epochs):
    # Seed for focusing sequence
    x0 = random.uniform(-80, 80)
    y0 = random.uniform(-80, 80)
    sigma_x = random.uniform(8, 12)
    sigma_y = sigma_x
    A = random.uniform(17, 19)
    offset = random.uniform(0, 0)
    theta = random.uniform(20, 160)
    data_size = 10

    # Generate focusing sequence
    images_out, next_image_out, next_volt = generate_gaussian2(
        x0, y0, sigma_x, sigma_y, A, offset, theta, data_size
    )

    image = Variable(tensor(images_out.copy(), device="cuda"))
    next_volt = Variable(tensor(next_volt.copy(), device="cuda"))

    norm_img = transforms.Normalize(mean=torch.mean(image), std=torch.std(image))

    row_mean = next_volt.mean(dim=1, keepdim=True)
    row_std = next_volt.std(dim=1, keepdim=True)
    norm_next_volt = (next_volt - row_mean) / row_std

    norm_images_out = norm_img(image)

    # Calculate loss, backpropagate etc
    epoch_loss = 0
    model_pred = model(norm_images_out)

    optimizer.zero_grad()
    loss = criterion(model_pred, norm_next_volt)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    epoch_loss = loss.data

    # Collect loss for plotting
    losses.append(loss.cpu().detach().numpy())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Loss: {epoch_loss}")

    if epochs == 0 or epochs % 50 == 0:
        for name, param in model.named_parameters():
            if "weight" in name:  # Only consider weight parameters
                layers.append(name)
                # Compute L2 norm (Frobenius norm) of the weights
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
plt.show()


for _ in range(50):
    # Seed for testing
    x0 = random.uniform(-80, 80)
    y0 = random.uniform(-80, 80)
    sigma_x = random.uniform(8, 12)
    sigma_y = sigma_x
    A = random.uniform(17, 19)
    offset = random.uniform(0, 0)
    theta = random.uniform(20, 160)
    data_size = 10

    # Generate focusing sequence
    images_out, next_images_out, next_volt = generate_gaussian2(
        x0, y0, sigma_x, sigma_y, A, offset, theta, data_size
    )

    image = Variable(tensor(images_out.copy(), device="cuda"))
    next_volt = Variable(tensor(next_volt.copy(), device="cuda"))

    norm_img = transforms.Normalize(mean=torch.mean(image), std=torch.std(image))

    row_mean = next_volt.mean(dim=1, keepdim=True)
    row_std = next_volt.std(dim=1, keepdim=True)
    norm_next_volt = (next_volt - row_mean) / row_std

    norm_images_out = norm_img(image)

    # Model prediction
    prediction = model(norm_images_out)

    prediction = prediction * row_std + row_mean

    print("=" * 20)
    print(f"x0: {prediction[:, 0].cpu().detach().numpy()}")
    print(f"y0: {prediction[:, 1].cpu().detach().numpy()}")
    print(f"sigma_x: {prediction[:, 2].cpu().detach().numpy()}")
    print(f"sigma_y: {prediction[:, 3].cpu().detach().numpy()}")
    print(f"A: {prediction[:, 4].cpu().detach().numpy()}")
    print(f"offset: {prediction[:, 5].cpu().detach().numpy()}")
    print(f"theta: {prediction[:, 6].cpu().detach().numpy()}")

    prediction_copy = prediction.cpu().detach().numpy().copy()
    x, y = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))

    # Compare model prediction to data
    for j in range(7):
        plt.subplot(3, 7, j + 1)
        plt.imshow(next_images_out[j, 0], cmap="hot", interpolation="nearest")

        model_images_out = elliptical_gaussian(x, y, *prediction_copy[j])
        plt.subplot(3, 7, j + 8)
        # plt.imshow(model_images_out, cmap="hot", interpolation="nearest",
        # vmin=np.min(next_images_out[j]), vmax=np.max(next_images_out[j]))
        plt.imshow(model_images_out, cmap="hot", interpolation="nearest")

        plt.subplot(3, 7, j + 15)
        plt.imshow(
            next_images_out[j, 0] - model_images_out,
            cmap="hot",
            interpolation="nearest",
        )
    plt.show()

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
        plt.show()
