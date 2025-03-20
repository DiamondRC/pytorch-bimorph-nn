#######################################################################
# Predict next image parameters in 2D Gaussian Sequence
#######################################################################

import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.share
from generate_toy_data import elliptical_gaussian
from torch.utils.data import DataLoader, Dataset, random_split

os.system("clear")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


################################
# Load data
################################


class GaussianHDF5Dataset(Dataset):
    def __init__(self):
        self.file_path = PATH
        with h5py.File(self.file_path, "r") as f:
            self.data_len = f["gaussian_seq"].shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as f:
            image_sequence = f["gaussian_seq"][idx]
            image_sequence = image_sequence[:-1]
            voltage_sequence = f["voltage_seq"][idx]
            shifted_voltage_sequence = voltage_sequence[1:]
            voltage_sequence = voltage_sequence[:-1]
            norm_info = f["normalisation_info"][idx]
        return image_sequence, shifted_voltage_sequence, voltage_sequence, norm_info


################################
# Model Setup
################################


class Focusing_Sequence(torch.nn.Module):
    """Conv-LSTM Model. Takes image and extracts features,
    then processes them the LSTM to learn how they change over time."""

    def __init__(self):
        super().__init__()

        self.image_conv = torch.nn.Sequential(
            # Extract image features.
            torch.nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(5, 5), padding=2
            ),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(2, 2),
            #
            torch.nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),
            # torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            # torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.voltage = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.LeakyReLU(),
        )

        self.params = torch.nn.Linear(128, 6)

    def forward(self, x, y):
        # print(x.size())
        x = x.view(BATCH_SIZE * SEQUENCE_LENGTH, 1, 224, 224)
        # print(x.size())
        x = self.image_conv(x)
        # print(x.size())
        x = x.view(BATCH_SIZE * SEQUENCE_LENGTH, -1)
        # print(x.size())
        # print(y.size())
        y = y.view(BATCH_SIZE * SEQUENCE_LENGTH, 6)
        # print(y.size())
        y = self.voltage(y)
        # print(y.size())
        # print(f"concat: {x.size()}, {y.size()}")
        x = torch.concatenate((x, y), dim=1)
        # print(x.size())
        x = self.params(x)
        # print(x.size())
        x = x.view(BATCH_SIZE, SEQUENCE_LENGTH, 6)
        # print(x.size())
        return x


# Define Constants and Hyperparameters
NUM_EPOCHS = 30
LEARNING_RATE = 0.01
SEQUENCE_LENGTH = 9
BATCH_SIZE = 5
PATH = "gaussian_2d_sequences.hdf5"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
loss_data = []
layers = []
grads = []


# Pre-set weights in all model layers.
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
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
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Split up the training data into training, validation and testing datasets
training_data = GaussianHDF5Dataset()
train_size = int(TRAIN_RATIO * len(training_data))
val_size = int(VAL_RATIO * len(training_data))
test_size = len(training_data) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    training_data, [train_size, val_size, test_size]
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
val_loader = DataLoader(  # For consistency in testing, shuffle=False
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)


################################
# Training
################################

for epoch in range(NUM_EPOCHS):
    for image_sequence, shifted_voltage_sequence, voltage_sequence, _ in train_loader:
        optimizer.zero_grad()
        output = model(image_sequence.cuda(), voltage_sequence.cuda())
        loss = criterion(output, shifted_voltage_sequence.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch: {epoch} Loss: {loss.data}")
    loss_data.append(loss.cpu().detach().numpy())

    # Collect debug information relating to gradient values.
    if NUM_EPOCHS == 0 or NUM_EPOCHS % 50 == 0:
        for name, param in model.named_parameters():
            if "weight" in name:  # Only consider weight parameters.
                layers.append(name)
                # Computing L2 norm (Frobenius norm) of the weights.
                grad = torch.norm(param, p=2).item()
                grads.append(grad)


################################
# Testing
################################

# Prepare model for testing
model.eval()

# Display loss
plt.plot(range(NUM_EPOCHS), loss_data)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()
plt.close()

# Plot model against truth
predicted_voltages = []
expected_voltages = []
with torch.no_grad():
    x, y = np.meshgrid(np.arange(-128, 128, 1), np.arange(-128, 128, 1))

    norm_scopes = None
    for (
        image_sequence,
        shifted_voltage_sequence,
        voltage_sequence,
        norm_info,
    ) in test_loader:
        output = model(image_sequence.cuda(), voltage_sequence.cuda())
        predicted_voltages.append(output.cpu())
        expected_voltages.append(shifted_voltage_sequence.cpu())
        norm = norm_info

    for _ in range(5):
        index = random.randrange(1, 5)
        index_predicted_voltages = np.asarray(predicted_voltages[index])
        index_expected_voltages = np.asarray(expected_voltages[index])
        err = index_expected_voltages - index_predicted_voltages
        per_err = (
            (index_predicted_voltages - index_expected_voltages)
            / index_expected_voltages
        ) * 100
        print(f"expected_voltages:  {index_expected_voltages[0, 0]}")
        print(f"predicted_voltages: {index_predicted_voltages[0, 0]}")
        print(f"err:                {err[0, 0]}")
        print(f"per_err:            {per_err[0, 0]}")
        print("=========")

        print(norm_info[:, 1:].size())
        print(np.shape(index_predicted_voltages))

        # Denormalise
        index_predicted_voltages = (
            index_predicted_voltages * norm_info[:, 1:, :1].numpy()
            + norm_info[:, 1:, 1:2].numpy()
        )

        print(index_predicted_voltages)
        print(predicted_voltages[index])
        for j in range(9):
            plt.subplot(1, 9, j + 1)
            plt.imshow(
                elliptical_gaussian(x, y, *index_predicted_voltages[0, j]),
                cmap="hot",
                interpolation="nearest",
            )
        plt.show()
