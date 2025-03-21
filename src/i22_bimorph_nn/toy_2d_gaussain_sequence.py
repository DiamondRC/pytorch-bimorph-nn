#######################################################################
# Predict next image parameters in 2D Gaussian Sequence
#######################################################################

import os

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
        return image_sequence, shifted_voltage_sequence, voltage_sequence


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
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.voltage = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.LeakyReLU(),
        )

        self.params = torch.nn.Linear(96, 6)

    def forward(self, x, y):
        x = x.view(BATCH_SIZE * SEQUENCE_LENGTH, 1, 224, 224)
        x = self.image_conv(x)
        x = x.view(BATCH_SIZE * SEQUENCE_LENGTH, -1)
        y = y.view(BATCH_SIZE * SEQUENCE_LENGTH, 6)
        y = self.voltage(y)
        x = torch.concatenate((x, y), dim=1)
        x = self.params(x)
        x = x.view(BATCH_SIZE, SEQUENCE_LENGTH, 6)
        return x


# Define Constants and Hyperparameters
NUM_EPOCHS = 60
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


# Create model and send to GPU if possible
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
    for image_sequence, shifted_voltage_sequence, voltage_sequence in train_loader:
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
if not NUM_EPOCHS == 1:
    plt.plot(range(NUM_EPOCHS), loss_data)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()
    plt.close()

# Plot model prediction against truth data
with torch.no_grad():
    for i, (_, shifted_voltage_sequence, voltage_sequence) in enumerate(test_loader):
        # Grab only one batch from the dataloader for visual comparison.
        if i == 1:
            break

        # Model out next voltages
        test_model_sequence = model(
            image_sequence.cuda(), voltage_sequence.cuda()
        ).cpu()

        # Data out next voltages
        test_shifted_voltages = shifted_voltage_sequence

    # Create fixed grid for plotting
    X, Y = np.meshgrid(np.arange(-112, 112, 1), np.arange(-112, 112, 1))

    for i in range(len(test_shifted_voltages)):
        for j in range(9):
            # Model prediction
            ax1 = plt.subplot(2, 9, j + 1)
            plt.imshow(
                elliptical_gaussian(X, Y, *test_model_sequence[i, j].numpy()),
                cmap="hot",
                interpolation="nearest",
            )
            # Data next voltage
            ax1 = plt.subplot(2, 9, j + 10)
            plt.imshow(
                elliptical_gaussian(X, Y, *test_shifted_voltages[i, j].numpy()),
                cmap="hot",
                interpolation="nearest",
            )
        plt.show()

# Display model weights
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
