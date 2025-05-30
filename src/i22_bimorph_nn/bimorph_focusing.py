#######################################################################
# i22 Bimorph Mirror Focusing - PyTorch Neural Network
#######################################################################

import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import tensor
from torch.utils.data import DataLoader, Dataset, random_split

os.system("clear")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


################################
# Data Loading
################################


def pair_files(path):
    print("Beginning file checks...")
    # Only interested in nexus or h5 files
    files = [
        file
        for file in os.listdir(path)
        if file.endswith(".nxs") or file.endswith(".h5")
    ]
    file_dict = {}

    for f in files:
        match = re.match(rf"{BEAMLINE}-(\d+).*", f)
        if match:
            # Extract number from filename
            num = match.group(1)
            # If there's no number already, create entry
            if num not in file_dict:
                file_dict[num] = []
            # Add file to associated number
            file_dict[num].append(f)

    return_paired_files = [
        (os.path.join(path, f1), os.path.join(path, f2))
        for _, file_list in file_dict.items()  # Grab each item in the pair dict
        if len(file_list) == 2  # Check if there's two files for the number
        for f1, f2 in [file_list]
    ]  # Select the two files and return path

    valid_pairs = []
    for pair in return_paired_files:
        # Order files in dict
        fail = False
        h5 = str
        nexus = str
        if pair[0][-3:] == ".h5":
            h5, nexus = pair
        else:
            nexus, h5 = pair

        # Disqualify bad pairs
        with h5py.File(f"{nexus}", "r") as f:
            # Check presense of correct number of channels
            try:
                f[
                    f"entry/instrument/bimorph_vfm/channels-{VFM_CHANNEL_NO}-output_voltage"
                ]
                f[
                    f"entry/instrument/bimorph_hfm/channels-{HFM_CHANNEL_NO}-output_voltage"
                ]
            except Exception as _:
                fail = True
        with h5py.File(f"{h5}", "r") as f:
            if not np.shape(f["entry/data/data"]) == (SEQUENCE_LENGTH, *DETECTOR_RES):
                fail = True
        if not fail:
            valid_pairs.append((nexus, h5))
    print("Beginning file checks... DONE")
    return valid_pairs


class BimorphHDF5Dataset(Dataset):
    def __init__(self):
        self.file_root = PATH
        self.file_list = os.listdir(PATH)
        self.paired_files = PAIRED_FILES

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        image_sequence = np.empty(shape=(SEQUENCE_LENGTH, 1, *DETECTOR_RES), dtype="f")
        volt_sequence = np.empty(
            shape=(SEQUENCE_LENGTH, VFM_CHANNEL_NO + HFM_CHANNEL_NO), dtype="f"
        )
        shifted_volt_sequence = np.empty(
            shape=(SEQUENCE_LENGTH, VFM_CHANNEL_NO + HFM_CHANNEL_NO), dtype="f"
        )
        nexus, h5 = self.paired_files[idx]
        with h5py.File(nexus, "r") as f:
            for item in range(1, VFM_CHANNEL_NO + 1):
                volt_sequence[:, item - 1] = f[
                    f"entry/instrument/bimorph_vfm/channels-{item}-output_voltage"
                ]
            for item in range(1, HFM_CHANNEL_NO + 1):
                volt_sequence[:, VFM_CHANNEL_NO + item - 1] = f[
                    f"entry/instrument/bimorph_hfm/channels-{item}-output_voltage"
                ]

        cut_volt_sequence = volt_sequence[1:]
        shifted_volt_sequence = np.concatenate(
            (cut_volt_sequence, FOCUSED_CHANNEL_VOLTAGES), axis=0, dtype="f"
        )

        with h5py.File(h5, "r") as f:
            image_sequence[:, 0, :, :] = f["entry/data/data"]

        return (
            tensor(image_sequence),
            tensor(volt_sequence),
            tensor(shifted_volt_sequence),
        )


################################
# Hyperparams
################################

# Data
PATH = "/dls/i22/data/2025/nr40718-1/"
HFM_CHANNEL_NO = 12
VFM_CHANNEL_NO = 32
SEQUENCE_LENGTH = 15
DETECTOR_RES = (2464, 2056)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BEAMLINE = "i22"
PAIRED_FILES = pair_files(PATH)
FOCUSED_CHANNEL_VOLTAGES = np.array(
    [
        500,
        500,
        91,
        -203,
        20,
        33,
        -53,
        44,
        -196,
        -142,
        -214,
        -454,
        -89,
        -35,
        -129,
        -150,
        -83,
        -93,
        104,
        -305,
        -500,
        -29,
        -33,
        -52,
        128,
        11,
        -24,
        139,
        -198,
        -9,
        144,
        380,
        -214,
        -320,
        -317,
        -324,
        -382,
        -272,
        -284,
        -232,
        -322,
        -428,
        -214,
        -216,
    ]
).reshape(1, -1)

# Model
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
LEARNING_RATE_PATIENCE = 10
NUM_EPOCHS = 10

loss_data = []
val_data = []

################################
# Dataloaders
################################

# Split up the training data into training, validation and testing datasets.
print("Beginning file loading...")
training_data = BimorphHDF5Dataset()
print("Beginning file loading... DONE")

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
# Model
################################


class Bimorph_Focusing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_conv = torch.nn.Sequential(
            torch.nn.AvgPool2d(2, 2),
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
            torch.nn.Dropout2d(p=0.2),
            torch.nn.AvgPool2d(2, 2),
            #
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            #
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.features_to_channels = torch.nn.Linear(32, HFM_CHANNEL_NO + VFM_CHANNEL_NO)

    def forward(self, images):
        batch = images.size()[0]
        x = images.view(batch * SEQUENCE_LENGTH, 1, *DETECTOR_RES)
        x = self.image_conv(x)
        x = x.view(batch, SEQUENCE_LENGTH, -1)
        x = self.features_to_channels(x)
        return x


# Pre-set weights in all model layers.
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(
            m.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# Create model and send to GPU if possible.
model = Bimorph_Focusing()
model.apply(init_weights)

if torch.cuda.is_available():
    model.to("cuda")

# Define loss, optimiser and run parameters.
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=LEARNING_RATE_PATIENCE
)

################################
# Training
################################

# Early stopping.
best_epoch = -1
best_val_loss = float("inf")

# Training loop.
print("Training...")
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    num_samples = 0.0
    val_loss = 0.0
    num_val_samples = 0.0

    # Train the model.
    model.train()
    for image_sequence, _voltage_sequence, shifted_voltage_sequence in train_loader:
        optimizer.zero_grad()
        output = model(image_sequence.cuda())
        loss = criterion(output, shifted_voltage_sequence.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Store batch training loss.
        batch_size = image_sequence.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    # Compute total loss for the epoch.
    epoch_loss = running_loss / num_samples

    # Validation test, adjust learning rate.
    model.eval()
    with torch.no_grad():
        for (
            val_image_sequence,
            _val_voltage_sequence,
            val_shifted_voltage_sequence,
        ) in val_loader:
            val_output = model(val_image_sequence.cuda())
            loss = criterion(val_output, val_shifted_voltage_sequence.cuda())

            # Store batch validation loss.
            batch_size = val_image_sequence.size(0)
            val_loss += loss.item() * batch_size
            num_val_samples += batch_size

    # Compute average loss across the batch.
    val_loss = val_loss / num_val_samples

    scheduler.step(val_loss)

    # Return training information.
    lr = optimizer.param_groups[0]["lr"]
    if epoch % 1 == 0:
        print(
            f"Epoch: [{epoch}/{NUM_EPOCHS - 1}], "
            f"Training Loss: {epoch_loss}, "
            f"Val Loss: {val_loss}, Lr: {lr:.6f}"
        )
    loss_data.append(epoch_loss)
    val_data.append(val_loss)

print("Training... DONE")


################################
# Testing
################################

# Prepare model for testing.
model.eval()

# Display loss.
if not NUM_EPOCHS == 1:
    plt.plot(range(NUM_EPOCHS), loss_data)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()
    plt.close()
