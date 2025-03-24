#######################################################################
# i22 Bimorph Mirror Focusing - pyTorch Neural Network
#######################################################################

import os
import re

import h5py
import numpy as np
import torch
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


def pair_files(PATH):
    print("Beginning file checks...")
    # Only interested in nexus or h5 files
    files = [
        file
        for file in os.listdir(PATH)
        if file.endswith(".nxs") or file.endswith(".h5")
    ]
    file_dict = {}

    for f in files:
        match = re.match(rf"{BEAMLINE}-(\d+).*", f)
        #  + re.escape(".h5")
        if match:
            # Extract number from filename
            num = match.group(1)
            # If there's no number already, create entry
            if num not in file_dict:
                file_dict[num] = []
            # Add file to associated number
            file_dict[num].append(f)

    mega_comp = [
        (os.path.join(PATH, f1), os.path.join(PATH, f2))
        for _, file_list in file_dict.items()  # Grab each item in the pair dict
        if len(file_list) == 2  # Check if there's two files for the number
        for f1, f2 in [file_list]
    ]  # Select the two files and return path

    valid_pairs = []
    for pair in mega_comp:
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


PATH = "/dls/i22/data/2025/nr40718-1/"
HFM_CHANNEL_NO = 12
VFM_CHANNEL_NO = 32
SEQUENCE_LENGTH = 15
DETECTOR_RES = (2464, 2056)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 8
NUM_EPOCHS = 1
BEAMLINE = "i22"
PAIRED_FILES = pair_files(PATH)


class BimorphHDF5Dataset(Dataset):
    def __init__(self):
        self.file_root = PATH
        self.file_list = os.listdir(PATH)
        self.paired_files = PAIRED_FILES

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        image_sequence_out = np.empty(shape=(SEQUENCE_LENGTH, 1, *DETECTOR_RES))
        volt_sequence_out = np.empty(
            shape=(SEQUENCE_LENGTH, VFM_CHANNEL_NO + HFM_CHANNEL_NO)
        )
        nexus, h5 = self.paired_files[idx]
        with h5py.File(nexus, "r") as f:
            for item in range(1, VFM_CHANNEL_NO + 1):
                volt_sequence_out[:, item - 1] = f[
                    f"entry/instrument/bimorph_vfm/channels-{item}-output_voltage"
                ]
            for item in range(1, HFM_CHANNEL_NO + 1):
                volt_sequence_out[:, VFM_CHANNEL_NO + item - 1] = f[
                    f"entry/instrument/bimorph_hfm/channels-{item}-output_voltage"
                ]

        with h5py.File(h5, "r") as f:
            image_sequence_out[:, 0, :, :] = f["entry/data/data"]

        return tensor(image_sequence_out), tensor(volt_sequence_out)


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
# Training
################################

# Training loop.
# print("Training...")
# for epoch in range(NUM_EPOCHS):
#     for image_sequence_out, volt_sequence_out in train_loader:
#         ...
# print("Training... DONE")


# ################################
# # Model
# ################################


# class Bimorph_Focusing(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#         # Extract beam features from the detector with a 2D Convolutional Network.
#         self.image_conv = torch.nn.Sequential(
#             torch.nn.AvgPool2d(2, 2),
#             torch.nn.Conv2d(
#                 in_channels=1, out_channels=16, kernel_size=(5, 5), padding=2
#             ),
#             torch.nn.BatchNorm2d(num_features=16),
#             torch.nn.LeakyReLU(),
#             torch.nn.AvgPool2d(2, 2),
#             #
#             torch.nn.Conv2d(
#                 in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1
#             ),
#             torch.nn.BatchNorm2d(num_features=32),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout2d(p=0.2),
#             torch.nn.AvgPool2d(2, 2),
#             torch.nn.Conv2d(
#                 in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
#             ),
#             torch.nn.BatchNorm2d(num_features=64),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout2d(p=0.2),
#             torch.nn.AvgPool2d(2, 2),
#             #
#             torch.nn.Conv2d(
#                 in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1
#             ),
#             torch.nn.BatchNorm2d(num_features=64),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout2d(p=0.2),
#             torch.nn.AvgPool2d(2, 2),
#             #
#             torch.nn.Conv2d(
#                 in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1
#             ),
#             torch.nn.BatchNorm2d(num_features=64),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout2d(p=0.2),
#             torch.nn.AvgPool2d(2, 2),
#             #
#             torch.nn.Conv2d(
#                 in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1
#             ),
#             torch.nn.BatchNorm2d(num_features=128),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout2d(p=0.2),
#             #
#             torch.nn.Conv2d(
#                 in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1
#             ),
#             torch.nn.BatchNorm2d(num_features=128),
#             torch.nn.LeakyReLU(),
#             torch.nn.Dropout2d(p=0.2),
#             torch.nn.AvgPool2d(2, 2),
#             #
#             # torch.nn.AdaptiveAvgPool2d((1, 1)),
#         )

#         self.attention = torch.nn.MultiheadAttention(
#             embed_dim=8192, num_heads=32, batch_first=False
#         )

#         # self.fully_connected = torch.nn.Linear(hidden_size, 44)
#         self.fully_connected = torch.nn.Linear(8192, 44)

#     def forward(self, images, voltages):

#         atten_out, _ = self.attention(image_features, image_features, image_features)

#         out = self.fully_connected(atten_out[:, -1, :])

#         return out


# # Pre-set weights in all model layers.
# def init_weights(m):
#     if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
#         torch.nn.init.kaiming_normal_(
#             m.weight, mode="fan_in", nonlinearity="leaky_relu"
#         )
#         if m.bias is not None:
#             torch.nn.init.zeros_(m.bias)

# # Define loss, optimiser and run parameters.
# model = Bimorph_Focusing()
# model.apply(init_weights)

# if torch.cuda.is_available():
#     model.to("cuda")

# # Define loss, optimiser and run parameters.
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# data_size = 10
# losses = []
# layers = []
# grads = []


# ################################
# # Training
# ################################


# ################################
# # Testing
# ################################
