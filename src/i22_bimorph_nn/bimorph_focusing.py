#######################################################################
# i22 Bimorph Mirror Focusing - pyTorch Neural Network
#######################################################################

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

os.system("clear")


################################
# Data Loading
################################


hfm_channels = 12
vfm_channels = 32
data_set_size = 15
detector = "-ss"
detector_res = (2464, 2056)
# dir = "/scratch/fye77278/bimorph_tmp_data/"
dir = "/dls/i22/data/2025/nr40718-1/"
file = "i22-803415"


def get_data_from_run(dir, file, detector):
    """Extract the channel voltages, corresponding images and values"""
    with h5py.File(f"{dir}{file}.nxs", "r") as f:
        volt_out = np.empty(shape=(data_set_size, vfm_channels + hfm_channels))
        for item in range(1, vfm_channels + 1):
            volt_out[:, item - 1] = f[
                f"entry/instrument/bimorph_vfm/channels-{item}-output_voltage"
            ]
        for item in range(1, hfm_channels + 1):
            volt_out[:, vfm_channels + item - 1] = f[
                f"entry/instrument/bimorph_hfm/channels-{item}-output_voltage"
            ]

    with h5py.File(f"{dir}{file}{detector}.h5", "r") as f:
        image_out = np.empty(shape=(data_set_size, 1, *detector_res))
        params_out = np.empty(shape=(data_set_size, 6))

        image_out[:, 0, :, :] = f["entry/data/data"]
        params_out[:, 0] = f["entry/instrument/NDAttributes/StatsCentroidSigmaX"]
        params_out[:, 1] = f["entry/instrument/NDAttributes/StatsCentroidSigmaY"]
        params_out[:, 2] = f["entry/instrument/NDAttributes/StatsCentroidSigmaXY"]
        params_out[:, 3] = f["entry/instrument/NDAttributes/StatsCentroidX"]
        params_out[:, 4] = f["entry/instrument/NDAttributes/StatsCentroidY"]
        params_out[:, 5] = f["entry/instrument/NDAttributes/StatsSigma"]
    return (volt_out.T, image_out, params_out.T)


################################
# Model
################################


class Bimorph_Focusing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat1 = torch.nn.Flatten(start_dim=0, end_dim=2)
        self.fc1 = torch.nn.Linear(in_features=21, out_features=250)
        self.fc2 = torch.nn.Linear(in_features=250, out_features=250)
        self.fc3 = torch.nn.Linear(in_features=250, out_features=250)
        self.fc4 = torch.nn.Linear(in_features=250, out_features=7)

        self.relu = torch.nn.ReLU()

        # in_channels = C_in, e.g. 1 for greyscale
        self.conv = torch.nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(8, 8), stride=1
        )
        # 2x2 w/ stride 2 halves each dim, 6x6 -> 3x3 etc.
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, image, params):
        x = self.flat1(params)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc4(x)
        return out


model = Bimorph_Focusing()

# Define loss, optimiser and run parameters.
critereon = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

data_size = 10
losses = []


################################
# Training
################################

epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()

    # Pass sequence through the model
    image = Variable(Tensor())
    params = Variable(Tensor())
    epoch_loss = 0

    model_pred = model(image, params)

    # Calculate loss, backpropagate etc
    loss = critereon(image)

    loss.backward()
    optimizer.step()
    epoch_loss = loss.data

    losses.append(loss.detach().numpy())

    if epoch % 1000 == 99:
        print(f"Epoch: {epoch} Loss: {epoch_loss}")

count = 0
for file in os.listdir(dir):
    if file.endswith(".nxs"):
        with h5py.File(f"{dir}{file}", "r") as f:
            try:
                if np.shape(f["entry/instrument/beam_device/beam_intensity"]) == (
                    data_set_size,
                ):
                    count += 1
                    file_path = Path(file)
                    file_path.name.split(".")[0]
                    file_hash = hash(f)
                    # 4/5ths of the dataset for testing
                    if file_hash % 5 != 0:
                        print(f"Training set: {file_path.name}")
                        print(f"Training set: {file_path.name[:-4]}-ss.hf5")
                        # get_data_from_run(dir, file_path.name[:-4], detector)
                    # 10% for val
                    elif file_hash % 2 != 0:
                        print(f"Validation set: {file_path.name}")
                        print(f"Validation set: {file_path.name[:-4]}-ss.hf5")
                        # get_data_from_run(dir, file_path.name[:-4], detector)
                    # 10% for tst
                    else:
                        print(f"Test set: {file_path.name}")
                        print(f"Test set: {file_path.name[:-4]}-ss.hf5")
                        # get_data_from_run(dir, file_path.name[:-4], detector)
            except KeyError:
                pass
print(count)


################################
# Execution
################################
