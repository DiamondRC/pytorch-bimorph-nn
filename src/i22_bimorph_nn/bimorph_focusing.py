#######################################################################
# i22 Bimorph Mirror Focusing - pyTorch Neural Network
#######################################################################

import os
import random
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
images_per_sequence = 3
detector = "-ss"
detector_res = (2464, 2056)
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
    return (volt_out, image_out, params_out.T)


################################
# Model
################################


class Bimorph_Focusing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(9, 9), stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(7, 7), stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.volt_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=44, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=128),
            torch.nn.ReLU(),
        )

        self.combine = torch.nn.Sequential(
            torch.nn.Linear(in_features=2400128, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=1),
        )

        # Want to encode temporal information of the sequence to give 'next step'
        # self.sequence = torch.nn.LSTM(
        #     input_size=
        #     hidden_size=hidden_size,
        #     num_layers=
        #     batch_first=
        # )

        # Fully connected layer for final prediction
        # self.fc = torch.nn.Linear(hidden_size, next_numbers)

        # # Normalises data for faster convergence
        # self.batch1 = torch.nn.BatchNorm2d(num_features=32)
        # self.batch2 = torch.nn.BatchNorm2d(num_features=64)
        # self.batch3 = torch.nn.BatchNorm2d(num_features=128)
        # self.batch4 = torch.nn.BatchNorm2d(num_features=128)

        # # Improve model flexability and bias
        # self.dropout = torch.nn.Dropout2d(p=0.15)

    def forward(self, image_crop, voltages):
        batch_size, sequence_length = image_crop.shape[:2]
        print(batch_size)
        print(sequence_length)

        # Process images at each timestep
        # image_features = []
        # for t in range(sequence_length):
        #     ...

        # return out


model = Bimorph_Focusing()

# Define loss, optimiser and run parameters.
critereon = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

data_size = 10
losses = []


################################
# Training
################################


count = 0
epoch = 0
for file in os.listdir(dir):
    epoch += 1
    if file.endswith(".nxs"):
        with h5py.File(f"{dir}{file}", "r") as f:
            try:
                if np.shape(f["entry/instrument/beam_device/beam_intensity"]) == (
                    data_set_size,
                ):
                    file_path = Path(file)
                    file_path.name.split(".")[0]
                    file_hash = hash(f)
                    # 80% testing split
                    if file_hash % 5 != 0:
                        print(f"Training set: {file_path.name}")
                        print(f"Training set: {file_path.name[:-4]}-ss.hf5")
                        volt_out, image_out, params_out = get_data_from_run(
                            dir, file_path.name[:-4], detector
                        )

                        optimizer.zero_grad()

                        slice = random.randrange(0, 11)
                        image_crop = image_out[slice : slice + 3, :, :, :]
                        image_next = image_out[slice + 3, :, :, :]
                        volt_crop = volt_out[slice : slice + 3, :]

                        image_crop = Variable(Tensor(image_crop))
                        image_next = Variable(Tensor(image_next))
                        voltages = Variable(Tensor(volt_crop))
                        epoch_loss = 0

                        print(image_crop.size())

                        model_pred = model(image_crop, voltages)
                        # Do something with the model prediction to generate the image
                        ...

                        # Calculate loss, backpropagate etc
                        loss = critereon(image_next, model_pred)

                        loss.backward()
                        optimizer.step()
                        epoch_loss = loss.data

                        losses.append(loss.detach().numpy())

                        if epoch % 1 == 0:
                            print(f"Epoch: {epoch} Loss: {epoch_loss}")

            except KeyError:
                pass


################################
# Execution
################################


# # 10% for val
# elif file_hash % 2 != 0:
#     print(f"Validation set: {file_path.name}")
#     print(f"Validation set: {file_path.name[:-4]}-ss.hf5")
#     # get_data_from_run(dir, file_path.name[:-4], detector)
# # 10% for tst
# else:
#     print(f"Test set: {file_path.name}")
#     print(f"Test set: {file_path.name[:-4]}-ss.hf5")
#     # get_data_from_run(dir, file_path.name[:-4], detector)
