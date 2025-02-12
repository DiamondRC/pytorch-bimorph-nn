#######################################################################
# i22 Bimorph Mirror Focusing - pyTorch Neural Network
#######################################################################

import os
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
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

        # Extract beam features from the detector with a 2D Convolutional Network.
        self.image_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(5, 5), stride=2, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 74 * 62, 128),
            torch.nn.ReLU(),
        )

        # Linearly transform the channel voltages.
        self.volt_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=44, out_features=128),
            torch.nn.ReLU(),
        )

        # Want to encode temporal information of the sequence to give 'next step'.
        self.sequence = torch.nn.LSTM(
            input_size=256,
            # Using 2/3*input layer + output layer as a baseline,
            # this hyperparameter should be optimised though!
            hidden_size=int((2 / 3 * 256) + 44),
            num_layers=1,
            batch_first=True,
        )

        # Fully connected layer for final prediction
        self.fully_connected = torch.nn.Linear(int((2 / 3 * 256) + 44), 44)

        # # Normalises data for faster convergence
        # self.batch1 = torch.nn.BatchNorm2d(num_features=32)
        # self.batch2 = torch.nn.BatchNorm2d(num_features=64)
        # self.batch3 = torch.nn.BatchNorm2d(num_features=128)
        # self.batch4 = torch.nn.BatchNorm2d(num_features=128)

        # # Improve model flexability and bias
        # self.dropout = torch.nn.Dropout2d(p=0.15)

    def forward(self, images, voltages):
        batch_size, sequence_length = images.shape[:2]

        # Process images and voltages at each timestep
        image_features = []
        volt_features = []
        for t in range(sequence_length):
            image_batch = images[:, t]
            volt_batch = voltages[:, t]

            image_features.append(self.image_conv(image_batch))
            volt_features.append(self.volt_linear(volt_batch))

        image_features = torch.stack(image_features, dim=1)
        volt_features = torch.stack(volt_features, dim=1)

        # print(image_features.size())
        # print(volt_features.size())

        combined_features = torch.cat((image_features, volt_features), dim=-1)
        # print(combined_features.size())

        LSTM_out, _ = self.sequence(combined_features)
        # print(LSTM_out.size())

        out = self.fully_connected(LSTM_out[:, -1])
        # print(out.size())

        # plt.imshow(
        #     images.detach().numpy()[0,0,0],
        #     cmap="hot",
        #     interpolation="nearest",
        # )
        # plt.show()

        # plt.imshow(
        #     image_features.detach().numpy()[0,0,0],
        #     cmap="hot",
        #     interpolation="nearest",
        # )
        # plt.show()

        return out


model = Bimorph_Focusing()

# Define loss, optimiser and run parameters.
critereon = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

data_size = 10
losses = []


################################
# Training
################################

count = 0
epoch = 0
epochs = 0
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
                        # image_crop = image_out[slice:slice + 3, :, :, :]

                        img_test = np.array(
                            [image_out[i : i + 3] for i in range(len(image_out) - 3)]
                        )
                        volt_test = np.array(
                            [volt_out[i : i + 3] for i in range(len(volt_out) - 3)]
                        )
                        img_next = np.array(
                            [image_out[i + 3] for i in range(len(image_out) - 3)]
                        )
                        volt_next = np.array(
                            [volt_out[i + 3] for i in range(len(volt_out) - 3)]
                        )
                        # print(np.shape(img_test))
                        # print(np.shape(volt_test))
                        # print(np.shape(img_next))
                        # print(np.shape(volt_next))
                        # print(volt_test[1][-1] == volt_next[0])
                        # print(img_test[1][-1] == img_next[0])

                        image_crop = np.reshape(
                            image_out,
                            (
                                data_set_size // images_per_sequence,
                                images_per_sequence,
                                1,
                                *detector_res,
                            ),
                        )
                        image_next = image_out[slice + 3, :, :, :]
                        # volt_crop = volt_out[slice : slice + 3, :]
                        volt_crop = np.reshape(
                            volt_out,
                            (
                                data_set_size // images_per_sequence,
                                images_per_sequence,
                                vfm_channels + hfm_channels,
                            ),
                        )

                        # plt.imshow(
                        #     image_out[11,0],
                        #     cmap="hot",
                        #     interpolation="nearest",
                        # )
                        # plt.show()

                        # plt.imshow(
                        #     image_crop[3,2,0],
                        #     cmap="hot",
                        #     interpolation="nearest",
                        # )
                        # plt.show()

                        voltages = Variable(Tensor(volt_test))
                        images = Variable(Tensor(img_test))
                        next_channels = Variable(Tensor(volt_next))
                        epoch_loss = 0

                        # print(images.size())

                        model_pred = model(images, voltages)
                        # Do something with the model prediction to generate the image
                        ...

                        # print(f"model_pred: {model_pred[1]}")
                        # print(f"Actual: {next_channels[1]}")
                        print(f"Diff: {model_pred[1] - next_channels[1]}")

                        # Calculate loss, backpropagate etc
                        loss = critereon(model_pred, next_channels)

                        loss.backward()
                        optimizer.step()
                        epoch_loss = loss.data

                        epochs += 1
                        losses.append(loss.detach().numpy())

                        if epoch % 1 == 0:
                            print(f"Epoch: {epoch} Loss: {epoch_loss}")

            except KeyError:
                pass


################################
# Testing
################################

plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.show()
