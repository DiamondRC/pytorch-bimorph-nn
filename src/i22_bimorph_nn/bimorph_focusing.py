#######################################################################
# i22 Bimorph Mirror Focusing - pyTorch Neural Network
#######################################################################

import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import tensor
from torch.autograd import Variable
from torchvision import transforms

os.system("clear")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


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
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            # torch.nn.AvgPool2d(2, 2),
            #
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1
            ),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            # torch.nn.AvgPool2d(2, 2),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flat = torch.nn.Sequential(
            torch.nn.Flatten(),
        )

        hidden_size = 2 * int((2 / 3 * 128) + 44)
        self.sequence = torch.nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.fully_connected = torch.nn.Linear(hidden_size, 44)

    def forward(self, images, voltages):
        batch_size, sequence_length = images.shape[:2]

        # Process images and voltages at each timestep
        image_features = []
        for t in range(batch_size):
            image_batch = images[t, :]
            x = self.flat(self.image_conv(image_batch))

            image_features.append(x)

        image_features = torch.stack(image_features, dim=0)

        LSTM_out, _ = self.sequence(image_features)

        out = self.fully_connected(LSTM_out[:, -1, :])

        return out


# Pre-set weights in all model layers.
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(
            m.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.GRU):
        for name, param in m.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_normal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)


# Define loss, optimiser and run parameters.
model = Bimorph_Focusing()
model.apply(init_weights)

if torch.cuda.is_available():
    model.to("cuda")

# Define loss, optimiser and run parameters.
criterion = torch.nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

data_size = 10
losses = []
layers = []
grads = []


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
                        volt_out, images_out, params_out = get_data_from_run(
                            dir, file_path.name[:-4], detector
                        )

                        # Crop for lower VRAM usage
                        images_out = images_out[:, :, 616:1848, 514:1542]

                        next_images_out = np.float32(
                            np.array(
                                [images_out[i + 3] for i in range(len(images_out) - 3)]
                            )
                        )

                        # Sliding window sequence into batches of three images.
                        images_out = np.float32(
                            np.array(
                                [
                                    images_out[i : i + 3]
                                    for i in range(len(images_out) - 3)
                                ]
                            )
                        )

                        # 'next step' for each image batch.
                        next_volt = np.float32(
                            np.array(
                                [volt_out[i + 3] for i in range(len(volt_out) - 3)]
                            )
                        )

                        # Channel information at each step.
                        volts_out = np.float32(
                            np.array(
                                [volt_out[i : i + 3] for i in range(len(volt_out) - 3)]
                            )
                        )

                        image = Variable(tensor(images_out.copy(), device="cuda"))
                        next_volt = Variable(tensor(next_volt.copy(), device="cuda"))
                        volts_out = Variable(tensor(volts_out.copy(), device="cuda"))

                        # Normalise the images and 'voltages'.
                        norm_img = transforms.Normalize(
                            mean=torch.mean(image), std=torch.std(image)
                        )

                        row_mean = next_volt.mean(dim=1, keepdim=True)
                        row_std = next_volt.std(dim=1, keepdim=True)
                        norm_next_volt = (next_volt - row_mean) / row_std

                        row_mean2 = volts_out.mean(dim=1, keepdim=True)
                        row_std2 = volts_out.std(dim=1, keepdim=True)
                        norm_volts_out = (volts_out - row_mean2) / (row_std2 + 1e-10)

                        norm_images_out = norm_img(image)

                        epoch_loss = 0

                        model_pred = model(norm_images_out, norm_volts_out)

                        # Do something with the model prediction to generate the image
                        ...

                        # Calculate loss, backpropagate etc
                        loss = criterion(model_pred, norm_next_volt)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_loss = loss.data
                        epochs += 1

                        # Collect loss on CPU for plotting.
                        losses.append(loss.cpu().detach().numpy())

                        if epoch % 1 == 0:
                            print(f"Epoch: {epoch} Loss: {epoch_loss}")

                        if epochs == 0 or epochs % 5 == 0:
                            for name, param in model.named_parameters():
                                if "weight" in name:  # Only consider weight parameters
                                    layers.append(name)
                                    # Compute L2 norm (Frobenius norm) of the weights
                                    grad = torch.norm(param, p=2).item()
                                    grads.append(grad)

            except KeyError:
                pass


################################
# Testing
################################

for name, param in model.named_parameters():
    if param.grad is not None:
        plt.figure(figsize=(6, 4))
        plt.hist(
            param.grad.view(-1).cpu().detach().numpy(),
            bins=200,
            alpha=0.7,
            color="blue",
        )
        plt.title(f"Gradient Histogram for {name}")
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"bimorph/{name}.png")
        plt.show()

plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.savefig("bimorph/losses.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(layers, grads, color="skyblue")
plt.title("Weight Magnitudes (L2 Norm) for Each Layer")
plt.xlabel("Layer")
plt.ylabel("Weight Magnitude (L2 Norm)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("bimorph/layer_weight.png")
plt.show()
