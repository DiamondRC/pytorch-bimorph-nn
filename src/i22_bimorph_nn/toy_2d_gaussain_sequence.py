#######################################################################
# Predict next image parameters in 2D Gaussian Sequence
#######################################################################

import os
import random

import h5py
import matplotlib.pyplot as plt
import torch
import torch.share
from torch.utils.data import DataLoader, Dataset

os.system("clear")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


class GaussianHDF5Dataset(Dataset):
    def __init__(self):
        self.file_path = PATH
        with h5py.File(self.file_path, "r") as f:
            self.data_len = len(f["image_dataset"])

    def __len__(self):
        return len(self.data_len)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as f:
            image_sequence = f["gaussian_seq"][idx]
            voltage_sequence = f["voltage_seq"][idx]
        return image_sequence, voltage_sequence


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
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            #
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1
            ),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            #
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1
            ),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            # Collape to number of features by discarding pixel leftovers.
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        hidden_size = int((2 / 3 * 256) + 44)
        # self.sequence = torch.nn.LSTM(
        #     # Learn temporal component of values.
        #     input_size=256,
        #     hidden_size=hidden_size,
        #     num_layers=3,
        #     batch_first=True,
        #     dropout=0.3,
        #     bidirectional=False,
        # )

        self.sequence = torch.nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.LSTM_out_norm = torch.nn.LayerNorm(hidden_size)

        self.flat = torch.nn.Sequential(
            torch.nn.Flatten(),
        )

        self.pos = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 2),
            torch.nn.LeakyReLU(),
        )
        self.sigmax = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.LeakyReLU(),
        )
        self.sigmay = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.LeakyReLU(),
        )
        self.A = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.LeakyReLU(),
        )
        self.THETA = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.LeakyReLU(),
        )

    def forward(self, image, volts_out):
        batch_size, sequence_length = image.shape[:2]
        features = []

        # Process each batch of images from the input.
        for t in range(batch_size):
            image_batch = image[t, :]
            features.append(self.flat(self.image_conv(image_batch)))
        image_features = torch.stack(features, dim=0)
        # Process features through LSTM
        LSTM_out = self.LSTM_out_norm(self.sequence(image_features)[0])

        # Determine params with seperate linear transforms.
        position = self.pos(LSTM_out[:, -1])
        sigmax = self.sigmax(LSTM_out[:, -1])
        sigmay = self.sigmay(LSTM_out[:, -1])
        amplitude = self.A(LSTM_out[:, -1])
        THETA = self.THETA(LSTM_out[:, -1])

        out = torch.cat((position, sigmax, sigmay, amplitude, THETA), dim=-1)

        return out


# Pre-set weights in all model layers.
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(
            m.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.LSTM):
        for name, param in m.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_normal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)


# Generate consistent seeds
def generate_seed():
    X_0 = random.uniform(-80, 80)
    Y_0 = random.uniform(-80, 80)
    SIGMA_X = random.uniform(8, 12)
    SIGMA_Y = SIGMA_X
    A = random.uniform(17, 19)
    THETA = random.uniform(20, 160)
    DATA_SIZE = 10
    return X_0, Y_0, SIGMA_X, SIGMA_Y, A, THETA, DATA_SIZE


NUM_EPOCHS = 100
DATA_SIZE = 10
LEARNING_RATE = 1e-2
TRAINING_DATA_SIZE = 500
BATCH_SIZE = 32
PATH = "gaussian_2d_sequences.hdf5"
loss_data = []
layers = []
grads = []

model = Focusing_Sequence()
model.apply(init_weights)

if torch.cuda.is_available():
    model.to("cuda")

# Define loss, optimiser and run parameters.
criterion = torch.nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Split up the training data into training, validation and testing datasets
training_data = GaussianHDF5Dataset(TRAINING_DATA_SIZE)
train_loader = DataLoader(
    dataset=training_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)


################################
# Training
################################

for epoch in range(NUM_EPOCHS):
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x.cuda())
        loss = criterion(output, y.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    if epoch % 10 == 0:
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

model.eval()

# Display loss
plt.plot(range(NUM_EPOCHS), loss_data)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
plt.close()

# for img_name in range(5):
#     # Generate focusing sequence
#     X_0, Y_0, SIGMA_X, SIGMA_Y, A, THETA, DATA_SIZE = generate_seed()

#     images_out, next_images_out, next_volt, volts_out = generate_gaussian2(
#         X_0, Y_0, SIGMA_X, SIGMA_Y, A, THETA, DATA_SIZE
#     )

#     # Normalise inputs.
#     image = Variable(tensor(images_out.copy(), device="cuda"))
#     volts_out = Variable(tensor(volts_out.copy(), device="cuda"))

#     norm_img = transforms.Normalize(mean=torch.mean(image), std=torch.std(image))

#     row_mean2 = volts_out.mean(dim=1, keepdim=True)
#     row_std2 = volts_out.std(dim=1, keepdim=True)
#     norm_volts_out = (volts_out - row_mean2) / (row_std2 + 1e-10)

#     norm_images_out = norm_img(image)

#     # Model prediction
#     prediction = model(norm_images_out, norm_volts_out)

#     # Denormalise
#     prediction = prediction * row_std + row_mean

#     # Debug out
#     print("=" * 20)
#     print(f"X_0: {prediction[:, 0].cpu().detach().numpy()}")
#     print(f"X_0_real: {next_volt[:, 0]}")
#     print()
#     print(f"Y_0: {prediction[:, 1].cpu().detach().numpy()}")
#     print(f"Y_0_real: {next_volt[:, 1]}")
#     print()
#     print(f"SIGMA_X: {prediction[:, 2].cpu().detach().numpy()}")
#     print(f"SIGMA_X_real: {next_volt[:, 2]}")
#     print()
#     print(f"SIGMA_Y: {prediction[:, 3].cpu().detach().numpy()}")
#     print(f"SIGMA_Y_real: {next_volt[:, 3]}")
#     print()
#     print(f"A: {prediction[:, 4].cpu().detach().numpy()}")
#     print(f"A_real: {next_volt[:, 4]}")
#     print()
#     print(f"THETA: {prediction[:, 5].cpu().detach().numpy()}")
#     print(f"THETA_real: {next_volt[:, 5]}")

#     # Copy/grid for plotting
#     prediction_copy = prediction.cpu().detach().numpy().copy()
#     x, y = np.meshgrid(np.arange(-128, 128, 1), np.arange(-128, 128, 1))

#     # Compare model prediction to data
#     for j in range(7):
#         plt.subplot(3, 7, j + 1)
#         plt.imshow(next_images_out[j, 0], cmap="hot", interpolation="nearest")

#         model_images_out = elliptical_gaussian(x, y, *prediction_copy[j])
#         plt.subplot(3, 7, j + 8)
#         plt.imshow(
#             model_images_out,
#             cmap="hot",
#             interpolation="nearest",
#             vmin=np.min(next_images_out[j]),
#             vmax=np.max(next_images_out[j]),
#         )
#         # ^Switch to this for a like-to-like comparison of model and actual.
#         # plt.imshow(model_images_out, cmap="hot", interpolation="nearest")

#         plt.subplot(3, 7, j + 15)
#         plt.imshow(
#             next_images_out[j, 0] - model_images_out,
#             cmap="hot",
#             interpolation="nearest",
#         )
#     plt.savefig(f"imgs/{img_name}.png")
#     plt.show()

# # Plot gradient values for debugging.
# for name, param in model.named_parameters():
#     if param.grad is not None:
#         plt.figure(figsize=(6, 4))
#         plt.hist(
#             param.grad.cpu().view(-1).detach().numpy(),
#             bins=500,
#             alpha=0.7,
#             color="blue",
#         )
#         plt.title(f"Gradient Histogram for {name}")
#         plt.xlabel("Gradient Value")
#         plt.ylabel("Frequency")
#         plt.grid(True)
#         # plt.savefig(f"imgs/{name}.png")
#         # plt.close()
#         plt.show()
