#######################################################################
# Predict next image parameters in 2D Gaussian Sequence
#######################################################################

import os

import h5py
import matplotlib.pyplot as plt
import torch
import torch.share
from torch.utils.data import DataLoader, Dataset, random_split

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


NUM_EPOCHS = 100
DATA_SIZE = 10
LEARNING_RATE = 1e-2
TRAINING_DATA_SIZE = 500
BATCH_SIZE = 32
PATH = "gaussian_2d_sequences.hdf5"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
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
