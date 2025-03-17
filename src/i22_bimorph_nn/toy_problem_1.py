#######################################################################
# Toy Problem 1 - Approximate a guassian function
#######################################################################

import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, MSELoss
from torch.utils.data import DataLoader, Dataset

os.system("clear")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


class GaussianDataset(Dataset):
    def __init__(self, data_size, range_min, range_max, mean, std):
        self.x = (range_max - range_min) * torch.rand(data_size) + range_min
        self.y = (1 / math.sqrt(2 * std**2)) * torch.exp(
            -((self.x - mean) ** 2) / 2 * std**2
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx].unsqueeze(0)


# Create the learning model
class Overkill_Function(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(1, 20)
        self.fc2 = Linear(20, 20)
        self.fc3 = Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Define Constants and Hyperparameters
MEAN = 0.5
STD = 0.2
BATCH_SIZE = 10
NUM_EPOCHS = 200
LEARNING_RATE = 1e-2
TRAINING_DATA_SIZE = 500
RANGE_MIN = -10
RANGE_MAX = 20

loss_data = []

# Create model and send to GPU
model = Overkill_Function()
if torch.cuda.is_available():
    model.to("cuda")
critereon = MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Split up the training data into training, validation and testing datasets
training_data = GaussianDataset(TRAINING_DATA_SIZE, RANGE_MIN, RANGE_MAX, MEAN, STD)
train_loader = DataLoader(
    dataset=training_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)

# Model training loop
for epoch in range(NUM_EPOCHS):
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x.cuda())
        loss = critereon(output, y.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    if epoch % 10 == 9:
        print(f"Epoch: {epoch} Loss: {loss.data}")
    loss_data.append(loss.cpu().detach().numpy())

# Prepare model for testing
model.eval()

# Display loss
plt.plot(range(NUM_EPOCHS), loss_data)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()
plt.close()

# Plot model against truth
model_y = []
training_y = []
with torch.no_grad():
    full_range = torch.arange(RANGE_MIN, RANGE_MAX, 0.01).unsqueeze(1)
    plt.scatter(training_data.x.cpu(), training_data.y.cpu(), label="Data")
    plt.plot(
        full_range.cpu().detach().numpy(),
        model(full_range.cuda()).cpu().detach().numpy(),
        color="red",
        label="Model",
    )
    plt.title("Test Model")
    plt.legend(loc="upper right")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
