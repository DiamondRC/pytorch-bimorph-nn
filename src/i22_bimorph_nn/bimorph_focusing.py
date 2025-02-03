#######################################################################
# i22 Bimorph Mirror Focusing - pyTorch Neural Network
#######################################################################

import os
from pathlib import Path

import h5py
import numpy as np

os.system("clear")

# /dls/i22/data/2025/nr40718-1

hfm_channels = 12
vfm_channels = 32
data_set_size = 15
detector_res = (2464, 2056)
dir = "/dls/i22/data/2025/nr40718-1/"
file = "i22-803416"

################################
# Data Loading
################################


def get_data_from_run(file):
    """Extract the channel voltages, corresponding images and values"""
    with h5py.File(f"{dir}{file}.nxs", "r") as f:
        volt_out = np.empty(shape=(vfm_channels + hfm_channels, data_set_size))
        for item in range(1, vfm_channels + 1):
            volt_out[item - 1, :] = f["entry"]["instrument"]["bimorph_vfm"][
                f"channels-{item}-output_voltage"
            ]
        for item in range(1, hfm_channels + 1):
            volt_out[vfm_channels + item - 1, :] = f["entry"]["instrument"][
                "bimorph_hfm"
            ][f"channels-{item}-output_voltage"]

    with h5py.File(f"{dir}{file}-ss.h5", "r") as f:
        image_out = np.empty(shape=(data_set_size, *detector_res))
        image_out[:, :, :] = f["entry"]["data"]["data"]
    return (volt_out.T, image_out)


volt_out, image_out = get_data_from_run(file)


for file in os.listdir(dir):
    if file.endswith(".nxs"):
        with h5py.File(f"{dir}{file}", "r") as f:
            print(file)
            print(f["entry"]["instrument"]["ss"])
            if np.shape(f["entry"]["instrument"]["beam_device"]) == (
                data_set_size,
                *detector_res,
            ):
                continue
        file_path = Path(file)
        file_path.name.split(".")[0]
        file_hash = hash(file)
        if file_hash % 5 != 0:
            print(f"Training set: {file_path.name}")
            # 4/5ths of the dataset for testing
            ...
        elif file_hash % 2 != 0:
            print(f"Validation set: {file_path.name}")
            ...
            # 10% for val
        else:
            print(f"Test set: {file_path.name}")
            ...
            # 10% for tst


################################
# Model
################################

# class Bimoprh_Focusing(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flat1 = torch.nn.Flatten(start_dim=0, end_dim=2)
#         self.fc1 = torch.nn.Linear(in_features=21, out_features=250)
#         self.fc2 = torch.nn.Linear(in_features=250, out_features=250)
#         self.fc3 = torch.nn.Linear(in_features=250, out_features=250)
#         self.fc4 = torch.nn.Linear(in_features=250, out_features=7)

#     def forward(self, image, params):
#         x = self.flat1(params)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         out = self.fc4(x)
#         return out

# model = Bimoprh_Focusing()

# # Define loss, optimiser and run parameters.
# critereon = MSELoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# epochs = 5000
# data_size = 10

# losses = []


################################
# Training
################################


# for epoch in range(epochs):

#     optimizer.zero_grad()


#     # Pass sequence through the model
#     image = Variable(Tensor(images_out))
#     params = Variable(Tensor(params_out))
#     epoch_loss = 0

#     model_pred = model(image, params)

#     # Calculate loss, backpropagate etc
#     loss = critereon()

#     loss.backward()
#     optimizer.step()
#     epoch_loss = loss.data

#     losses.append(loss.detach().numpy())

#     if epoch % 1000 == 99:
#         print(f"Epoch: {epoch} Loss: {epoch_loss}")


################################
# Execution
################################
