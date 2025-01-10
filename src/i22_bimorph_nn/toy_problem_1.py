#######################################################################
# Toy problem 1 - Approximate a guassian function x to return y
#######################################################################

# https://stackoverflow.com/questions/55920015/how-to-realize-a-polynomial-regression-in-pytorch

import os

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Linear, MSELoss
from torch.optim import SGD

os.system("clear")


def gaussian_generator(data_size):
    # Creates a big dataset of x and y values for the given function.
    # y = a*e**(-((x-b)**2)/2*c**2)
    inputs = []
    labels = []

    # Loop data_size times to generate the data
    for _ in range(data_size):
        # Generate x between 0 and 1000
        x = np.random.randint(2000) / 1000

        # Corresponding y value using the function y = 4.2*e**(-((x--0.1)**2)/2*1.3**2)
        y = -3.8 * np.exp(-((x - 0.8) ** 2) / 2 * 2.4**2)

        # Append the values to our input and labels lists
        inputs.append([x])
        labels.append([y])

    return inputs, labels


# define the model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(1, 1)

    def forward(self, x):
        return self.fc1(x)


model = Net()

# define the loss function
critereon = MSELoss()
# define the optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# define the number of epochs and the data set size
nb_epochs = 20000
data_size = 1000

# create our training loop
for epoch in range(nb_epochs):
    X, y = gaussian_generator(data_size)
    X = Variable(Tensor(X))
    y = Variable(Tensor(y))
    epoch_loss = 0
    y_pred = model(X)
    loss = critereon(y_pred, y)
    epoch_loss = loss.data
    if epoch % 1000 == 99:
        print(f"Epoch: {epoch} Loss: {epoch_loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Grab a single piece of test data and pass through the NN.
# Compare the result to the forumla to determine model accuracy.
model.eval()
test_data = gaussian_generator(1)
prediction = model(Variable(Tensor(test_data[0][0])))

print("=" * 50)
print(f"Checking model with x = {test_data[0][0][0]}")

print()
print(f"Expected: {test_data[1][0][0]}")
print(f"Prediction: {prediction.data[0]}")

diff = prediction.data[0] - test_data[1][0][0]
percentage_diff = (abs(diff) / ((prediction.data[0] + test_data[1][0][0]) / 2)) * 100

print()
print(f"Diff is: {diff}, {abs(percentage_diff):0.2f}%")
# Get ~10% error on average.
# Seems inflated by floating point precision.
print("=" * 50)


# x = 0.974: 16.16%
# x = 0.508: 3.91%
# x = 0.9: 12.04%
# x = 1.139: 25.07%
# x = 1.643: 8.45%
# x = 0.446: 5.2%
# x = 0.22: 5.98%
# x = 0.725: 3.44%
# x = 0.434: 5.39%
# x = 0.359: 6.2%
# x = 0.256: 6.27%
# x = 0.003: 0.65%
# x = 1.132: 24.77%
