#######################################################################
# Solve Legendre polynomial of degree 3 fitting using sin(x) with Torch
#######################################################################

import math

import torch

# Create input and output tensors

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Here y is a linear function of (x, x**2, x**3), so we can consider it as a
# linear layer neural network.
# Prepare tensor (x, x^2, x^3).
# x.unsqueeze(-1) has shape (2000, 1), and p has shape (3,).
# Thus get tensor of shape (2000,3)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# Use nn package to dfefine model as series of layers.
# nn.Sequential is a module containing modules - it applies them in sequence
# to produce its output.
# Linear module computes output from input using a linear function and holds internal
# tensors for its weights and biases.
# The Flatten layer flatens the output of the linear layer to a 1d tensor to match the
# shape of y.

model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

# The nn package conatins definitiaons of popular loss functions,
# here have Mean Squared Error.

loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-6
for epoch in range(2000):
    # Forwards pass - compute predicted y by passing x to the model.
    # Module object overwrite __call__ so they can be called like functions.
    # Doing so passes a tensor of input data to the module and it
    # produces a tensor of output data.

    y_pred = model(xx)

    # Compute and print loss

    y_diff = y_pred - y
    loss = (y_diff).pow(2).sum()
    if epoch % 100 == 99:
        print(epoch, loss.item())

    # Zero the graidents before running the backwards pass

    model.zero_grad()

    # Backwards pass - compute the gradient of the loss with respect to the
    # learnable parameters of the model.

    loss.backward()

    # Update the weights with gradient descent.
    # Wrap in torch.no_grad() as weights have requires_grad=True but we don't
    # need to track this in autograd.

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# Access first layer of the model lieka accessing a list
linear_layer = model[0]

# For each linear layer, its parameters are stored as weights and biases.
print(
    f"Result: y = {linear_layer.bias.item()} + \
        {linear_layer.weight[:, 0].item()} x + \
        {linear_layer.weight[:, 1].item()} x^2 + \
        {linear_layer.weight[:, 2].item()} x^3"
)
