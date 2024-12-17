import math
import os

import torch

os.system("clear")

#######################################################################
# Solve 3rd order polynomial fitting using sin(x) with Torch
#######################################################################

# Setup data type and GPU acceleration

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Create random inputs and outputs

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random inital weights

a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for epoch in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x**2 + d * x**3

    # Compute the loss

    y_diff = y_pred - y
    loss = (y_diff).pow(2).sum().item()
    if epoch % 100 == 99:
        print(epoch, loss)

    # Compute gradients using the loss

    grad_y_pred = 2.0 * (y_diff)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    # Update weights using gradient descent

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f"Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")
