import math

import torch

# Define input datatype and GPU acceleration.

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Create tensors to hold input and outputs.
# Defaults to requires_grad=False:
# we don't need to compute gradients for these on backwards pass.

x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
y = torch.sin(x)

# Create random tensors for weights.
# requires_grad=True here, what to compute gradients with respect
# to these tensors on back propagation.

a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for epoch in range(2000):
    # Forwards pass - compute predicted y using operations on tensors.

    y_pred = a + b * x + c * x**2 + d * x**3

    # Compute and print loss using tensors.
    # Loss tensors has shape (1,).
    # loss.item() gets the scalar value held in the loss.

    y_diff = y_pred - y
    loss = (y_diff).pow(2).sum()
    if epoch % 100 == 99:
        print(epoch, loss.item())

    # Use autograd to compute backwards pass.
    # After this, a-through-d.grad will be tensors holding the graident of the loss.

    loss.backward()

    # -----------------------------------
    # Old Loss
    # grad_y_pred = 2.0 * (y_diff)
    # grad_a = grad_y_pred.sum()
    # grad_b = (grad_y_pred * x).sum()
    # grad_c = (grad_y_pred * x**2).sum()
    # grad_d = (grad_y_pred * x**3).sum()
    # -----------------------------------

    # Manually update weights using gradient descent.
    # Wrap in torch.no_grad() as weights have requires_grad=True,
    # but we don't need to track this in autograd.

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero gradients after weights are updated
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")
