#######################################################################
# Solve Legendre polynomial of degree 3 fitting using sin(x) with Torch
#######################################################################

import math

import torch

# Define our model now as a class.
# Use a custom autograd function for computing forwards and backwards passes.


class LegendrePolynomial3(torch.autograd.Function):
    """
    Implement our own autograd functions bu subclassing
    torch.autograd.Function and implementing the forwards
    and backwards passes which operate on tensors ourselves.
    """

    # staticmethod is liek a plain function except you call them
    # from an instance or the class.
    # Group functions which have some logical connection to the class
    # with the class
    @staticmethod
    def forward(ctx, input):
        """
        Receive a tensor containing the input and a return
        tensor containing the output.
        ctx is a context object used to stash information used
        in backwards computation.
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input**3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Receive a tensor containing the gradient of the loss with
        respect to the output. Need to compute gradient of loss
        with respect to the input.
        """
        # Pull cache from forward()
        (input,) = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input**2 - 1)


# Setup data type and GPU acceleration

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Create input and output tensors

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random tensors for weights. Need 4 weights here and they need to be
# initalised near the correct value for this example to work
# equires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.

a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for epoch in range(2000):
    # To apply to our function, use the Function.apply method - aliased as P3
    P3 = LegendrePolynomial3.apply

    # Forwards pass - compute predicted y using operations
    # P3 computed using custom autograd operation.
    y_pred = a + b * P3(c + d * x)

    # Compute and print loss

    y_diff = y_pred - y
    loss = (y_diff).pow(2).sum()
    if epoch % 100 == 99:
        print(epoch, loss.item())

    # Use autograd to compute backwards pass.
    loss.backward()

    # Manually update weights using gradient descent.
    # Wrap in torch.no_grad() as weights have requires_grad=True but we don't
    # need to track this in autograd.

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

print(f"Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)")
