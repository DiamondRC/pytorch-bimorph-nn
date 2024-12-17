#######################################################################
# Solve Legendre polynomial of degree 3 fitting using sin(x) with Torch
#######################################################################

import math

import torch

# Create input and output tensors

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Create the model and its layers.


class Polynomial3(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        In the constructor we instantiate four parameters
        and assign the, as member parameters.
        """
        super().__init__(*args, **kwargs)
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        In the forward function we accept a tensor of input data and return
        a tensor of output data.
        Can use modules defined in the constructor as well as arbitrary operators
        on Tensors.
        """
        return self.a + self.b * x + self.c * x**2 + self.d * x**3

    def string(self):
        """
        Can define custom methods on PyTorch modules.
        """
        return f"y = {self.a.item()} + {self.b.item()} x + \
            {self.c.item()} x^2 + {self.d.item()} x^3"


# Instantiate call to create our model.

model = Polynomial3()

# Will use Mean Squared Error (MSE) as our loss function.

criterion = torch.nn.MSELoss(reduction="sum")

# Construct Optimizer.
# The call to model.parameters() in the Stat. Grad. Des.
# constructor will contain the learnable parameters,
# which are members of the model.

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for epoch in range(2000):
    # Forwards pass - compute predicted y by passing x to the model

    y_pred = model(x)

    # Compute and print loss

    loss = criterion(y_pred, y)
    if epoch % 100 == 99:
        print(epoch, loss.item())

    # Zero gradients and preform a backwards pass, then update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Result: {model.string()}")
