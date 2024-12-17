#######################################################################
# Solve Legendre polynomial of degree 3-5 fitting sin(x) with Torch
#######################################################################

import math
import random

import torch

# Create input and output tensors
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)


class DynamicNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Instantiate all the parameters of the polynominal.
        """
        super().__init__(*args, **kwargs)
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        For the forwards path of this model, randomly choose either 4
        or 5 and reuse the 'e' parameter to compute the contribution of
        these orders.
        Since each forwards pass builds a dynamic computation graph,
        can use loop and conditional statements. Can also reuse parameters.
        """
        y = self.a + self.b * x + self.c * x**2 + self.d * x**3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x**exp
        return y

    def display(self):
        return f"y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + \
            {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?"


# Instantiate our class to create the model.
model = DynamicNet()

# Will use Mean Squared Error (MSE) as our loss function.
criterion = torch.nn.MSELoss(reduction="sum")

# Construct the optimiser. Using momentum here.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

for epoch in range(30000):
    # Forwards pass - compute y by passing x into the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if epoch % 2000 == 1999:
        print(epoch, loss.item())

    # Zero gradients, preform a backwards pass and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Result: {model.display()}")


input = torch.Tensor(
    [random.randint(-100, 100), random.randint(-100, 100), random.randint(-50, 50)]
)
print(input)
input = torch.split(input, 1)
print(input)
