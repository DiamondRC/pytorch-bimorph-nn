#######################################################################
# Toy problem - position of a circle
#######################################################################

import os
import random

import torch

os.system("clear")

input = torch.Tensor(
    [random.randint(-100, 100), random.randint(-100, 100), random.randint(-50, 50)]
)
fit = torch.Tensor([31, -41, 17])


class CircleMover(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Instantiate all the parameters of the non-linear function.
        """
        super().__init__(*args, **kwargs)
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))
        self.f = torch.nn.Parameter(torch.randn(()))

    def forward(self, input):
        """
        In the forward function we accept a tensor of input data and return
        a tensor of output data.
        Can use modules defined in the constructor as well as arbitrary operators
        on Tensors.
        """
        input = torch.split(input, 1)

        x = self.a * input[0] + self.b
        y = self.c * input[1] + self.d
        r = self.e * input[2] + self.f

        input = torch.Tensor([x, y, r])
        # return self.a + self.b*x + self.c*y**2 + self.d*z**3
        return input

    def display(self):
        # return f'y = {self.a.item()} + {self.b.item()} x \
        # + {self.c.item()} x^2 + {self.d.item()} x^3 + \
        # {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'
        return f"x = {self.a.item()} + {self.b.item()}, \
            y = {self.c.item()} + {self.d.item()}, \
            z = {self.e.item()} + {self.f.item()}"


learning_rate = 1e-8
# Instantiate our class to create the model.
model = CircleMover()

# Will use Mean Squared Error (MSE) as our loss function.
criterion = torch.nn.MSELoss(reduction="sum")

# Construct the optimiser. Using momentum here.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(3000):
    # Forwards pass - compute y by passing x into the model
    y_pred = model(input)

    # Compute and print loss
    loss = criterion(y_pred, fit)
    if epoch % 100 == 99:
        print(epoch, loss.item())

    # Zero gradients, preform a backwards pass and update the weights.
    loss.requires_grad = True
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

print(f"Target: {fit}, Input: {input}, Params: {model.display()}")
