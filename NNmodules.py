import torch 
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

# self.w = nn.Parameter(torch.tensor(0.0))  

# def forward(self, x):
#     return self.w * x
    
# nn.Linear(input_features=1, output_features=1)

# nn.Linear(3,1)

layer = nn.Linear(3,2)

x = torch.tensor([[1.0, 2.0, 3.0]])

y = layer(x)

print(y)