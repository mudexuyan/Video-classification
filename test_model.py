import torch
import torch.nn as nn

m = nn.GELU()
input = torch.randn(4,3,2)
output = m(input)

print("input: ", input)   # input:  tensor([-1.2732, -0.4936, -0.8219,  0.1772])
print("output: ", output) # output:  tensor([-0.1292, -0.1534, -0.1690,  0.1010])
print(output.shape)
