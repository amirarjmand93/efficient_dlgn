import torch
import torchlogix

# Create a simple logic layer
layer = torchlogix.layers.LogicDense(in_dim=10, out_dim=5)
x = torch.randn(32, 10)
output = layer(x)
print(f"Output shape: {output.shape}")
