import torch
from swin_transformer import SwinTransformer,SwinTransformerBlock

input = torch.randn(8, 3, 224, 224)


model = SwinTransformer()

# total = sum([param.nelement() for param in model.parameters()])
# print(total)
out = model(input)
# print(out)
print(out.shape)