# import torch
# from timesformer.models.vit import TimeSformer,VisionTransformer


import torch
from timesformer.models.MLPMixer.mlp_mixer import MLPMixer

model = MLPMixer(
    image_size = 256,
    channels = 3,
    patch_size = 16,
    dim = 512,
    depth = 1,
    num_classes = 1000
)

img = torch.randn(1, 3, 256, 256)
pred = model(img) # (1, 1000)

print(pred.shape)
total = sum([param.nelement() for param in model.parameters()])
print(total)
# model = TimeSformer(img_size=224, num_classes=100, num_frames=8, attention_type='divided_space_time',
#                     pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')

# dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

# pred = model(dummy_video,) # (2, 400)


# print(pred)

# print(pred.shape)

# print(torch.cuda.is_available())

# 预测结果
# import pickle
# with open('result/test/pred_label.txt', 'rb') as f:
#     data = pickle.load(f)
# print(data) 


# model = TimeSformer(img_size=224, num_classes=10, num_frames=8, attention_type='divided_space_time',
#                     pretrained_model='result/model/train/checkpoints/checkpoint_epoch_00030.pyth')

# total = sum([param.nelement() for param in model.parameters()])
# print(total)

# model = TimeSformer(img_size=224, num_classes=10, num_frames=8, attention_type='joint_space_time',
#                     pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')

# total = sum([param.nelement() for param in model.parameters()])
# print(total)

# model = TimeSformer(img_size=224, num_classes=10, num_frames=8, attention_type='space_only',
#                     pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')

# total = sum([param.nelement() for param in model.parameters()])
# print(total)

# import time
# print(time.time())
# timeArray = time.localtime(1651994476)
# otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
# print(otherStyleTime)

# input = torch.randn(2, 3, 8, 224, 224)
# model = VisionTransformer()

# out = model(input)
# print(out)
# print(out.shape)
