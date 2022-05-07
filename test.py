import torch
from timesformer.models.vit import TimeSformer

# model = TimeSformer(img_size=224, num_classes=100, num_frames=8, attention_type='divided_space_time',
#                     pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')

# dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

# pred = model(dummy_video,) # (2, 400)

# print(pred)

# print(pred.shape)

# print(torch.cuda.is_available())

import pickle
with open('result/test/pred_label.txt', 'rb') as f:
    data = pickle.load(f)
print(data) # {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}