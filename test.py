import torch
from timesformer.models.vit import TimeSformer,VisionTransformer,MLPMixerBase,MLPMixerModel
from torch.utils.tensorboard import SummaryWriter
import pickle

# model = TimeSformer(img_size=224, num_classes=100, num_frames=16, attention_type='divided_space_time',
#                     pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')
# model = TimeSformer(img_size=224, num_classes=100, num_frames=16, attention_type='divided_space_time')

# dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

# writer = SummaryWriter("result/tensorboard")
# writer.add_graph(model,dummy_video)
# # tensorboard  --port=6006 --logdir result/tensorboard/

# writer.close()

# # pred = model(dummy_video,) # (2, 400)

# total = sum([param.nelement() for param in model.parameters()])
# print(total)

# print(pred)

# print(pred.shape)


# model2 = MLPMixerBase(img_size=224, num_classes=100, num_frames=16)

# dummy_video2 = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

# writer = SummaryWriter("result/tensorboard")

# writer.add_graph(model2,dummy_video2)

# writer.close()

# pic = open(r'result/videoformer/test','rb')
# data = pickle.load(pic)
# print(data)


# pred2 = model2(dummy_video2,) # (2, 400)

# total2 = sum([param.nelement() for param in model2.parameters()])
# print(total2)

# print(pred2)

# print(pred2.shape)

# print(torch.cuda.is_available())



# model = TimeSformer(img_size=224, num_classes=10, num_frames=8, attention_type='divided_space_time',
#                     pretrained_model='result/model/train/checkpoints/checkpoint_epoch_00030.pyth')


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

# x = torch.randn(2,3)
# print(x)
# # 返回每行的最大数
# y = torch.max(x,1)[0]
# print(y)
# # 返回每行最大数的索引
# y1 = torch.max(x,1)[1]
# print(y1)

# pic = open(r'pickle.txt','wb')
# pickle.dump(y1,pic)
# pic.close()

# pic = open(r'pickle.txt','rb')
# data = pickle.load(pic)
# print(data)


