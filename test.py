import torch
from timesformer.models.vit import TimeSformer,VisionTransformer,MLPMixerBase,MLPMixerModel,VideoTransformer
from timesformer.models.video_model_builder import SlowFast,ResNet
from timesformer.utils.parser import load_config, parse_args
# from torch.utils.tensorboard import SummaryWriter
# import pickle

# model = TimeSformer(img_size=224, num_classes=100, num_frames=16, attention_type='divided_space_time',
#                     pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')
# model = TimeSformer(img_size=224, num_classes=100, num_frames=16, attention_type='divided_space_time')
# print(model)

# dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

# writer = SummaryWriter("result/tensorboard")
# epoch = 1
# top1 = [12,15,22]
# top5 = [33,44,55]

# writer.add_scalars('1',
#                         {"Val/Top1_err": 22, "Val/Top5_err": 33},
#                         global_step=1,
#                     )
# writer.add_scalars('1',
#                         {"Val/Top1_err": 28, "Val/Top5_err": 44},
#                         global_step=2,
#                     )
# writer.add_graph(model,dummy_video)
# # tensorboard  --port=6006 --logdir result/tensorboard/

# writer.close()

# # pred = model(dummy_video,) # (2, 400)



# print(pred)

# print(pred.shape)


model2 = MLPMixerBase(img_size=224, patch_size=16, in_chans=3, num_classes=94, embed_dim=768, depth=2, num_frames=8)
# model2 = VideoTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=94, embed_dim=768, depth=8, num_frames=8)
# model2 = VideoTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=94, embed_dim=768, depth=8, num_frames=8)


# args = parse_args()
# if args.num_shards > 1:
#     args.output_dir = str(args.job_dir)
# cfg = load_config(args)
# model2 = SlowFast(cfg=cfg) 
# model2 = ResNet(cfg=cfg) 
total = sum([param.nelement() for param in model2.parameters()])
# print(model2)
print(total)
 ## python test.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml


def get_indexing(string):
    """
    Parse numpy-like fancy indexing from a string.
    Args:
        string (str): string represent the indices to take
            a subset of from array. Indices for each dimension
            are separated by `,`; indices for different dimensions
            are separated by `;`.
            e.g.: For a numpy array `arr` of shape (3,3,3), the string "1,2;1,2"
            means taking the sub-array `arr[[1,2], [1,2]]
    Returns:
        final_indexing (tuple): the parsed indexing.
    """
    index_ls = string.strip().split(";")
    final_indexing = []
    for index in index_ls:
        index_single_dim = index.split(",")
        index_single_dim = [int(i) for i in index_single_dim]
        final_indexing.append(index_single_dim)

    return tuple(final_indexing)
layer_name_prefix = "prefix-"
# layer_ls = ['s5/pathway1_res2', 's5/pathway0_res2']
layer_ls = ["layer1 1,2;1,2", "layer2", "layer3 150,151;3,4"]
layer_name, indexing_dict = [], {}
for layer in layer_ls:
        ls = layer.split()
        name = layer_name_prefix + ls[0]
        layer_name.append(name)
        if len(ls) == 2:
            indexing_dict[name] = get_indexing(ls[1])
        else:
            indexing_dict[name] = ()
# print(layer_name)

# print(indexing_dict)

 ## python test.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml

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
# x = torch.ones(2,3,4)
# y = torch.ones(1,3,4)
# print(x+y)


# x = torch.randn(4, 5)
# print(x)
# a,b = torch.topk(x,2)
# print(a)
# print(a.shape)
# print(b)
# print(b.shape)



categories1 = ['SalsaSpin', 'BlowingCandles', 'BlowDryHair', 'Skiing', 'Bowling', 'BrushingTeeth', 'HandstandWalking', 'JumpRope', 'ThrowDiscus', 'PullUps', 'brush_hair', 'PommelHorse', 'JumpingJack', 'Diving', 'PlayingPiano', 'stand', 'StillRings', 'shoot_gun', 'sit', 'BreastStroke', 'Swing', 'LongJump', 'SoccerJuggling', 
'Hammering', 'push', 'wave', 'YoYo', 'VolleyballSpiking', 'PlayingFlute', 'Billiards', 'pullup', 'Kayaking', 'Skijet', 'WallPushups', 'Shotput', 'Haircut', 'bow', 'PoleVault', 'UnevenBars', 'TaiChi', 'HighJump', 'Punch', 'phone', 'Rafting', 'Surfing', 'MoppingFloor', 'ShavingBeard', 'PlayingSitar', 'BoxingPunchingBag', 'pick', 'jump', 'run', 'HulaHoop', 'CliffDiving', 'FrontCrawl', 'PlayingGuitar', 'pour', 'TennisSwing', 'Biking', 'PlayingDhol', 'Fencing', 'swing_baseball', 
'talk', 'ParallelBars', 'BasketballDunk', 'WalkingWithDog', 'ApplyEyeMakeup', 'Rowing', 'Typing', 'shoot_bow', 'drink', 'SkyDiving', 'CuttingInKitchen', 'SkateBoarding', 'straight', 'TrampolineJumping', 'ApplyLipstick', 'clap', 'SumoWrestling', 'BenchPress', 'climb_stairs', 'catch', 'PlayingViolin', 'SoccerPenalty', 'WritingOnBoard', 'Basketball', 'PlayingTabla', 'BabyCrawling', 'TableTennisShot', 'throw', 'walk', 'HorseRiding', 'Drumming', 'PushUps']

categories2 = ['跳舞旋转', '吹蜡烛', '吹干头发', '滑雪', '打保龄球', '刷牙', '倒立走', '跳绳', '掷铁饼', '引体向上 ','刷毛','鞍马','开合跳','跳水','弹钢琴','站立','静止环','射击枪','坐下','蛙泳','摆动', '跳远', '足球杂耍',
  '锤击','推','招手','溜溜球','排球扣球','吹长笛','台球','引体向上','皮划艇','游艇游行','墙上俯卧撑', '铅球','剪发','射弓箭','撑杆跳高','高低杠','太极','跳高','拳击','开车打电话','漂流','冲浪','拖地', '刮胡子', '弹西塔琴', '打沙袋', '弯腰捡东西', '跳跃', '跑步', '转呼啦圈', '悬崖跳水', '爬泳', '弹吉他', '倒水', '网球挥拍', '骑自行车', '打击双面鼓', '击剑', '棒球挥杆'
  , '开车聊天', '双杠运动', '篮球扣篮运动', '遛狗', '化妆', '赛艇运动', '键盘打字', '射箭', '开车喝东西', '跳伞', '厨房做饭', '滑板', '开车', '蹦床跳', '涂眼妆', '拍手 ', '相扑运动', '举重运动', '爬楼梯', '双手接东西', '弹小提琴', '足球点球', '板书', '打篮球', '打单面鼓', '婴儿爬行', '乒乓球','投掷','步行','骑马','击鼓','俯卧撑']

# print(len(categories1))
# print(len(categories2))
# import matplotlib.pyplot as plt
# import numpy as np

# x = ['SalsaSpin', 'BlowingCandles', 'BlowDryHair', 'Skiing', 'Bowling', 'BrushingTeeth', 'HandstandWalking', 'JumpRope', 'ThrowDiscus', 'PullUps', 'brush_hair', 'PommelHorse', 'JumpingJack', 'Diving', 'PlayingPiano', 'stand', 'StillRings', 'shoot_gun', 'sit', 'BreastStroke', 'Swing', 'LongJump', 'SoccerJuggling', 'Hammering', 'push', 'wave', 'YoYo', 'VolleyballSpiking', 'PlayingFlute', 'Billiards', 'pullup', 'Kayaking', 'Skijet', 'WallPushups', 'Shotput', 'Haircut', 'bow', 'PoleVault', 'UnevenBars', 'TaiChi', 'HighJump', 'Punch', 'phone', 'Rafting', 'Surfing', 'MoppingFloor', 'ShavingBeard', 'PlayingSitar', 'BoxingPunchingBag', 'pick', 'jump', 'run', 'HulaHoop', 'CliffDiving', 'FrontCrawl', 'PlayingGuitar', 'pour', 'TennisSwing', 'Biking', 'PlayingDhol', 'Fencing', 'swing_baseball', 'talk', 'ParallelBars', 'BasketballDunk', 'WalkingWithDog', 'ApplyEyeMakeup', 'Rowing', 'Typing', 'shoot_bow', 'drink', 'SkyDiving', 'CuttingInKitchen', 'SkateBoarding', 'straight', 'TrampolineJumping', 'ApplyLipstick', 'clap', 'SumoWrestling', 'BenchPress', 'climb_stairs', 'catch', 'PlayingViolin', 'SoccerPenalty', 'WritingOnBoard', 'Basketball', 'PlayingTabla', 'BabyCrawling', 'TableTennisShot', 'throw', 'walk', 'HorseRiding', 'Drumming', 'PushUps']
# y = [133, 109, 131, 135, 155, 131, 111, 144, 130, 100, 41, 123, 123, 150, 105, 36, 112, 55, 39, 101, 131, 131, 147, 140, 42, 42, 128, 116, 155, 150, 55, 141, 100, 130, 144, 130, 232, 149, 104, 100, 123, 160, 221, 111, 126, 110, 161, 157, 163, 40, 39, 40, 125, 138, 137, 160, 55, 166, 134, 164, 111, 54, 230, 114, 131, 123, 145, 137, 136, 53, 213, 110, 110, 120, 215, 119, 114, 44, 116, 160, 40, 48, 100, 137, 152, 134, 111, 132, 140, 46, 41, 164, 161, 102]
# z = z = np.random.rand(94)
# plt.scatter(x, y, s=z*1000, alpha=0.5)
# plt.show()
# plt.savefig("image.png")

# d = {}
# for i in range(len(categories1)):
#   d[categories1[i]]=categories2[i]
# print(d)