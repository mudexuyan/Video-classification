# import torch
# from timesformer.models.vit import TimeSformer,VisionTransformer,MLPMixerBase,MLPMixerModel
# from torch.utils.tensorboard import SummaryWriter
# import pickle

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

print(len(categories1))
print(len(categories2))