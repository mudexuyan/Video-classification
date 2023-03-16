from timesformer.datasets import utils as utils
from timesformer.datasets import video_container as container
from timesformer.datasets import decoder as decoder
import av
import torch
from timesformer.models.vit import  MLPTest,TimeSformer

import torch.nn.functional as F

category = ['SalsaSpin', 'BlowingCandles', 'BlowDryHair', 'Skiing', 'Bowling', 'BrushingTeeth', 'HandstandWalking', 'JumpRope', 'ThrowDiscus', 'PullUps', 'brush_hair', 'PommelHorse', 'JumpingJack', 'Diving', 'PlayingPiano', 'stand', 'StillRings', 'shoot_gun', 'sit', 'BreastStroke', 'Swing', 'LongJump', 'SoccerJuggling', 'Hammering', 'push', 'wave', 'YoYo', 'VolleyballSpiking', 'PlayingFlute', 'Billiards', 'pullup', 'Kayaking', 'Skijet', 'WallPushups', 'Shotput', 'Haircut', 'bow', 'PoleVault', 'UnevenBars', 'TaiChi', 'HighJump', 'Punch', 'phone', 'Rafting', 'Surfing', 'MoppingFloor', 'ShavingBeard', 'PlayingSitar', 'BoxingPunchingBag', 'pick', 'jump', 'run', 'HulaHoop', 'CliffDiving', 'FrontCrawl', 'PlayingGuitar', 'pour', 'TennisSwing', 'Biking', 'PlayingDhol', 'Fencing', 'swing_baseball', 'talk', 'ParallelBars', 'BasketballDunk', 'WalkingWithDog', 'ApplyEyeMakeup', 'Rowing', 'Typing', 'shoot_bow', 'drink', 'SkyDiving', 'CuttingInKitchen', 'SkateBoarding', 'straight', 'TrampolineJumping', 'ApplyLipstick', 'clap', 'SumoWrestling', 'BenchPress', 'climb_stairs', 'catch', 'PlayingViolin', 'SoccerPenalty', 'WritingOnBoard', 'Basketball', 'PlayingTabla', 'BabyCrawling', 'TableTennisShot', 'throw', 'walk', 'HorseRiding', 'Drumming', 'PushUps']



# path = "../TestData/talk_DCX_01.mp4"
path = "http://rri7sgufa.hd-bkt.clouddn.com/video/test1.mp4"

container = av.open(path)

temporal_sample_index = 0 # -1随机采样，其它均匀采样，0表示从第0帧开始，2表示从第2帧开始。最大30
NUM_ENSEMBLE_VIEWS = 10
min_scale = 256
max_scale = 320
crop_size = 224
sampling_rate = 8

MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

# The std value of the video raw pixels across the R G B channels.
STD = [0.225, 0.225, 0.225]

frames = decoder.decode(
                container,
                sampling_rate,
                8,
                temporal_sample_index,
                NUM_ENSEMBLE_VIEWS,
                None,
                target_fps=30,
                backend="pyav",
                max_spatial_scale=min_scale,
            )

# Perform color normalization.
frames = utils.tensor_normalize(frames, MEAN, STD)
            
# T H W C -> C T H W.
frames = frames.permute(3, 0, 1, 2)

# Perform data augmentation.

frames = utils.spatial_sampling(
    frames,
    spatial_idx=0,
    min_scale=min_scale,
    max_scale=max_scale,
    crop_size=crop_size,
    random_horizontal_flip=False,
    inverse_uniform_sampling=False,
)

# print(frames)
# print(frames.shape)

x = frames.unsqueeze(0)
# model = MLPTest(pretrained_model='result/mlp_4_depth4_mix4/checkpoints/checkpoint_epoch_00080.pyth')
model = TimeSformer(img_size=224, num_classes=94, num_frames=16, attention_type='divided_space_time',
                    pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')

y = F.softmax(model(x,),1).squeeze(0)

prob,index = torch.topk(y,5)
prob = prob.tolist()
index = index.tolist()

# c = list(map(lambda i:category[i],index))
print(index)
res = []
for i in range(0,len(index)):
    dict = {}
    dict['topK']='top-'+str(i+1)
    dict['index']=index[i]
    dict['category']=category[index[i]]
    dict['prob']=prob[i]
    res.append(dict)
print(res)

