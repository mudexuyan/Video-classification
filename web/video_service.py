# -*- coding: utf-8 -*-
import web              #web.py
import os
import torch
from timesformer.datasets import utils as utils
from timesformer.datasets import video_container as container
from timesformer.datasets import decoder as decoder
import av
from timesformer.models.vit import  MLPTest,TimeSformer
import torch.nn.functional as F


# categories = ['SalsaSpin', 'BlowingCandles', 'BlowDryHair', 'Skiing', 'Bowling', 'BrushingTeeth', 'HandstandWalking', 'JumpRope', 'ThrowDiscus', 'PullUps', 'brush_hair', 'PommelHorse', 'JumpingJack', 'Diving', 'PlayingPiano', 'stand', 'StillRings', 'shoot_gun', 'sit', 'BreastStroke', 'Swing', 'LongJump', 'SoccerJuggling', 
# 'Hammering', 'push', 'wave', 'YoYo', 'VolleyballSpiking', 'PlayingFlute', 'Billiards', 'pullup', 'Kayaking', 'Skijet', 'WallPushups', 'Shotput', 'Haircut', 'bow', 'PoleVault', 'UnevenBars', 'TaiChi', 'HighJump', 'Punch', 'phone', 'Rafting', 'Surfing', 'MoppingFloor', 'ShavingBeard', 'PlayingSitar', 'BoxingPunchingBag', 'pick', 'jump', 'run', 'HulaHoop', 'CliffDiving', 'FrontCrawl', 'PlayingGuitar', 'pour', 'TennisSwing', 'Biking', 'PlayingDhol', 'Fencing', 'swing_baseball', 
# 'talk', 'ParallelBars', 'BasketballDunk', 'WalkingWithDog', 'ApplyEyeMakeup', 'Rowing', 'Typing', 'shoot_bow', 'drink', 'SkyDiving', 'CuttingInKitchen', 'SkateBoarding', 'straight', 'TrampolineJumping', 'ApplyLipstick', 'clap', 'SumoWrestling', 'BenchPress', 'climb_stairs', 'catch', 'PlayingViolin', 'SoccerPenalty', 'WritingOnBoard', 'Basketball', 'PlayingTabla', 'BabyCrawling', 'TableTennisShot', 'throw', 'walk', 'HorseRiding', 'Drumming', 'PushUps']

categories = ['跳舞旋转', '吹蜡烛', '吹干头发', '滑雪', '打保龄球', '刷牙', '倒立走', '跳绳', '掷铁饼', '引体向上 ','刷毛','鞍马','开合跳','跳水','弹钢琴','站立','静止环','开枪设计','坐下','蛙泳','摆动', '跳远', '足球杂耍',
  '锤击','推东西','招手','溜溜球','排球扣球','吹长笛','台球','引体向上','皮划艇','游艇游行','墙上俯卧撑', '铅球','剪发','射弓箭','撑杆跳高','高低杠','太极','跳高','拳击','开车打电话','漂流','冲浪','拖地', '刮胡子', '弹西塔琴', '打沙袋', '弯腰捡东西', '跳跃', '跑步', '转呼啦圈', '悬崖跳水', '爬泳', '弹吉他', '倒水', '网球挥拍', '骑自行车', '打击双面鼓', '击剑', '棒球挥杆'
  , '开车聊天', '双杠运动', '篮球扣篮运动', '遛狗', '化妆', '赛艇运动', '键盘打字', '射箭', '开车喝东西', '跳伞', '厨房做饭', '滑板', '开车', '蹦床跳', '涂眼妆', '拍手 ', '相扑运动', '举重运动', '爬楼梯', '双手接东西', '弹小提琴', '足球点球', '板书', '打篮球', '打单面鼓', '婴儿爬行', '乒乓球','投掷','步行','骑马','击鼓','俯卧撑']

urls = (
        '/model.*' , 'model', 
        '/category.*','category',
        )
 

class MyApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

class category:
    # 响应GET请求（声明函数）
    def GET(self):
        web.header("Content-Type","application/json;charset=UTF-8")
        return categories
 
class model:
    def __init__(self):
        self.return_msg = {'errorCode': 0, 'msg': '系统正常！'}     
 
    def POST(self):                    #POST处理方式与GET一致
    
        # web.header("Content-Type","multipart/form-data; charset=gbk")
        # content  = web.input()
        # print('content keys: ', content.keys())
        # path = './web/test.mp4'
        # fout = open(path, 'wb')
        # fout.write(content.video)
        # fout.close()

        # print(content.name)
        # content  = web.input()
        # print('content keys: ', content.keys())
        # print('收到消息：', content.imagename.decode(), content.imagetype.decode(), content.key1.decode())
        # fout = open('test_recv.mp4', 'wb')
        # fout.write(content.image)
        # fout.close()

        content  = web.input()
        print('content keys: ', content.keys())
        path = content.video
        # path = "http://rri7sgufa.hd-bkt.clouddn.com/video/test1.mp4"
        # path = "../TestData/talk_DCX_01.mp4"

        container = av.open(path,'r')

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
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
        )

        # print(frames)
        # print(frames.shape)

        if os.path.exists(path):
            os.remove(path)
       
        x = frames.unsqueeze(0)
        # model = MLPTest(pretrained_model='result/mlp_4_depth4_mix4/checkpoints/checkpoint_epoch_00080.pyth')
        # model = TimeSformer(img_size=224, num_classes=94, num_frames=16, attention_type='divided_space_time',
        #                     pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')
        model = TimeSformer(img_size=224, num_classes=94, num_frames=16, attention_type='divided_space_time',
                            pretrained_model='result/timesformer/checkpoints/checkpoint_epoch_00020.pyth')

        y = F.softmax(model(x,),1).squeeze(0)

        prob,index = torch.topk(y,5)
        prob = prob.tolist()
        index = index.tolist()
        res = []
        for i in range(0,len(index)):
            dict = {}
            dict['topK']='top-'+str(i+1)
            dict['index']=index[i]+1
            dict['category']=categories[index[i]]
            dict['prob']=round(prob[i],4)
            res.append(dict)
        print(res)
        return res

 
if __name__ == "__main__":
    app = MyApplication(urls ,globals())
    app.run(port=1111)