import os
import random

random.seed(0)

index = 0
# xml文件中tensorboard class_name
class_name = {}
parent_child = {}
subset = []



# filePath = "../video_dataset/"  # 文件夹目录
filePath = "../DataSet/"  # 文件夹目录

# 返回path下所有文件构成的一个list列表
fileList = os.listdir(filePath)

trainList = []
testList = []
valList = []

# for file in fileList:
#     # 是文件则略过
#     if os.path.isfile(filePath+file):
#         continue

#     # 输出指定后缀类型的文件
#     tmp = []
#     subFileList = os.listdir(filePath + file)
#     if (file == '驾驶手势数据'):
#         continue
#     for subFile in subFileList:
#         tmp.append(filePath + file + '/' + subFile + ' ' + str(index))
#     # "\"straight\"": 0,
#     # class_name["\\"+file+"\\"]=index
#     # t = []
#     # t.append(file)
#     # parent_child[file]=t
#     # subset.append(file)
#     index = index + 1

#     testTemp = random.sample(tmp, int(0.3*len(tmp)))
#     # train_val = [x for x in tmp if x not in testTemp]
#     trainTemp = [x for x in tmp if x not in testTemp]
#     valTemp = testTemp
    
#     trainList.extend(trainTemp)
#     testList.extend(testTemp)
#     valList.extend(valTemp)
# # print(class_name)
# # print(parent_child)

# # print(7172+2216+1836)
# # print(7172/11224)
# # print(1836/11224)

# # print(2216/11224)

# # print(subset)

# if os.path.exists("../DataSet/train.csv"):
#     os.remove("../DataSet/train.csv")

# with open("../DataSet/train.csv", "a", newline='', encoding='utf-8') as csvfile:
#     for row in trainList:
#         csvfile.write(row)
#         csvfile.write('\n')

# if os.path.exists("../DataSet/test.csv"):
#     os.remove("../DataSet/test.csv")

# with open("../DataSet/test.csv", "a", newline='', encoding='utf-8') as csvfile:
#     for row in testList:
#         csvfile.write(row)
#         csvfile.write('\n')

# if os.path.exists("../DataSet/val.csv"):
#     os.remove("../DataSet/val.csv")

# with open("../DataSet/val.csv", "a", newline='', encoding='utf-8') as csvfile:
#     for row in valList:
#         csvfile.write(row)
#         csvfile.write('\n')

cate = {'SalsaSpin': '跳舞旋转', 'BlowingCandles': '吹蜡烛', 'BlowDryHair': '吹干头发', 'Skiing': '滑雪', 'Bowling': '打保龄球', 'BrushingTeeth': '刷牙', 'HandstandWalking': '倒立走', 'JumpRope': '跳绳', 'ThrowDiscus': '掷铁饼', 'PullUps': '引体向上 ', 'brush_hair': '刷毛', 'PommelHorse': '鞍马', 'JumpingJack': '开合跳', 'Diving': '跳水', 'PlayingPiano': '弹钢琴', 'stand': '站立', 'StillRings': '静止环', 'shoot_gun': '射击枪', 'sit': '坐下', 'BreastStroke': '蛙泳', 'Swing': '摆动', 'LongJump': '跳远', 'SoccerJuggling': '足球杂耍', 'Hammering': '锤击', 'push': '推', 'wave': '招手', 'YoYo': '溜溜球', 'VolleyballSpiking': '排球扣球', 'PlayingFlute': '吹长笛', 'Billiards': '台球', 'pullup': '引体向上', 'Kayaking': '皮划艇', 'Skijet': '游艇游行', 'WallPushups': '墙上俯卧撑', 'Shotput': '铅球', 'Haircut': '剪发', 'bow': '射弓箭', 'PoleVault': '撑杆跳高', 'UnevenBars': '高低杠', 'TaiChi': '太极', 'HighJump': '跳高', 'Punch': '拳击', 'phone': '开车打电话', 'Rafting': '漂流', 'Surfing': '冲浪', 'MoppingFloor': '拖地', 'ShavingBeard': '刮胡子', 'PlayingSitar': '弹西塔琴', 'BoxingPunchingBag': '打沙袋', 'pick': '弯腰捡东西', 'jump': '跳跃', 'run': '跑步', 'HulaHoop': '转呼啦圈', 'CliffDiving': '悬崖跳水', 'FrontCrawl': '爬泳', 'PlayingGuitar': '弹吉他', 'pour': '倒水', 'TennisSwing': '网球挥拍', 'Biking': '骑自行车', 'PlayingDhol': '打击双面鼓', 'Fencing': '击剑', 'swing_baseball': '棒球挥杆', 'talk': '开车聊天', 'ParallelBars': '双杠运动', 'BasketballDunk': '篮球扣篮运动', 'WalkingWithDog': '遛狗', 'ApplyEyeMakeup': '化妆', 'Rowing': '赛艇运动', 'Typing': '键盘打字', 'shoot_bow': '射箭', 'drink': '开车喝东西', 'SkyDiving': '跳伞', 'CuttingInKitchen': '厨房做饭', 'SkateBoarding': '滑板', 'straight': '开车', 'TrampolineJumping': '蹦床跳', 'ApplyLipstick': '涂眼妆', 'clap': '拍手 ', 'SumoWrestling': '相扑运动', 'BenchPress': '举重运动', 'climb_stairs': '爬楼梯', 'catch': '双手接东西', 'PlayingViolin': '弹小提琴', 'SoccerPenalty': '足球点球', 'WritingOnBoard': '板书', 'Basketball': '打篮球', 'PlayingTabla': '打单面鼓', 'BabyCrawling': '婴儿爬行', 'TableTennisShot': '乒乓球', 'throw': '投掷', 'walk': '步行', 'HorseRiding': '骑马', 'Drumming': '击鼓', 'PushUps': '俯卧撑'}

category = []
count = []
rate = []
all = 0
l = []
for file in fileList:
#     # 是文件则略过
    if os.path.isfile(filePath+file):
        continue

    # 输出指定后缀类型的文件
    tmp = []
    subFileList = os.listdir(filePath + file)
    i=0
    for subFile in subFileList:
        i = i+1
        tmp.append(filePath + file + '/' + subFile + ' ' + str(index))
    # "\"straight\"": 0,
    class_name["\\"+file+"\\"]=index
    tmp = []
    tmp.append(file)
    parent_child[file]=tmp
    subset.append(file)
    index = index + 1
    category.append(file)
    count.append(i)
    rate.append(i/2303*100)
    all = all+i
    # d = {}
  
    # d["name"]=cate[file]
    # d["value"]=i*10

    # ll = []
    # ll.append(d)
    # t = {}
    # t["name"]=cate[file]
    # t["data"]=ll
    # l.append(t)

    # testTemp = random.sample(tmp, int(0.2*len(tmp)))
    # train_val = [x for x in tmp if x not in testTemp]
    # trainTemp = random.sample(train_val, int(0.8*len(train_val)))
    # valTemp = [x for x in train_val if x not in trainTemp]
    
    # trainList.extend(trainTemp)
    # testList.extend(testTemp)
    # valList.extend(valTemp)

print(category)
print(count)
print(rate)
print(all)
# print(len(category))
# print(l)

series = []
for i in range(len(category)):
    data = {}
    data['name']=category[i]
    data['y']=count[i]*5
    series.append(data)
print(series)