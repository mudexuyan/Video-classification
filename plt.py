
# categories1 = ['SalsaSpin', 'BlowingCandles', 'BlowDryHair', 'Skiing', 'Bowling', 'BrushingTeeth', 'HandstandWalking', 'JumpRope', 'ThrowDiscus', 'PullUps', 'brush_hair', 'PommelHorse', 'JumpingJack', 'Diving', 'PlayingPiano', 'stand', 'StillRings', 'shoot_gun', 'sit', 'BreastStroke', 'Swing', 'LongJump', 'SoccerJuggling', 
# 'Hammering', 'push', 'wave', 'YoYo', 'VolleyballSpiking', 'PlayingFlute', 'Billiards', 'pullup', 'Kayaking', 'Skijet', 'WallPushups', 'Shotput', 'Haircut', 'bow', 'PoleVault', 'UnevenBars', 'TaiChi', 'HighJump', 'Punch', 'phone', 'Rafting', 'Surfing', 'MoppingFloor', 'ShavingBeard', 'PlayingSitar', 'BoxingPunchingBag', 'pick', 'jump', 'run', 'HulaHoop', 'CliffDiving', 'FrontCrawl', 'PlayingGuitar', 'pour', 'TennisSwing', 'Biking', 'PlayingDhol', 'Fencing', 'swing_baseball', 
# 'talk', 'ParallelBars', 'BasketballDunk', 'WalkingWithDog', 'ApplyEyeMakeup', 'Rowing', 'Typing', 'shoot_bow', 'drink', 'SkyDiving', 'CuttingInKitchen', 'SkateBoarding', 'straight', 'TrampolineJumping', 'ApplyLipstick', 'clap', 'SumoWrestling', 'BenchPress', 'climb_stairs', 'catch', 'PlayingViolin', 'SoccerPenalty', 'WritingOnBoard', 'Basketball', 'PlayingTabla', 'BabyCrawling', 'TableTennisShot', 'throw', 'walk', 'HorseRiding', 'Drumming', 'PushUps']

categories2 = ['跳舞旋转', '吹蜡烛', '吹干头发', '滑雪', '打保龄球', '刷牙', '倒立走', '跳绳', '掷铁饼', '引体向上 ','刷毛','鞍马','开合跳','跳水','弹钢琴','站立','静止环','射击枪','坐下','蛙泳','摆动', '跳远', '足球杂耍',
  '锤击','推','招手','溜溜球','排球扣球','吹长笛','台球','引体向上','皮划艇','游艇游行','墙上俯卧撑', '铅球','剪发','射弓箭','撑杆跳高','高低杠','太极','跳高','拳击','开车打电话','漂流','冲浪','拖地', '刮胡子', '弹西塔琴', '打沙袋', '弯腰捡东西', '跳跃', '跑步', '转呼啦圈', '悬崖跳水', '爬泳', '弹吉他', '倒水', '网球挥拍', '骑自行车', '打击双面鼓', '击剑', '棒球挥杆'
  , '开车聊天', '双杠运动', '篮球扣篮运动', '遛狗', '化妆', '赛艇运动', '键盘打字', '射箭', '开车喝东西', '跳伞', '厨房做饭', '滑板', '开车', '蹦床跳', '涂眼妆', '拍手 ', '相扑运动', '举重运动', '爬楼梯', '双手接东西', '弹小提琴', '足球点球', '板书', '打篮球', '打单面鼓', '婴儿爬行', '乒乓球','投掷','步行','骑马','击鼓','俯卧撑']

# print(len(categories1))
# print(len(categories2))
import matplotlib.pyplot as plt
import numpy as np

x = ['SalsaSpin', 'BlowingCandles', 'BlowDryHair', 'Skiing', 'Bowling', 'BrushingTeeth', 'HandstandWalking', 'JumpRope', 'ThrowDiscus', 'PullUps', 'brush_hair', 'PommelHorse', 'JumpingJack', 'Diving', 'PlayingPiano', 'stand', 'StillRings', 'shoot_gun', 'sit', 'BreastStroke', 'Swing', 'LongJump', 'SoccerJuggling', 'Hammering', 'push', 'wave', 'YoYo', 'VolleyballSpiking', 'PlayingFlute', 'Billiards', 'pullup', 'Kayaking', 'Skijet', 'WallPushups', 'Shotput', 'Haircut', 'bow', 'PoleVault', 'UnevenBars', 'TaiChi', 'HighJump', 'Punch', 'phone', 'Rafting', 'Surfing', 'MoppingFloor', 'ShavingBeard', 'PlayingSitar', 'BoxingPunchingBag', 'pick', 'jump', 'run', 'HulaHoop', 'CliffDiving', 'FrontCrawl', 'PlayingGuitar', 'pour', 'TennisSwing', 'Biking', 'PlayingDhol', 'Fencing', 'swing_baseball', 'talk', 'ParallelBars', 'BasketballDunk', 'WalkingWithDog', 'ApplyEyeMakeup', 'Rowing', 'Typing', 'shoot_bow', 'drink', 'SkyDiving', 'CuttingInKitchen', 'SkateBoarding', 'straight', 'TrampolineJumping', 'ApplyLipstick', 'clap', 'SumoWrestling', 'BenchPress', 'climb_stairs', 'catch', 'PlayingViolin', 'SoccerPenalty', 'WritingOnBoard', 'Basketball', 'PlayingTabla', 'BabyCrawling', 'TableTennisShot', 'throw', 'walk', 'HorseRiding', 'Drumming', 'PushUps']
y = [133, 109, 131, 135, 155, 131, 111, 144, 130, 100, 41, 123, 123, 150, 105, 36, 112, 55, 39, 101, 131, 131, 147, 140, 42, 42, 128, 116, 155, 150, 55, 141, 100, 130, 144, 130, 232, 149, 104, 100, 123, 160, 221, 111, 126, 110, 161, 157, 163, 40, 39, 40, 125, 138, 137, 160, 55, 166, 134, 164, 111, 54, 230, 114, 131, 123, 145, 137, 136, 53, 213, 110, 110, 120, 215, 119, 114, 44, 116, 160, 40, 48, 100, 137, 152, 134, 111, 132, 140, 46, 41, 164, 161, 102]
color = np.random.rand(94)

x1 = ['left', 'bow', 'right', 'phone', 'press', 'talk', 'drink', 'faint', 'straight', 'shift']
y1 = [245, 232, 229, 221, 246, 230, 213, 225, 215, 247]
color1 = np.random.rand(10)

plt.scatter(x1, y1, s=y1, alpha=0.5)
# plt.colorbar()  # 显示颜色条
plt.show()
plt.savefig("image1.png")


# print(len(categories2))
# print(len(y))
# import matplotlib
# matplotlib.rc("font",family='YouYuan')
# # 中文和负号的正常显示
# # plt.rcParams['font.sans-serif'] = ['SimHei'] 
# # plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
# plt.rcParams['axes.unicode_minus'] = False

# # 使用ggplot的绘图风格
# plt.style.use('ggplot')



# N = len(y)
# # 设置雷达图的角度，用于平分切开一个圆面
# angles=np.linspace(0, 2*np.pi, N, endpoint=False)

# # 为了使雷达图一圈封闭起来，需要下面的步骤
# y=np.concatenate((y,[y[0]]))
# x=np.concatenate((x,[x[0]]))
# angles=np.concatenate((angles,[angles[0]]))

# # 绘图
# fig=plt.figure()
# # 这里一定要设置为极坐标格式
# ax = fig.add_subplot(111, polar=True)
# # 绘制折线图
# ax.plot(angles, y, 'o-', linewidth=2)
# # 填充颜色
# ax.fill(angles, y, alpha=0.25)
# # 添加每个特征的标签
# ax.set_thetagrids(angles * 180/np.pi, x)
# # 设置雷达图的范围
# ax.set_ylim(0,250)
# # 添加标题
# plt.title('behave')
# # 添加网格线
# ax.grid(True)
# # 显示图形
# plt.show()
# plt.savefig("image.png")

