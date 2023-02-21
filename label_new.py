import os
import random

random.seed(0)

index = 0
# xml文件中tensorboard class_name
class_name = {}
parent_child = {}
subset = []



filePath = "../video_dataset/"  # 文件夹目录
# 返回path下所有文件构成的一个list列表
fileList = os.listdir(filePath)

trainList = []
testList = []
valList = []

# 遍历输出每一个文件的名字和类型
for file in fileList:
    # 是文件则略过
    if os.path.isfile(filePath+file):
        continue

    # 输出指定后缀类型的文件
    tmp = []
    subFileList = os.listdir(filePath + file)
    if (file == '驾驶手势数据'):
        continue
    for subFile in subFileList:
        tmp.append(filePath + file + '/' + subFile + ' ' + str(index))
    # "\"straight\"": 0,
    class_name["\\"+file+"\\"]=index
    tmp = []
    tmp.append(file)
    parent_child[file]=tmp
    subset.append(file)
    index = index + 1

    testTemp = random.sample(tmp, int(0.2*len(tmp)))
    train_val = [x for x in tmp if x not in testTemp]
    trainTemp = random.sample(train_val, int(0.8*len(train_val)))
    valTemp = [x for x in train_val if x not in trainTemp]
    
    trainList.extend(trainTemp)
    testList.extend(testTemp)
    valList.extend(valTemp)

# print(class_name)
print(parent_child)

# print(subset)

# if os.path.exists("../video_dataset/train.csv"):
#     os.remove("../video_dataset/train.csv")

# with open("../video_dataset/train.csv", "a", newline='', encoding='utf-8') as csvfile:
#     for row in trainList:
#         csvfile.write(row)
#         csvfile.write('\n')

# if os.path.exists("../video_dataset/test.csv"):
#     os.remove("../video_dataset/test.csv")

# with open("../video_dataset/test.csv", "a", newline='', encoding='utf-8') as csvfile:
#     for row in testList:
#         csvfile.write(row)
#         csvfile.write('\n')

# if os.path.exists("../video_dataset/val.csv"):
#     os.remove("../video_dataset/val.csv")

# with open("../video_dataset/val.csv", "a", newline='', encoding='utf-8') as csvfile:
#     for row in valList:
#         csvfile.write(row)
#         csvfile.write('\n')
