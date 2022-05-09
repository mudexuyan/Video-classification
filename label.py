import os
import random

random.seed(0)

dict = {'straight': 0, 'left': 1, 'right': 2, 'shift': 3, 'press': 4, 'drink': 5, 'faint': 6, 'bow': 7, 'phone': 8,
        'talk': 9}

filePath = "../DataSet/"  # 文件夹目录
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
        tmp.append(filePath + file + '/' + subFile + ' ' + str(dict[file]))

    # testTemp = random.sample(tmp, int(0.2*len(tmp)))
    # train_val = [x for x in tmp if x not in testTemp]
    # trainTemp = random.sample(train_val, int(0.8*len(train_val)))
    # valTemp = [x for x in train_val if x not in trainTemp]
    # testTemp = random.sample(tmp, int(0.01*len(tmp)))
    # train_val = [x for x in tmp if x not in testTemp]
    # trainTemp = random.sample(train_val, int(0.8*len(train_val)))
    # valTemp = [x for x in train_val if x not in trainTemp]
    
    testTemp = random.sample(tmp, int(0.98*len(tmp)))
    train_val = [x for x in tmp if x not in testTemp]
    trainTemp = random.sample(train_val, int(0.8*len(train_val)))
    valTemp = [x for x in train_val if x not in trainTemp]

    trainList.extend(trainTemp)
    testList.extend(testTemp)
    valList.extend(valTemp)


if os.path.exists("../DataSet/train.csv"):
    os.remove("../DataSet/train.csv")

with open("../DataSet/train.csv", "a", newline='', encoding='utf-8') as csvfile:
    for row in trainList:
        csvfile.write(row)
        csvfile.write('\n')

if os.path.exists("../DataSet/test.csv"):
    os.remove("../DataSet/test.csv")

with open("../DataSet/test.csv", "a", newline='', encoding='utf-8') as csvfile:
    for row in testList:
        csvfile.write(row)
        csvfile.write('\n')

if os.path.exists("../DataSet/val.csv"):
    os.remove("../DataSet/val.csv")

with open("../DataSet/val.csv", "a", newline='', encoding='utf-8') as csvfile:
    for row in valList:
        csvfile.write(row)
        csvfile.write('\n')

