# 测试
1. 多视图测试  
沿着时间轴均匀地对视频中的N个片段进行采样。对于每个剪辑，需要3个作物来覆盖空间维度，然后在所有Nx3视图中平均softmax分数，形成视频级预测。所有的视频预测将与地面真实标签进行比较，并记录最终的测试性能。


数据集预处理
1. 训练、验证：随机裁剪、缩放、翻转
2. 测试：均匀裁剪，均匀采样（需要转换30fps帧率）
   1. 帧率=帧数/时间
   2. 时间戳增量=采样频率/帧率
3. 为了统一种植，如果宽度大于高度，我们采用左边、中间和右边的作物，或者如果高度大于宽度，我们采用顶部、中心和底部的作物。



4. pyav进行解码和时间采样
   1. decode
   2. 帧采样前转换为30fps的视频
5. 对给定的视频帧执行空间采样。如果spatial_idx为-1，则对给定的帧执行随机缩放、随机裁剪和随机翻转。如果spatial_idx为0、1或2，则对给定的spatial_idx执行空间均匀采样。
   1. spatial_sampling()
   2. torchvision.transforms.Resize()可实现


## python注册器
```
from fvcore.common.registry import Registry

registry_machine = Registry('registry_machine')

@registry_machine.register()
def print_hello_world(word):
    print('hello {}'.format(word))

@registry_machine.register()
def print_hi_world(word):
    print('hi {}'.format(word))

if __name__ == '__main__':

    cfg1 = 'print_hello_world'
    registry_machine.get(cfg1)('world')

    cfg2 = 'print_hi_world'
    registry_machine.get(cfg2)('world')
    for k, v in registry_machine.__iter__():
        print(f"key: {k}, value: {v}")
```

## argparse解析命令行参数，用于服务器训练
```
import argparse
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument()
# 解析命令行参数
parser.parse_args()
```
## yaml

## setuptools，打包分发工具

## tensorboard 可视化工具


## 数据预处理，kinetics.py
pyav解码视频，pyav container.decode解码一个视频，得到解码的帧，失败则重新寻找视频解码（重试10次）

1. 构建Kinetics数据集，根据模式train、val、test，读取数据集文件夹下的csv文件。
2. 读取csv文件内容，将路径名放入列表中path_to_videos。训练模式每个路径放一次，测试模式放三十次（训练时每个视频随机采样一个，测试采集三十个聚合预测）。对于每个视频进行随机采样，训练时，对每帧进行放大
3. pyav（decode）解码视频返回帧列表。生成pyav container容器，再进行解码。
   1. 根据container获取视频流
   2. 帧采样，同时转换成30fps帧率。训练验证随机采样，测试均匀采样。采样率为8（帧之间间隔帧），一次采样8帧。实现是通过视频流的索引，获取起始帧的idx，从而得到每次采样的初始帧、结束帧的时间戳（PTS，显示时间戳）用于视频流解码
4. 颜色标准化，通过减去均值除以标准差来标准化一个给定张量。图像标准化是将数据通过去均值实现中心化的处理，根据凸优化理论与数据概率分布相关知识，数据中心化符合数据分布规律，更容易取得训练之后的泛化效果,
   1. RGB，mean均为0.45，std均为0.225
5. b t h w c -> (b t) c h w
6. 数据增强，空间采样，区间[256,320]，原始尺寸224
   1. -1表示训练、验证
      1. 随机缩放，256-320；
      2. 随机水平翻转，翻转概率0.5
      3. 随机采样3个
   2. 0，1，2表示测试，分别对应如果宽度大于高度，则对应左、中或右;如果高度大于宽度，则对应顶、中、底



## dataloader加载数据集用于训练、测试，
1. batch_size = 8

## 构建模型
### vit构建
1. 图片扁平化，卷积公式,nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
2. b c t h w -> (b t) c h w -> (b t) embed_dim d d -> (b t) d*d embed_dim
   1. d=image_size/patch_size=224/16=14
3. 自学习位置变量，1 1 embed_dim == 1 1 768
4. 分类头,1 num_patch+1 embed_dim == 1 14*14+1 768
5. x先加分类头再加position embedding：``(b t) d*d embed_dim -> (b t) d*d+1 embed_dim -> (b t) d*d+1 embed_dim ``
```
图片输入大小为：W x W x D1
卷积核尺寸为: F x F
步长为： S
填充为：P
卷积核个数为：K
输出图片大小为：N x N x K
N = （W-F+2P）/ S   +1
```
### space vit
1. x: (b t) n embed_dim -> (b n) t embed_dim
   1. n=d*d

## 构造优化器
1. 初始学习率0.1
2. 最大epoch=300
3. 动量0.9
4. L2正则化，权重衰减1e-4
5. SGD


训练过程
1. 训练
   1. model.train，更新学习率
   2. 计算loss
   3. 反向传播
   ```
    optimizer.zero_grad()
    loss.backward()
    # Update the parameters.
    optimizer.step()
   ```
2. 保存checkpoint




# 标签
```
import os
import pandas as pd
import csv
mainpath = "D:\BaiduNetdiskDownload\dataSet" #文件夹目录

dict={'right':0,'left':0,'shift':0,'straight':0,'press':0,
      'bow':1,'phone':1,'drink':2,'talk':3,'faint':4}
result = []#所有的文件
for maindir, subdir, file_name_list in os.walk(mainpath):
     # print("1:", maindir)  # 当前主目录
     # print("2:", subdir)  # 当前主目录下的所有目录
     # print("3:", file_name_list)  # 当前主目录下的所有文件
     for filename in subdir:
          print(filename)
          apath = os.path.join(maindir, filename)  # 合并成一个完整路径
          for d,s,list in os.walk(apath):
               for file in list:
                    label = apath+'/'+file+' '+str(dict[filename])
                    result.append(label)
     break
print(result)

with open(mainpath+'\\'+'label.csv', 'w', newline='') as csvfile:
     writer = csv.writer(csvfile)
     for line in result:
          if line != '':  # 去除空行
               writer.writerow([line])



import os
import pandas as pd
import csv
# mainpath = "D:\BaiduNetdiskDownload\dataSet" #文件夹目录
mainpath = "drive_dataset" 
dict={'right':0,'left':0,'shift':0,'straight':0,'press':0,
      'bow':1,'phone':1,'drink':2,'talk':3,'faint':4}
result = []#所有的文件
for maindir, subdir, file_name_list in os.walk(mainpath):
     # print("1:", maindir)  # 当前主目录
     # print("2:", subdir)  # 当前主目录下的所有目录
     # print("3:", file_name_list)  # 当前主目录下的所有文件
     for filename in subdir:
          print(filename)
          apath = os.path.join(maindir, filename)  # 合并成一个完整路径
          for d,s,list in os.walk(apath):
               for file in list:
                    label = apath+'/'+file+' '+str(dict[filename])
                    result.append(label)
     break
print(result)

result=pd.DataFrame(result)
train_val = result.sample(frac=0.9,random_state=0,axis=0)
test=result[~result.index.isin(train_val.index)]

train=train_val.sample(frac=0.9,random_state=0,axis=0)
val=train_val[~train_val.index.isin(train.index)]

train.to_csv(mainpath+'/'+'train.csv',header=0,index=0)
val.to_csv(mainpath+'/'+'val.csv',header=0,index=0)
test.to_csv(mainpath+'/'+'test.csv',header=0,index=0)
# with open(mainpath+'\\'+'label.csv', 'w', newline='') as csvfile:
#      writer = csv.writer(csvfile)
#      for line in result:
#           if line != '':  # 去除空行
#                writer.writerow([line])


```


阿里云 http://mirrors.aliyun.com/pypi/simple/
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
豆瓣(douban) http://pypi.douban.com/simple/