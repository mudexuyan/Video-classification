source activate
conda activate timesformer


nvidia-smi
top 

# 训练
python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml 
# 测试
python tools/run_net.py --cfg configs/Kinetics/TimeSformer_TEST.yaml

# 后台运行
nohup python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml &
# 输出指定文件
nohup yourcommand > nohup.log 2>&1 &
nohup python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml > result/train/nohup.log 2>&1 &

# 查看9999端口
lsof -i:9999

可以使用以下六种方法查看端口信息。

ss：可以用于转储套接字统计信息。
netstat：可以显示打开的套接字列表。
lsof：可以列出打开的文件。
fuser：可以列出那些打开了文件的进程的进程 ID。
nmap：是网络检测工具和端口扫描程序。
systemctl：是 systemd 系统的控制管理器和服务管理器。

# 指定Gpu
CUDA_VISIBLE_DEVICES=1 python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml 

CUDA_VISIBLE_DEVICES=1 nohup python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml &



# 可视化
tensorboard  --port=<port-number> --logdir result/train/tensorboard/log
tensorboard  --port=<port-number> --logdir result/test/tensorboard/log
tensorboard  --port=6006 --logdir result/test/tensorboard/log
# 显示直方图
tensorboard  --port=6006 --logdir result/test/tensorboard/log  --load_fast false

# 预测结果
import pickle
with open('result/test/pred_label.txt', 'rb') as f:
    data = pickle.load(f)
print(data) 


divided_space_time  121266442
joint_space_time  85812490
space_only  85806346


('model.blocks.11', Block(
  (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
    )
  (temporal_norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  (temporal_attn): Attention(
    (qkv): Linear(in_features=768, out_features=2304, bias=True)
    (proj): Linear(in_features=768, out_features=768, bias=True)
    (proj_drop): Dropout(p=0.0, inplace=False)
    (attn_drop): Dropout(p=0.0, inplace=False)
  )
  (temporal_fc): Linear(in_features=768, out_features=768, bias=True)
  (drop_path): DropPath()
  (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  (mlp): Mlp(
    (fc1): Linear(in_features=768, out_features=3072, bias=True)
    (act): GELU()
    (fc2): Linear(in_features=3072, out_features=768, bias=True)
    (drop): Dropout(p=0.0, inplace=False)
  )
))
('model.blocks.11.norm1', LayerNorm((768,), eps=1e-06, elementwise_affine=True))
('model.blocks.11.attn', Attention(
  (qkv): Linear(in_features=768, out_features=2304, bias=True)
  (proj): Linear(in_features=768, out_features=768, bias=True)
  (proj_drop): Dropout(p=0.0, inplace=False)
  (attn_drop): Dropout(p=0.0, inplace=False)
))
('model.blocks.11.attn.qkv', Linear(in_features=768, out_features=2304, bias=True))
('model.blocks.11.attn.proj', Linear(in_features=768, out_features=768, bias=True))
('model.blocks.11.attn.proj_drop', Dropout(p=0.0, inplace=False))
('model.blocks.11.attn.attn_drop', Dropout(p=0.0, inplace=False))
('model.blocks.11.temporal_norm1', LayerNorm((768,), eps=1e-06, elementwise_affine=True))
('model.blocks.11.temporal_attn', Attention(
  (qkv): Linear(in_features=768, out_features=2304, bias=True)
  (proj): Linear(in_features=768, out_features=768, bias=True)
  (proj_drop): Dropout(p=0.0, inplace=False)
  (attn_drop): Dropout(p=0.0, inplace=False)
))
('model.blocks.11.temporal_attn.qkv', Linear(in_features=768, out_features=2304, bias=True))
('model.blocks.11.temporal_attn.proj', Linear(in_features=768, out_features=768, bias=True))
('model.blocks.11.temporal_attn.proj_drop', Dropout(p=0.0, inplace=False))
('model.blocks.11.temporal_attn.attn_drop', Dropout(p=0.0, inplace=False))
('model.blocks.11.temporal_fc', Linear(in_features=768, out_features=768, bias=True))
('model.blocks.11.drop_path', DropPath())
('model.blocks.11.norm2', LayerNorm((768,), eps=1e-06, elementwise_affine=True))
('model.blocks.11.mlp', Mlp(
  (fc1): Linear(in_features=768, out_features=3072, bias=True)
  (act): GELU()
  (fc2): Linear(in_features=3072, out_features=768, bias=True)
  (drop): Dropout(p=0.0, inplace=False)
))
('model.blocks.11.mlp.fc1', Linear(in_features=768, out_features=3072, bias=True))
('model.blocks.11.mlp.act', GELU())
('model.blocks.11.mlp.fc2', Linear(in_features=3072, out_features=768, bias=True))
('model.blocks.11.mlp.drop', Dropout(p=0.0, inplace=False))


