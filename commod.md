source activate
conda activate timesformer


nvidia-smi

python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml 


nohup python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml &
nohup yourcommand > nohup.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml 

CUDA_VISIBLE_DEVICES=1 nohup python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml &

python tools/run_net.py --cfg configs/Kinetics/TimeSformer_TEST.yaml


tensorboard  --port=<port-number> --logdir result/train/tensorboard/log
tensorboard  --port=<port-number> --logdir result/test/tensorboard/log
tensorboard  --port=6006 --logdir result/test/tensorboard/log

top 



# 预测结果
import pickle
with open('result/test/pred_label.txt', 'rb') as f:
    data = pickle.load(f)
print(data) 
