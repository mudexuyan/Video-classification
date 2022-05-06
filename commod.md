nvidia-smi

python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml 

nohup python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml &

CUDA_VISIBLE_DEVICES=1 python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml 

CUDA_VISIBLE_DEVICES=1 nohup python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml &

python tools/run_net.py --cfg configs/Kinetics/TimeSformer_TEST.yaml

top 

