from timesformer.datasets import utils as utils
from timesformer.datasets import video_container as container
from timesformer.datasets import decoder as decoder
import av

path = "../TestData/talk_DCX_01.mp4"
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
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
)

print(frames)
print(frames.shape)
