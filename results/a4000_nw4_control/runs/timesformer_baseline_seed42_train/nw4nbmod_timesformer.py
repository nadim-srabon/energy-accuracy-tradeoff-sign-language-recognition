import subprocess
import sys
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
from transformers import TimesformerModel, AutoImageProcessor
import transformers as _transformers_pkg

plt.rcParams['figure.dpi'] = 110
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
import platform
HARDWARE_INFO = {'torch_version': torch.__version__, 'cuda_version_torch_built_with': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None, 'python_version': platform.python_version(), 'platform': platform.platform(), 'cpu_count_logical': __import__('os').cpu_count(), 'gpu_available': torch.cuda.is_available()}

class CONFIG:
    WORK_DIR = Path('/kaggle/working')
    CACHE_DIRS = [Path('/kaggle/input/notebooks/aniruddha2022/asl-shared-cache-building-id00/feature_cache_shard_0'), Path('/kaggle/input/notebooks/redoanmozumder/asl-shared-cache-building-id02/feature_cache_shard_2'), Path('/kaggle/input/notebooks/pranaysahaps/asl-shared-cache-building-id01/feature_cache_shard_1')]
    SPLITS_DIR = Path('/kaggle/input/datasets/pranaysahaps/asl-citizen/splits')
    MODEL_TAG = 'timesformer_base_transformer_heavy'
    CHECKPOINT_DIR = WORK_DIR / f'checkpoints_{MODEL_TAG}'
    OUTPUT_DIR = WORK_DIR / f'green_report_{MODEL_TAG}'
    TOP_K_CLASSES = 100
    APPEARANCE_STREAM_IDX = 0
    IMAGE_SIZE = 224
    FRAME_COUNT = 16
    TIMESFORMER_NUM_FRAMES = 8
    TIMESFORMER_CHECKPOINT = 'facebook/timesformer-base-finetuned-k400'
    TIMESFORMER_UNFROZEN_LAYERS = 2
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    LR = 0.0001
    WEIGHT_DECAY = 0.0003
    GRAD_CLIP_NORM = 0.5
    LABEL_SMOOTHING = 0.1
    EPOCHS_FULL = 30
    MIN_EPOCHS_BEFORE_STOPPING = 15
    EARLY_STOPPING_PATIENCE = 6
    ACCURACY_GATE_THRESHOLD = 0.6
    ACCURACY_GATE_THRESHOLDS_FOR_SENSITIVITY = [0.5, 0.6, 0.7]
    SEED = 42
    SEEDS = [42, 123, 2024]
    DEVICE = DEVICE
    GRID_CARBON_INTENSITY_KG_PER_KWH = 0.41272
    EXPECTED_INFERENCE_VOLUME = 100000
    DEPLOYMENT_VOLUME_SENSITIVITY = [10000, 100000, 1000000]
    INFERENCE_PROFILING_PASSES = 3

_timesformer_processor = AutoImageProcessor.from_pretrained(CONFIG.TIMESFORMER_CHECKPOINT)
TIMESFORMER_MEAN = torch.tensor(_timesformer_processor.image_mean).view(1, 3, 1, 1)
TIMESFORMER_STD = torch.tensor(_timesformer_processor.image_std).view(1, 3, 1, 1)
print(f'TimeSformer processor normalization -- mean: {_timesformer_processor.image_mean}, std: {_timesformer_processor.image_std}')
_subsample_idx = np.linspace(0, CONFIG.FRAME_COUNT - 1, CONFIG.TIMESFORMER_NUM_FRAMES).round().astype(int)
print(f'Subsampling cached frame indices: {_subsample_idx.tolist()}')

def frames_to_tensor(frames_uint8):
    """(T_cached, H, W, 3) uint8 -> (TIMESFORMER_NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
    float, subsampled and normalized with TimeSformer's own processor stats."""
    frames_uint8 = frames_uint8[_subsample_idx]
    t = torch.from_numpy(frames_uint8).float() / 255.0
    t = t.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), mode='bilinear', align_corners=False)
    t = (t - TIMESFORMER_MEAN) / TIMESFORMER_STD
    return t

class WLASLAppearanceDataset(Dataset):
    """Same cache and manifest as notebooks 1-3 -- only frames_to_tensor differs
    (subsampling + TimeSformer-specific normalization instead of the full 16 frames
    with ImageNet normalization)."""

    def __init__(self, nslt_data, cache_index, subset):
        self.cache_index = cache_index
        self.samples = [(vid, meta['action'][0]) for vid, meta in nslt_data.items() if meta['subset'] == subset and vid in self.cache_index]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, class_idx = self.samples[idx]
        data = np.load(self.cache_index[video_id])
        appearance_raw = data['appearance']
        frames = appearance_raw[:, CONFIG.APPEARANCE_STREAM_IDX]
        frames_t = frames_to_tensor(frames)
        label = torch.tensor(class_idx, dtype=torch.long)
        return (frames_t, label)
