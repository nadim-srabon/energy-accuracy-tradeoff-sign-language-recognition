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
import torchvision

plt.rcParams['figure.dpi'] = 110
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
import platform
HARDWARE_INFO = {'torch_version': torch.__version__, 'cuda_version_torch_built_with': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None, 'python_version': platform.python_version(), 'platform': platform.platform(), 'cpu_count_logical': __import__('os').cpu_count(), 'gpu_available': torch.cuda.is_available()}

class CONFIG:
    WORK_DIR = Path('/kaggle/working')
    CACHE_DIRS = [Path('/kaggle/input/notebooks/aniruddha2022/asl-shared-cache-building-id00/feature_cache_shard_0'), Path('/kaggle/input/notebooks/redoanmozumder/asl-shared-cache-building-id02/feature_cache_shard_2'), Path('/kaggle/input/notebooks/pranaysahaps/asl-shared-cache-building-id01/feature_cache_shard_1')]
    SPLITS_DIR = Path('/kaggle/input/datasets/pranaysahaps/asl-citizen/splits')
    MODEL_TAG = 'resnet50_cnn_heavy'
    CHECKPOINT_DIR = WORK_DIR / f'checkpoints_{MODEL_TAG}'
    OUTPUT_DIR = WORK_DIR / f'green_report_{MODEL_TAG}'
    TOP_K_CLASSES = 100
    FRAME_COUNT = 16
    APPEARANCE_STREAM_IDX = 0
    IMAGE_SIZE = 224
    BACKBONE = 'resnet50'
    UNFREEZE_FROM_LAYER = 'layer4'
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

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def frames_to_tensor(frames_uint8):
    """(T, H, W, 3) uint8 -> (T, 3, IMAGE_SIZE, IMAGE_SIZE) float, ImageNet-normalized."""
    t = torch.from_numpy(frames_uint8).float() / 255.0
    t = t.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), mode='bilinear', align_corners=False)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t

class WLASLAppearanceDataset(Dataset):
    """Same cache, same manifest logic as the Capstone geometry-only notebook --
    this pulls the RGB appearance frames instead of the MediaPipe landmarks, since
    the CNN family (MobileNetV3 / ResNet-50) needs pixels, not landmark vectors."""

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
