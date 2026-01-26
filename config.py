
import os
import sys
import os

if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

CONFIG = {
    'epochs': 50,
    'lr': 5e-6,
    'head_lr': 1e-5,
    'batch_size': 32,
    'max_len': 160,
    'weight_decay': 0.05,
    'early_stopping_patience': 8,
    'base_path': 'e:/dlcw',
    'model_save_path': 'best_model.pth',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

EMOTION_MAP = {
    'neutral': 0, 'joy': 1, 'surprise': 2, 'anger': 3,
    'sadness': 4, 'disgust': 5, 'fear': 6
}

