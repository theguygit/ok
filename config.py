
import os
import sys
import os

# Add project root to sys.path
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

CONFIG = {
    'epochs': 30,
    'lr': 3e-5,              # Base LR for BERT
    'head_lr': 3e-4,         # Higher LR for classification heads
    'batch_size': 16,
    'max_len': 160,          # Large enough for 3 sentences
    'weight_decay': 0.1,
    'base_path': 'e:/dlcw',
    'model_save_path': 'best_model.pth',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

EMOTION_MAP = {
    'neutral': 0, 'joy': 1, 'surprise': 2, 'anger': 3,
    'sadness': 4, 'disgust': 5, 'fear': 6
}
