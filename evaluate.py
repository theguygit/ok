
import sys
import os

# Add project root to sys.path
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from config import CONFIG, EMOTION_MAP
from dataset import MECPEDataset
from model import DualStreamMECPE
from transformers import RobertaTokenizer

def load_model_for_eval(device):
    model = DualStreamMECPE()
    path = os.path.join(CONFIG['base_path'], CONFIG['model_save_path'])
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded model from {path}")
    else:
        print(f"Warning: Model file not found at {path}")
    model.to(device)
    model.eval()
    return model

def plot_cm(y_true, y_pred, title, labels, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    # Save figure
    filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.png'
    save_path = os.path.join(CONFIG['base_path'], filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def evaluate_set(loader, model, device, title_suffix="(Validation)"):
    all_preds_e, all_labels_e = [], []
    all_preds_c, all_labels_c = [], []

    print(f"Generating metrics for {title_suffix}...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            audio = batch['audio_vec'].to(device).float()
            lbl_e = batch['emotion_label'].to(device)
            lbl_c = batch['cause_label'].to(device)

            out_e, out_c = model(ids, mask, audio)

            p_e = torch.argmax(out_e, dim=1)
            p_c = torch.argmax(out_c, dim=1)

            all_preds_e.extend(p_e.cpu().numpy())
            all_labels_e.extend(lbl_e.cpu().numpy())

            valid_mask = lbl_c != -1
            if valid_mask.sum() > 0:
                all_preds_c.extend(p_c[valid_mask].cpu().numpy())
                all_labels_c.extend(lbl_c[valid_mask].cpu().numpy())

    # Reports
    print("\n" + "="*40)
    print(f"EMOTION RECOGNITION {title_suffix}")
    print("="*40)
    emotion_names = list(EMOTION_MAP.keys())
    print(classification_report(all_labels_e, all_preds_e, target_names=emotion_names, zero_division=0))
    plot_cm(all_labels_e, all_preds_e, f"Emotion CM {title_suffix}", emotion_names, cmap='Blues')

    print("\n" + "="*40)
    print(f"CAUSAL SPAN EXTRACTION {title_suffix}")
    print("="*40)
    
    if len(all_labels_c) > 0:
        unique = sorted(list(set(all_labels_c)))
        names = [f"Lag {i}" for i in unique]
        print(classification_report(all_labels_c, all_preds_c, labels=unique, target_names=names, zero_division=0))
        plot_cm(all_labels_c, all_preds_c, f"Cause CM {title_suffix}", names, cmap='Greens')
    else:
        print("No valid cause labels found.")

def run_evaluation():
    device = CONFIG['device']
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Load Dev Set
    dev_ds = MECPEDataset(os.path.join(CONFIG['base_path'], 'dev_sent_emo.csv'),
                          os.path.join(CONFIG['base_path'], 'dev.json'),
                          os.path.join(CONFIG['base_path'], 'dev_audio_features.pkl'), tokenizer, CONFIG['max_len'])
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False)
    
    model = load_model_for_eval(device)
    evaluate_set(dev_loader, model, device)

if __name__ == "__main__":
    run_evaluation()
