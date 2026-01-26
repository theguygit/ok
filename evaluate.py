import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer

from config import CONFIG, EMOTION_MAP
from dataset import MECPEDataset
from model import DualStreamMECPE

def load_model_for_eval(device):
    model = DualStreamMECPE()
    path = os.path.join(CONFIG['base_path'], CONFIG['model_save_path'])
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"✅ Loaded: {path}")
    else:
        print(f"❌ Not found: {path}")
    return model.to(device).eval()

def plot_cm(y_true, y_pred, title, labels):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.nan_to_num(cm_norm), annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    save_path = os.path.join(CONFIG['base_path'], title.lower().replace(' ', '_') + '.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()
    plt.close()

def plot_cause_cm(y_true, y_pred, title):
    labels = [f"Lag {i}" for i in range(6)]
    matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            matrix[i, j] = np.sum((y_true[:, i] == 1) & (y_pred[:, j] == 1))
            
    row_sums = y_true.sum(axis=0)
    matrix_norm = np.divide(matrix, row_sums[:, np.newaxis], out=np.zeros_like(matrix), where=row_sums[:, np.newaxis]!=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_norm, annot=True, fmt='.2f', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Lag'); plt.ylabel('True Lag (Actual Cause)'); plt.title(title)
    save_path = os.path.join(CONFIG['base_path'], title.lower().replace(' ', '_') + '.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()
    plt.close()

def evaluate_set(loader, model, device, title_suffix="(Val)"):
    all_p_e, all_l_e, all_p_c, all_l_c, all_h_c = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            ids, mask, audio = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['audio_vec'].to(device)
            _, out_e_metric, out_c = model(ids, mask, audio)
            all_p_e.extend(torch.argmax(out_e_metric, 1).cpu().numpy())
            all_l_e.extend(batch['emotion_label'].numpy())
            all_p_c.append((torch.sigmoid(out_c)>0.5).float().cpu().numpy())
            all_l_c.append(batch['cause_label'].numpy())
            all_h_c.extend(batch['has_cause'].numpy())

    p_c, l_c, h_c = np.concatenate(all_p_c, 0), np.concatenate(all_l_c, 0), np.array(all_h_c)
    emo_names = list(EMOTION_MAP.keys())
    
    print(f"\n--- Emotion {title_suffix} ---")
    report_e = classification_report(all_l_e, all_p_e, target_names=emo_names, zero_division=0)
    print(report_e)
    print("\nText Confusion Matrix (Emotion):")
    print(confusion_matrix(all_l_e, all_p_e))
    plot_cm(all_l_e, all_p_e, f"Emotion CM {title_suffix}", emo_names)

    print(f"\n--- Cause {title_suffix} ---")
    idx = np.where(h_c == 1.0)[0]
    if len(idx) > 0:
        print(classification_report(l_c[idx], p_c[idx], target_names=[f"Lag {i}" for i in range(6)], zero_division=0))
        print("\nText Confusion Matrix (Cause):")
        
        plot_cause_cm(l_c[idx], p_c[idx], f"Cause CM {title_suffix}")
        
        f1_e, f1_c = f1_score(all_l_e, all_p_e, average='macro'), f1_score(l_c[idx], p_c[idx], average='macro', zero_division=0)
        print(f"\nFinal Combined F1: {(f1_e+f1_c)/2:.4f}")

def run_evaluation():
    device = CONFIG['device']
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = load_model_for_eval(device)

    print("\n" + "="*50)
    print(">>> EVALUATING ON VALIDATION SET (DEV) <<<")
    print("="*50)
    dev_ds = MECPEDataset(os.path.join(CONFIG['base_path'], 'dev_sent_emo.csv'), os.path.join(CONFIG['base_path'], 'dev.json'), os.path.join(CONFIG['base_path'], 'dev_audio_features.pkl'), tokenizer, CONFIG['max_len'])
    evaluate_set(DataLoader(dev_ds, batch_size=CONFIG['batch_size']), model, device, title_suffix="(Val)")


if __name__ == "__main__":
    run_evaluation()
