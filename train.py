import sys
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import collections
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import json
import matplotlib.pyplot as plt

from config import CONFIG, EMOTION_MAP
from dataset import MECPEDataset
from model import DualStreamMECPE

def get_optimizer_params(model, base_lr, weight_decay=0.01):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = []
    for i in range(12):
        lr = base_lr * (0.9 ** (11 - i))
        optimizer_grouped_parameters.append({
            "params": [p for n, p in param_optimizer if f"roberta.encoder.layer.{i}." in n and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay, "lr": lr
        })
        optimizer_grouped_parameters.append({
            "params": [p for n, p in param_optimizer if f"roberta.encoder.layer.{i}." in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": lr
        })
        
    head_params = [p for n, p in param_optimizer if "roberta" not in n]
    optimizer_grouped_parameters.append({
        "params": head_params, "weight_decay": weight_decay, "lr": CONFIG.get('head_lr', 1e-4)
    })
    
    return optimizer_grouped_parameters

def train():
    device = CONFIG['device']
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    print("Loading Datasets...")
    train_ds = MECPEDataset(os.path.join(CONFIG['base_path'], 'train_sent_emo.csv'),
                            os.path.join(CONFIG['base_path'], 'Subtask_2_train.json'),
                            os.path.join(CONFIG['base_path'], 'audio_features.pkl'), tokenizer, CONFIG['max_len'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    dev_ds = MECPEDataset(os.path.join(CONFIG['base_path'], 'dev_sent_emo.csv'),
                          os.path.join(CONFIG['base_path'], 'dev.json'),
                          os.path.join(CONFIG['base_path'], 'dev_audio_features.pkl'), tokenizer, CONFIG['max_len'])
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False)

    model = DualStreamMECPE().to(device)
    save_path = os.path.join(CONFIG['base_path'], CONFIG['model_save_path'])

    if os.path.exists(save_path):
        print(f"🔄 Resuming from: {save_path}")
        try: model.load_state_dict(torch.load(save_path, map_location=device))
        except: print("⚠️ Starting fresh.")
    
    opt_params = get_optimizer_params(model, CONFIG['lr'], CONFIG['weight_decay'])
    optimizer = AdamW(opt_params)
    
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)


    total_e = len(train_ds.df)
    counts_e = collections.Counter(train_ds.df['Emotion'].str.lower())
    weights_e = torch.tensor([total_e/(len(EMOTION_MAP)*counts_e.get(e,1)) for e in EMOTION_MAP.keys()]).float().to(device)
    criterion_e = nn.CrossEntropyLoss(weight=torch.clamp(weights_e, 0.5, 20.0), label_smoothing=0.1)
    
    cause_counts = np.zeros(6)
    for key in train_ds.cause_map: cause_counts += train_ds.cause_map[key]
    pos_weights_c = torch.tensor([(len(train_ds)-c-1)/(c+1) for c in cause_counts]).float().to(device)
    criterion_c = nn.BCEWithLogitsLoss(pos_weight=torch.clamp(pos_weights_c, 1.0, 50.0))

    best_score, history = 0, collections.defaultdict(list)
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        preds_e, labels_e, preds_c, labels_c = [], [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            ids, mask, audio = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['audio_vec'].to(device)
            lbl_e, lbl_c = batch['emotion_label'].to(device), batch['cause_label'].to(device)

            optimizer.zero_grad()
            out_e, out_e_metric, out_c = model(ids, mask, audio, emotion_label=lbl_e)
            loss = 0.5*criterion_e(out_e, lbl_e) + 0.5*criterion_c(out_c, lbl_c)
            
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            preds_e.extend(torch.argmax(out_e_metric, dim=1).cpu().numpy())
            labels_e.extend(lbl_e.cpu().numpy())
            preds_c.append((torch.sigmoid(out_c)>0.5).float().cpu().detach().numpy())
            labels_c.append(lbl_c.cpu().numpy())


        model.eval()
        v_preds_e, v_labels_e, v_preds_c, v_labels_c = [], [], [], []
        val_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                ids, mask, audio = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['audio_vec'].to(device)
                lbl_e, lbl_c = batch['emotion_label'].to(device), batch['cause_label'].to(device)
                
                out_e, out_e_metric, out_c = model(ids, mask, audio, emotion_label=lbl_e)
                loss = 0.5*criterion_e(out_e, lbl_e) + 0.5*criterion_c(out_c, lbl_c)
                val_loss += loss.item()
                
                v_preds_e.extend(torch.argmax(out_e_metric, dim=1).cpu().numpy())
                v_labels_e.extend(lbl_e.cpu().numpy())
                v_preds_c.append((torch.sigmoid(out_c)>0.5).float().cpu().numpy())
                v_labels_c.append(lbl_c.cpu().numpy())



        train_preds_c, train_labels_c = np.concatenate(preds_c, 0), np.concatenate(labels_c, 0)
        v_preds_c, v_labels_c = np.concatenate(v_preds_c, 0), np.concatenate(v_labels_c, 0)

        results = {
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(dev_loader),
            'tr_f1_e_m': f1_score(labels_e, preds_e, average='macro'),
            'tr_f1_e_w': f1_score(labels_e, preds_e, average='weighted'),
            'tr_prec_e_m': precision_score(labels_e, preds_e, average='macro', zero_division=0),
            'tr_rec_e_m': recall_score(labels_e, preds_e, average='macro', zero_division=0),
            'val_f1_e_m': f1_score(v_labels_e, v_preds_e, average='macro'),
            'val_f1_e_w': f1_score(v_labels_e, v_preds_e, average='weighted'),
            'val_prec_e_m': precision_score(v_labels_e, v_preds_e, average='macro', zero_division=0),
            'val_rec_e_m': recall_score(v_labels_e, v_preds_e, average='macro', zero_division=0),
            'tr_f1_c_m': f1_score(train_labels_c, train_preds_c, average='macro', zero_division=0),
            'tr_f1_c_w': f1_score(train_labels_c, train_preds_c, average='weighted', zero_division=0),
            'tr_prec_c_m': precision_score(train_labels_c, train_preds_c, average='macro', zero_division=0),
            'tr_rec_c_m': recall_score(train_labels_c, train_preds_c, average='macro', zero_division=0),
            'val_f1_c_m': f1_score(v_labels_c, v_preds_c, average='macro', zero_division=0),
            'val_f1_c_w': f1_score(v_labels_c, v_preds_c, average='weighted', zero_division=0),
            'val_prec_c_m': precision_score(v_labels_c, v_preds_c, average='macro', zero_division=0),
            'val_rec_c_m': recall_score(v_labels_c, v_preds_c, average='macro', zero_division=0),
        }

        for k, v in results.items(): history[k].append(v)

        print(f"\n📊 Epoch {epoch+1}:")
        print(f"   [Loss] Train: {results['train_loss']:.4f} | Val: {results['val_loss']:.4f}")
        print(f"   [Train] Emo F1 (M/W): {results['tr_f1_e_m']:.4f}/{results['tr_f1_e_w']:.4f} | Prec/Rec: {results['tr_prec_e_m']:.4f}/{results['tr_rec_e_m']:.4f}")
        print(f"   [Val]   Emo F1 (M/W): {results['val_f1_e_m']:.4f}/{results['val_f1_e_w']:.4f} | Prec/Rec: {results['val_prec_e_m']:.4f}/{results['val_rec_e_m']:.4f}")
        print(f"   [Train] Cause F1 (M/W): {results['tr_f1_c_m']:.4f}/{results['tr_f1_c_w']:.4f} | Prec/Rec: {results['tr_prec_c_m']:.4f}/{results['tr_rec_c_m']:.4f}")
        print(f"   [Val]   Cause F1 (M/W): {results['val_f1_c_m']:.4f}/{results['val_f1_c_w']:.4f} | Prec/Rec: {results['val_prec_c_m']:.4f}/{results['val_rec_c_m']:.4f}")
        
        if (epoch + 1) % 5 == 0:
            print("\n📋 Detailed Emotion Classification Report (Validation):")
            emotion_names = list(EMOTION_MAP.keys())
            print(classification_report(v_labels_e, v_preds_e, target_names=emotion_names, zero_division=0))
        
        score = 0.5*results['val_f1_e_m'] + 0.5*results['val_f1_c_m']
        if score > best_score:
            best_score = score; torch.save(model.state_dict(), save_path)
            print(f"✅ Best Model Saved! (Score: {best_score:.4f})")
            patience_counter = 0
        elif patience_counter >= CONFIG['early_stopping_patience']:
            print("⏹️ Early stopping!"); break
        else: patience_counter += 1

    history_path = os.path.join(CONFIG['base_path'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Training history saved to: {history_path}")
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    fig = plt.figure(figsize=(20, 12))
    
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(epochs_range, history['tr_f1_e_m'], 'b--', label='Train F1 (Macro)', linewidth=2, alpha=0.7)
    plt.plot(epochs_range, history['val_f1_e_m'], 'r-', label='Val F1 (Macro)', linewidth=2)
    plt.plot(epochs_range, history['val_f1_e_w'], 'r--', label='Val F1 (Weighted)', linewidth=2, alpha=0.7)
    plt.title('Emotion Recognition F1', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(epochs_range, history['val_prec_e_m'], 'g-', label='Val Precision', linewidth=2)
    plt.plot(epochs_range, history['val_rec_e_m'], 'orange', label='Val Recall', linewidth=2)
    plt.title('Emotion Precision & Recall', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 3, 4)
    plt.plot(epochs_range, history['tr_f1_c_m'], 'b--', label='Train F1 (Macro)', linewidth=2, alpha=0.7)
    plt.plot(epochs_range, history['val_f1_c_m'], 'r-', label='Val F1 (Macro)', linewidth=2)
    plt.plot(epochs_range, history['val_f1_c_w'], 'r--', label='Val F1 (Weighted)', linewidth=2, alpha=0.7)
    plt.title('Cause Extraction F1', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    plt.plot(epochs_range, history['val_prec_c_m'], 'g-', label='Val Precision', linewidth=2)
    plt.plot(epochs_range, history['val_rec_c_m'], 'orange', label='Val Recall', linewidth=2)
    plt.title('Cause Precision & Recall', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 3, 6)
    combined_scores = [0.5*e + 0.5*c for e, c in zip(history['val_f1_e_m'], history['val_f1_c_m'])]
    plt.plot(epochs_range, combined_scores, 'purple', label='Combined Score', linewidth=2)
    plt.axhline(y=best_score, color='r', linestyle='--', label=f'Best: {best_score:.4f}')
    plt.title('Combined Validation Score', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 7)
    plt.plot(epochs_range, history['tr_f1_e_m'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs_range, history['val_f1_e_m'], 'r-', label='Validation', linewidth=2)
    plt.title('Emotion F1: Train vs Val', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (Macro)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    plt.plot(epochs_range, history['tr_f1_c_m'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs_range, history['val_f1_c_m'], 'r-', label='Validation', linewidth=2)
    plt.title('Cause F1: Train vs Val', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (Macro)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax9 = plt.subplot(3, 3, 9)
    emo_gap = [t - v for t, v in zip(history['tr_f1_e_m'], history['val_f1_e_m'])]
    cause_gap = [t - v for t, v in zip(history['tr_f1_c_m'], history['val_f1_c_m'])]
    plt.plot(epochs_range, emo_gap, 'b-', label='Emotion Gap', linewidth=2)
    plt.plot(epochs_range, cause_gap, 'r-', label='Cause Gap', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Train-Val Gap (Overfitting)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Gap (Train - Val)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('MECPE Training Curves - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_path = os.path.join(CONFIG['base_path'], 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    train()
