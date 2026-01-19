
import sys
import os
import pandas as pd


# Add project root to sys.path
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from collections import Counter
import collections
from sklearn.metrics import f1_score

from config import CONFIG
from dataset import MECPEDataset
from model import DualStreamMECPE

# Initialize Tokenizer globally or pass it around
try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
except:
    print("Warning: Could not load roberta-base from hub locally. Ensure internet access or cache.")
    raise

def train():
    device = CONFIG['device']
    
    # Initialize Datasets
    print("Loading Training Set...")
    train_ds = MECPEDataset(os.path.join(CONFIG['base_path'], 'train_sent_emo.csv'),
                            os.path.join(CONFIG['base_path'], 'Subtask_2_train.json'),
                            os.path.join(CONFIG['base_path'], 'audio_features.pkl'), tokenizer, CONFIG['max_len'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)

    print("Loading Dev Set...")
    dev_ds = MECPEDataset(os.path.join(CONFIG['base_path'], 'dev_sent_emo.csv'),
                          os.path.join(CONFIG['base_path'], 'dev.json'),
                          os.path.join(CONFIG['base_path'], 'dev_audio_features.pkl'), tokenizer, CONFIG['max_len'])
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False)

    save_path = os.path.join(CONFIG['base_path'], CONFIG['model_save_path'])
    model = DualStreamMECPE().to(device)
    
    # Load checkpoint to train deeper
    if os.path.exists(save_path):
        print(f"🔄 Resuming from checkpoint: {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("🆕 No checkpoint found, starting fresh.")

    bert_params = [p for n, p in model.named_parameters() if "roberta" in n]
    head_params = [p for n, p in model.named_parameters() if "roberta" not in n]
    optimizer = AdamW([{'params': bert_params, 'lr': CONFIG['lr']},
                       {'params': head_params, 'lr': CONFIG['head_lr']}], weight_decay=CONFIG['weight_decay'])

    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(0.1 * total_steps)
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print("Calculating Class Weights...")
    # 1. Cause Weights
    counts_c = collections.Counter()
    for idx, row in train_ds.df.iterrows():
        did, uid = row['Dialogue_ID'], row['Utterance_ID']
        key = f"dia{did}_utt{uid}"
        lbl = train_ds.cause_map.get(key, -1)
        if lbl != -1:
            counts_c[lbl] += 1
    
    weights_c = torch.tensor([np.sqrt(sum(counts_c.values())/(counts_c.get(i, 0)+1)) for i in range(6)]).float().to(device)
    weights_c = torch.clamp(weights_c, 1.0, 10.0) # Prevent extreme values

    # 2. Emotion Weights
    from config import EMOTION_MAP
    counts_e = collections.Counter(train_ds.df['Emotion'].str.lower())
    total_e = len(train_ds.df)
    weights_e = []
    for emo in EMOTION_MAP.keys():
        count = counts_e.get(emo, 1) 
        weights_e.append(np.sqrt(total_e / count))
    weights_e = torch.tensor(weights_e).float().to(device)
    weights_e = torch.clamp(weights_e, 1.0, 10.0) # Prevent extreme values

    criterion_e = nn.CrossEntropyLoss(weight=weights_e, label_smoothing=0.1)
    criterion_c = nn.CrossEntropyLoss(weight=weights_c, ignore_index=-1, label_smoothing=0.1)

    best_score = 0
    save_path = os.path.join(CONFIG['base_path'], CONFIG['model_save_path'])
    
    # History initialization
    history = {
        'train_loss': [],
        'train_emo_f1': [],
        'train_cause_f1': [],
        'val_emo_f1': [],
        'val_cause_f1': []
    }

    # Early Stopping variables
    patience_counter = 0
    best_loss = float('inf')

    print(f"Starting training on {device}...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        # Track Training Metrics
        train_preds, train_labels = [], []
        train_e_preds, train_e_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            ids, mask, audio = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['audio_vec'].to(device).float()
            lbl_e, lbl_c = batch['emotion_label'].to(device), batch['cause_label'].to(device)

            optimizer.zero_grad()
            out_e, out_c = model(ids, mask, audio)
            # Give slightly more weight to Cause (0.6) as it is the harder task
            loss = (0.4 * criterion_e(out_e, lbl_e)) + (0.6 * criterion_c(out_c, lbl_c))
            
            if torch.isnan(loss):
                print("   [Warning] NaN loss detected! Skipping batch.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Store Train Predictions
            with torch.no_grad():
                train_p_c = torch.argmax(out_c, dim=1)
                train_p_e = torch.argmax(out_e, dim=1)
                
                # Filter valid causes
                valid_train = lbl_c != -1
                if valid_train.sum() > 0:
                    train_preds.extend(train_p_c[valid_train].cpu().numpy())
                    train_labels.extend(lbl_c[valid_train].cpu().numpy())
                    
                train_e_preds.extend(train_p_e.cpu().numpy())
                train_e_labels.extend(lbl_e.cpu().numpy())

        # Validation logic
        model.eval()
        preds, labels = [], []
        e_preds, e_labels = [], []
        
        with torch.no_grad():
            for batch in dev_loader:
                lbl_c = batch['cause_label'].to(device)
                lbl_e = batch['emotion_label'].to(device)
                
                out_e, out_c = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['audio_vec'].to(device).float())
                
                valid = lbl_c != -1
                if valid.sum() > 0:
                    preds.extend(torch.argmax(out_c, dim=1)[valid].cpu().numpy())
                    labels.extend(lbl_c[valid].cpu().numpy())
                
                e_preds.extend(torch.argmax(out_e, dim=1).cpu().numpy())
                e_labels.extend(lbl_e.cpu().numpy())

        # Update History
        avg_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        print(f"\n📊 Epoch {epoch+1} Summary:")
        
        # 1. ACTUAL (TRAINING) SCORES
        if len(train_labels) > 0:
            tr_f1_c = f1_score(train_labels, train_preds, average='macro')
            tr_f1_e = f1_score(train_e_labels, train_e_preds, average='macro')
            
            history['train_cause_f1'].append(tr_f1_c)
            history['train_emo_f1'].append(tr_f1_e)
            
            print(f"   [Train] Cause F1: {tr_f1_c:.4f} || Emo F1: {tr_f1_e:.4f}")
            print(f"   [Train] Pred Dist: {dict(Counter(train_preds))}")
        
        # 2. VALIDATED (DEV) SCORES
        if len(labels) > 0:
            val_f1_c = f1_score(labels, preds, average='macro')
            val_f1_e = f1_score(e_labels, e_preds, average='macro')
            
            history['val_cause_f1'].append(val_f1_c)
            history['val_emo_f1'].append(val_f1_e)
            
            print(f"   [Val]   Cause F1: {val_f1_c:.4f} || Emo F1: {val_f1_e:.4f}")
            print(f"   [Val]   Pred Dist: {dict(Counter(preds))}")
            
            current_score = (0.5 * val_f1_c) + (0.5 * val_f1_e)
            if current_score > best_score:
                best_score = current_score
                torch.save(model.state_dict(), save_path)
                print(f"✅ New Best Model Saved (Weighted F1: {best_score:.4f} | Cause: {val_f1_c:.4f} | Emo: {val_f1_e:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"   [Info] Score did not improve (Patience: {patience_counter}/{CONFIG['early_stopping_patience']})")
            
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs!")
                break
        else:
             print(f"   [Val] No valid cause labels in dev set.")

    # --- PLOTTING ---
    import matplotlib.pyplot as plt
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(20, 6))

    # Plot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss', color='red')
    plt.title('Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Emotion F1 (Train vs Val)
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, [x*100 for x in history['train_emo_f1']], 'o-', label='Train Emotion F1', color='lightblue')
    plt.plot(epochs_range, [x*100 for x in history['val_emo_f1']], '^-', label='Val Emotion F1', color='blue', linewidth=2)
    plt.title('Emotion Recognition F1 (%)')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)

    # Plot 3: Cause F1 (Train vs Val)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, [x*100 for x in history['train_cause_f1']], 's-', label='Train Cause F1', color='lightgreen')
    plt.plot(epochs_range, [x*100 for x in history['val_cause_f1']], 's-', label='Val Cause F1', color='green', linewidth=2)
    plt.title('Cause Span F1-Score (%)')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['base_path'], 'training_results.png'), dpi=300, bbox_inches='tight')
    print("✅ Training Complete. Plot saved as training_results.png")
    plt.close()

if __name__ == "__main__":
    train()
