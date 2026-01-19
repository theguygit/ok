
import sys
import os

# Add project root to sys.path
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
from config import CONFIG, EMOTION_MAP
from dataset import MECPEDataset
from evaluate import load_model_for_eval, plot_cm
from transformers import RobertaTokenizer

def run_test_inference():
    print(f"🚀 Running Inference on TEST Data...")
    device = CONFIG['device']
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Check if test audio exists, otherwise warn
    test_audio_path = os.path.join(CONFIG['base_path'], 'audio_test.pkl')
    if not os.path.exists(test_audio_path):
        print(f"❌ Test audio features not found at {test_audio_path}. calculating...")
        # In a real scenario we would call feature_extraction here, but let's assume it exists or fail
    
    test_ds = MECPEDataset(os.path.join(CONFIG['base_path'], 'test_sent_emo.csv'),
                           os.path.join(CONFIG['base_path'], 'Subtask_2_test.json'), # Is this filename correct in your dir?
                           test_audio_path, tokenizer, CONFIG['max_len'])
    
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = load_model_for_eval(device)
    
    all_preds_e, all_labels_e = [], []
    all_preds_c, all_labels_c = [], []
    
    with torch.no_grad():
        for batch in test_loader:
             ids = batch['input_ids'].to(device)
             mask = batch['attention_mask'].to(device)
             audio = batch['audio_vec'].to(device).float()
             
             out_e, out_c = model(ids, mask, audio)
             
             p_e = torch.argmax(out_e, dim=1)
             p_c = torch.argmax(out_c, dim=1)
             
             all_preds_e.extend(p_e.cpu().numpy())
             all_labels_e.extend(batch['emotion_label'].numpy())
             
             lbl_c = batch['cause_label']
             valid = lbl_c != -1
             if valid.sum() > 0:
                 all_preds_c.extend(p_c[valid].cpu().numpy())
                 all_labels_c.extend(lbl_c[valid].numpy())
                 
    # Results
    print("\n" + "🏆"*15)
    print("FINAL TEST RESULTS")
    print("🏆"*15)
    
    emo_acc = accuracy_score(all_labels_e, all_preds_e)
    emo_f1 = f1_score(all_labels_e, all_preds_e, average='macro')
    print(f"Emotion Acc: {emo_acc:.2%}, F1: {emo_f1:.4f}")
    
    if len(all_labels_c) > 0:
        cause_acc = accuracy_score(all_labels_c, all_preds_c)
        cause_f1 = f1_score(all_labels_c, all_preds_c, average='macro')
        print(f"Cause Acc: {cause_acc:.2%}, F1: {cause_f1:.4f}")
        print(f"⭐ Combined F1: {(emo_f1+cause_f1)/2:.4f}")
    else:
        print("Cause Acc: N/A (Blind Set)")

if __name__ == "__main__":
    run_test_inference()
