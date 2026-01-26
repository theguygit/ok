import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from config import CONFIG
from dataset import MECPEDataset
from evaluate import load_model_for_eval, evaluate_set

def run_test_inference():
    print(f"--- 🧪 Running Final Inference (Test Set) ---")
    device = CONFIG['device']
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_ds = MECPEDataset(os.path.join(CONFIG['base_path'], 'test_sent_emo.csv'), os.path.join(CONFIG['base_path'], 'test.json'), os.path.join(CONFIG['base_path'], 'audio_test.pkl'), tokenizer, CONFIG['max_len'])
    model = load_model_for_eval(device)
    evaluate_set(DataLoader(test_ds, batch_size=CONFIG['batch_size']), model, device, title_suffix="(Test Set)")

if __name__ == "__main__":
    run_test_inference()
