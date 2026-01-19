
import sys
import os

# Add project root to sys.path
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from transformers import RobertaTokenizer
from config import CONFIG, EMOTION_MAP
from dataset import MECPEDataset

def run_eda():
    print("📊 Starting Exploratory Data Analysis (EDA)...")
    
    # Initialize Tokenizer (needed for Dataset)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Paths
    files = {
        'Train': (os.path.join(CONFIG['base_path'], 'train_sent_emo.csv'), 
                  os.path.join(CONFIG['base_path'], 'Subtask_2_train.json'),
                  os.path.join(CONFIG['base_path'], 'audio_features.pkl')),
        'Dev': (os.path.join(CONFIG['base_path'], 'dev_sent_emo.csv'), 
                os.path.join(CONFIG['base_path'], 'dev.json'),
                os.path.join(CONFIG['base_path'], 'dev_audio_features.pkl'))
    }
    
    for split_name, (csv_path, json_path, audio_path) in files.items():
        if not os.path.exists(csv_path) or not os.path.exists(json_path):
            print(f"⚠️ Skipping {split_name}: Files not found.")
            continue
            
        # Check audio
        if not os.path.exists(audio_path):
            print(f"⚠️ Warning: Audio features ({audio_path}) not found. Creating dummy for EDA.")
            # We create a dummy pickle just to load the dataset class, or patch it.
            # For now, let's just create a dummy dict file if missing? No, that's risky.
            # Let's assume user extracts features. If not, we fail gracefully.
            print("Please run src/feature_extraction.py first!")
            return

        print(f"\n--- Analyzing {split_name} Set ---")
        ds = MECPEDataset(csv_path, json_path, audio_path, tokenizer, CONFIG['max_len'])
        
        # 1. EMOTION DISTRIBUTION
        emotions = [row['Emotion'] for _, row in ds.df.iterrows()]
        plt.figure(figsize=(10, 5))
        sns.countplot(y=emotions, order=ds.df['Emotion'].value_counts().index, palette='viridis')
        plt.title(f'{split_name} - Emotion Distribution')
        save_path = os.path.join(CONFIG['base_path'], f'eda_{split_name.lower()}_emotion_dist.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
        # 2. CAUSE DISTRIBUTION
        # We need to iterate the dataset to get the robustly mapped labels
        cause_labels = []
        for i in range(len(ds)):
            l = ds[i]['cause_label'].item()
            if l != -1:
                cause_labels.append(l)
                
        if cause_labels:
            plt.figure(figsize=(8, 5))
            # countplot for integers
            sns.countplot(x=cause_labels, palette='magma')
            plt.title(f'{split_name} - Causal Lag Distribution (0=Self, 1=Prev...)')
            plt.xlabel("Lag Distance")
            plt.ylabel("Count")
            save_path = os.path.join(CONFIG['base_path'], f'eda_{split_name.lower()}_cause_dist.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
            plt.close()
            
            print(f"Total Valid Cause Labels: {len(cause_labels)}")
            print(f"Cause Distribution: {dict(Counter(cause_labels))}")
        else:
            print("❌ No valid cause labels found! Check mapping logic.")

        # 3. TEXT LENGTH
        lengths = [len(str(r['Utterance']).split()) for _, r in ds.df.iterrows()]
        plt.figure(figsize=(10, 5))
        sns.histplot(lengths, bins=30, kde=True, color='skyblue')
        plt.title(f'{split_name} - Utterance Length Distribution')
        plt.xlabel("Number of Words")
        save_path = os.path.join(CONFIG['base_path'], f'eda_{split_name.lower()}_text_length.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

        # 4. WORD CLOUD (Weighted by Emotion)
        try:
            from wordcloud import WordCloud
            print("Generating Word Clouds...")
            # Combine all text
            all_text = " ".join([str(t) for t in ds.df['Utterance']])
            wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{split_name} - Most Frequent Words')
            save_path = os.path.join(CONFIG['base_path'], f'eda_{split_name.lower()}_wordcloud.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
            plt.close()
        except ImportError:
            print("Skipping WordCloud (library not installed).")

        # 5. CONTEXTUAL ANALYSIS (Emotion Transitions)
        # Verify if previous emotion predicts current emotion
        print("Analyzing Emotion Transitions...")
        transitions = []
        df = ds.df
        for i in range(1, len(df)):
            if df.iloc[i]['Dialogue_ID'] == df.iloc[i-1]['Dialogue_ID']:
                prev = df.iloc[i-1]['Emotion']
                curr = df.iloc[i]['Emotion']
                transitions.append(f"{prev} -> {curr}")
        
        if transitions:
            top_trans = Counter(transitions).most_common(10)
            plt.figure(figsize=(12, 6))
            path_labels, path_counts = zip(*top_trans)
            sns.barplot(x=list(path_counts), y=list(path_labels), palette='magma')
            plt.title(f'{split_name} - Top 10 Emotion Transitions')
            plt.xlabel("Count")
            save_path = os.path.join(CONFIG['base_path'], f'eda_{split_name.lower()}_transitions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
            plt.close()

if __name__ == "__main__":
    run_eda()
