
import sys
import os

# Add project root to sys.path so 'src' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os

# Add project root to sys.path so 'src' module can be found
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import json
import numpy as np
from config import EMOTION_MAP

class MECPEDataset(Dataset):
    def __init__(self, csv_path, json_path, audio_pkl_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = pd.read_csv(csv_path)

        with open(audio_pkl_path, 'rb') as f:
            self.audio_data = pickle.load(f)

        with open(json_path, 'r') as f:
            self.cause_data = json.load(f)

        self.cause_map = {}
        mapped_count = 0
        
        # --- ROBUST TEXT-BASED ALIGNMENT ---
        # 1. Group CSV conversations by Dialogue_ID
        csv_convs = {}
        for idx, row in self.df.iterrows():
            did = row['Dialogue_ID']
            if did not in csv_convs: csv_convs[did] = []
            text = str(row['Utterance']).strip().lower()
            text = "".join([c for c in text if c.isalnum()])
            csv_convs[did].append( (row['Utterance_ID'], text) )
            
        # Map first utterance text to list of Match Candidates (Dialogue IDs)
        first_utt_map = {}
        for did, utts in csv_convs.items():
            if utts:
                # Sort by Utterance_ID to ensure order
                utts.sort(key=lambda x: x[0])
                first_text = utts[0][1]
                # Use slice
                k = first_text[:50]
                if k not in first_utt_map: first_utt_map[k] = []
                first_utt_map[k].append(did)

        # 2. Iterate JSON and find matches in CSV
        for item in self.cause_data:
            json_utts = item['conversation']
            if not json_utts: continue
            
            # Normalize first text
            j_text0 = str(json_utts[0]['text']).strip().lower()
            j_text0 = "".join([c for c in j_text0 if c.isalnum()])
            
            candidates = first_utt_map.get(j_text0[:50], [])
            
            matched_did = None
            for cand_did in candidates:
                # Verify match with more utterances
                csv_u = csv_convs[cand_did]
                match_count = 0
                check_len = min(len(json_utts), len(csv_u))
                if check_len == 0: continue
                
                for i in range(check_len):
                    ct = csv_u[i][1]
                    jt = str(json_utts[i]['text']).strip().lower()
                    jt = "".join([c for c in jt if c.isalnum()])
                    if ct == jt:
                        match_count += 1
                
                if match_count / check_len > 0.5: # >50% overlap
                    matched_did = cand_did
                    break
            
            if matched_did is not None:
                # 3. Map JSON Utterance IDs to CSV Utterance IDs
                csv_u_list = csv_convs[matched_did] # list of (csv_uid, text)
                
                j_id_to_c_id = {}
                for i in range(min(len(json_utts), len(csv_u_list))):
                    j_uid = json_utts[i]['utterance_ID']
                    c_uid = csv_u_list[i][0]
                    j_id_to_c_id[j_uid] = c_uid
                
                if 'emotion-cause_pairs' in item:
                    for pair in item['emotion-cause_pairs']:
                        try:
                            # pair format: ["3_joy", "2"]
                            e_raw = pair[0].split('_')[0]
                            c_raw = pair[1]
                            
                            e_json_id = int(e_raw)
                            c_json_id = int(c_raw)
                            
                            # Get CSV IDs
                            e_csv_id = j_id_to_c_id.get(e_json_id)
                            c_csv_id = j_id_to_c_id.get(c_json_id)
                            
                            if e_csv_id is not None and c_csv_id is not None:
                                # Calculate Dist
                                e_idx = -1
                                c_idx = -1
                                for idx, u in enumerate(json_utts):
                                    if u['utterance_ID'] == e_json_id: e_idx = idx
                                    if u['utterance_ID'] == c_json_id: c_idx = idx
                                
                                # e_idx and c_idx are indices in the conversation list (0, 1, 2...)
                                # We check lag based on position
                                if e_idx != -1 and c_idx != -1:
                                    dist = e_idx - c_idx
                                    if 0 <= dist <= 5:
                                        key = f"dia{matched_did}_utt{e_csv_id}"
                                        self.cause_map[key] = dist
                                        mapped_count += 1
                        except: pass
        
        print(f"✅ Dataset Loaded: Mapped {mapped_count} cause labels via matched IDs.")

    def __len__(self): return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        dia_id, utt_id = row['Dialogue_ID'], row['Utterance_ID']

        # CONTEXT WINDOW (Look back 2 turns)
        context = []
        for i in [2, 1]:
            prev_idx = index - i
            if prev_idx >= 0:
                prev_row = self.df.iloc[prev_idx]
                if prev_row['Dialogue_ID'] == dia_id:
                    context.append(str(prev_row['Utterance']))

        full_text = f" {self.tokenizer.sep_token} ".join(context + [str(row['Utterance'])])
        inputs = self.tokenizer(full_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

        unique_id = f"dia{dia_id}_utt{utt_id}"
        audio_vec = torch.tensor(self.audio_data.get(unique_id, np.zeros(768)), dtype=torch.float32)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'audio_vec': audio_vec,
            'emotion_label': torch.tensor(EMOTION_MAP.get(row['Emotion'].lower(), 0), dtype=torch.long),
            'cause_label': torch.tensor(self.cause_map.get(unique_id, -1), dtype=torch.long)
        }
