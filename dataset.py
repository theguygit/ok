import sys
import os
import pandas as pd
import pickle
import json
import numpy as np
import torch
from torch.utils.data import Dataset
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
        
        csv_convs = {}
        for idx, row in self.df.iterrows():
            did = row['Dialogue_ID']
            if did not in csv_convs: csv_convs[did] = []
            text = "".join([c for c in str(row['Utterance']).strip().lower() if c.isalnum()])
            csv_convs[did].append( (row['Utterance_ID'], text) )
            
        first_utt_map = {}
        for did, utts in csv_convs.items():
            if utts:
                utts.sort(key=lambda x: x[0])
                k = utts[0][1][:50]
                if k not in first_utt_map: first_utt_map[k] = []
                first_utt_map[k].append(did)

        for item in self.cause_data:
            json_utts = item['conversation']
            if not json_utts: continue
            
            j_text0 = "".join([c for c in str(json_utts[0]['text']).strip().lower() if c.isalnum()])
            candidates = first_utt_map.get(j_text0[:50], [])
            
            matched_did = None
            for cand_did in candidates:
                csv_u = csv_convs[cand_did]
                match_count = 0
                check_len = min(len(json_utts), len(csv_u))
                if check_len == 0: continue
                
                for i in range(check_len):
                    ct = csv_u[i][1]
                    jt = "".join([c for c in str(json_utts[i]['text']).strip().lower() if c.isalnum()])
                    if ct == jt: match_count += 1
                
                if match_count / check_len > 0.5:
                    matched_did = cand_did
                    break
            
            if matched_did is not None:
                csv_u_list = csv_convs[matched_did]
                j_id_to_c_id = {u['utterance_ID']: csv_u_list[i][0] for i, u in enumerate(json_utts) if i < len(csv_u_list)}
                
                if 'emotion-cause_pairs' in item:
                    for pair in item['emotion-cause_pairs']:
                        try:
                            e_json_id = int(pair[0].split('_')[0])
                            c_json_id = int(pair[1])
                            e_csv_id = j_id_to_c_id.get(e_json_id)
                            c_csv_id = j_id_to_c_id.get(c_json_id)
                            
                            if e_csv_id is not None and c_csv_id is not None:
                                e_idx = next(i for i, u in enumerate(json_utts) if u['utterance_ID'] == e_json_id)
                                c_idx = next(i for i, u in enumerate(json_utts) if u['utterance_ID'] == c_json_id)
                                dist = e_idx - c_idx
                                if 0 <= dist <= 5:
                                    key = f"dia{matched_did}_utt{e_csv_id}"
                                    if key not in self.cause_map:
                                        self.cause_map[key] = np.zeros(6, dtype=np.float32)
                                    self.cause_map[key][dist] = 1.0
                                    mapped_count += 1
                        except: pass
        
        print(f"Dataset Loaded: Mapped {mapped_count} labels.")

    def __len__(self): return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        dia_id, utt_id = row['Dialogue_ID'], row['Utterance_ID']

        context = []
        for i in range(5, 0, -1):
            prev_idx = index - i
            if prev_idx >= 0 and self.df.iloc[prev_idx]['Dialogue_ID'] == dia_id:
                context.append(str(self.df.iloc[prev_idx]['Utterance']))

        full_text = f" {self.tokenizer.sep_token} ".join(context + [str(row['Utterance'])])
        inputs = self.tokenizer(full_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

        unique_id = f"dia{dia_id}_utt{utt_id}"
        audio_vec = torch.tensor(self.audio_data.get(unique_id, np.zeros(768)), dtype=torch.float32)
        cause_label = torch.tensor(self.cause_map.get(unique_id, np.zeros(6)), dtype=torch.float32)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'audio_vec': audio_vec,
            'emotion_label': torch.tensor(EMOTION_MAP.get(row['Emotion'].lower(), 0), dtype=torch.long),
            'cause_label': cause_label,
            'has_cause': torch.tensor(1.0 if unique_id in self.cause_map else 0.0, dtype=torch.float32)
        }
