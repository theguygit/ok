
import sys
import os

# Add project root to sys.path
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class DualStreamMECPE(nn.Module):
    def __init__(self, num_emotions=7, window_size=6):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # Unfreeze half of RoBERTa
        for layer in list(self.roberta.encoder.layer)[:6]:
            for param in layer.parameters(): param.requires_grad = False

        self.audio_fc = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Dropout(0.3))
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Bi-LSTM over the sequence
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        self.emotion_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_emotions))
        self.cause_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, window_size))

    def forward(self, input_ids, attention_mask, audio_vec):
        text_seq = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        audio_emb = self.audio_fc(audio_vec).unsqueeze(1)

        # Cross-Attention Fusion
        fused, _ = self.cross_attention(query=text_seq, key=audio_emb, value=audio_emb)
        fused = text_seq + fused # Residual connection

        # Temporal/Contextual Processing
        lstm_out, _ = self.lstm(fused)
        # Max-pool over sequence length
        feat, _ = torch.max(lstm_out, dim=1)

        return self.emotion_head(feat), self.cause_head(feat)
