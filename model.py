import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), 1e-9, 1.0))
        phi = torch.where(cosine > self.th, cosine * self.cos_m - sine * self.sin_m, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

class FusionBlock(nn.Module):
    def __init__(self, text_dim, audio_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim + audio_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
    def forward(self, text, audio):
        audio = audio.expand(-1, text.size(1), -1)
        return self.proj(torch.cat([text, audio], dim=-1))

class DualStreamMECPE(nn.Module):
    def __init__(self, num_emotions=7, window_size=6):
        super().__init__()
        config = RobertaConfig.from_pretrained('roberta-base')
        config.hidden_dropout_prob = config.attention_probs_dropout_prob = 0.2
        self.roberta = RobertaModel.from_pretrained('roberta-base', config=config)
        for param in self.roberta.embeddings.parameters(): param.requires_grad = False

        self.audio_fc = nn.Sequential(nn.Linear(768, 768), nn.LayerNorm(768), nn.ReLU(), nn.Dropout(0.2))
        self.fusion = FusionBlock(768, 768, 768)
        self.bi_lstm = nn.LSTM(768, 256, num_layers=1, batch_first=True, bidirectional=True)
        
        self.feat_dim = 512
        self.emotion_arcface = ArcMarginProduct(self.feat_dim, num_emotions, s=64.0, m=0.5)
        self.cause_head = nn.Sequential(nn.Linear(self.feat_dim, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, window_size))

    def forward(self, input_ids, attention_mask, audio_vec, emotion_label=None):
        text_seq = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        audio_emb = self.audio_fc(audio_vec).unsqueeze(1)
        fused = self.fusion(text_seq, audio_emb)
        lstm_out, _ = self.bi_lstm(fused)
        feat, _ = torch.max(lstm_out, dim=1)

        out_e_metric = F.linear(F.normalize(feat), F.normalize(self.emotion_arcface.weight)) * self.emotion_arcface.s
        out_e = self.emotion_arcface(feat, emotion_label) if (self.training and emotion_label is not None) else out_e_metric
        return out_e, out_e_metric, self.cause_head(feat)
