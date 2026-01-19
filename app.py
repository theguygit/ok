
import sys
import os

# Add project root to sys.path
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import torch
import librosa
import soundfile as sf
import whisper
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model, RobertaTokenizer
from config import CONFIG, EMOTION_MAP
from model import DualStreamMECPE

# Initialize Models
print("⏳ Loading Models...")
DEVICE = CONFIG['device']
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model_feat = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
transcriber = whisper.load_model("base")

model = DualStreamMECPE(num_emotions=7, window_size=6)
path = os.path.join(CONFIG['base_path'], CONFIG['model_save_path'])
if os.path.exists(path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
else:
    print(f"⚠️ Model not found at {path}")

model.to(DEVICE)
model.eval()

EMOTIONS = list(EMOTION_MAP.keys())

def process_call(audio_path, text_input):
    if audio_path is None:
        return "⚠️ Error: Please speak or upload audio.", {}, ""

    # ASR
    if text_input is None or text_input.strip() == "":
        try:
            result = transcriber.transcribe(audio_path)
            text_input = result["text"].strip()
        except Exception as e:
            text_input = "(Transcription Failed)"
            print(f"ASR Error: {e}")

    # Features
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=6.0)
        inputs = audio_processor(y, sampling_rate=16000, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.to(DEVICE)
        with torch.no_grad():
            outputs = audio_model_feat(input_values)
            audio_vec = torch.mean(outputs.last_hidden_state, dim=1)

        text_inputs = tokenizer(text_input, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
        ids = text_inputs['input_ids'].to(DEVICE)
        mask = text_inputs['attention_mask'].to(DEVICE)

        with torch.no_grad():
            out_e, out_c = model(ids, mask, audio_vec)
            probs_emo = torch.nn.functional.softmax(out_e, dim=1)[0]
            pred_emo_idx = torch.argmax(probs_emo).item()
            pred_cause_idx = torch.argmax(out_c, dim=1).item()

        emo_label = EMOTIONS[pred_emo_idx]
        emo_conf = probs_emo[pred_emo_idx].item()
        
        cause_text = f"Lag {pred_cause_idx}"
        report = f"**Emotion:** {emo_label.upper()} ({emo_conf:.1%})\n\n**Cause:** {cause_text}"
        confidences = {EMOTIONS[i]: float(probs_emo[i]) for i in range(len(EMOTIONS))}
        
        return report, confidences, text_input

    except Exception as e:
        return f"Error: {e}", {}, text_input

def launch_app():
    with gr.Blocks(title="AI Call Center", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎧 AI Call Center Dashboard")
        with gr.Row():
            audio = gr.Audio(type="filepath", sources=["microphone", "upload"])
            text = gr.Textbox(label="Transcript")
        btn = gr.Button("Analyze")
        output = gr.Markdown()
        chart = gr.Label()
        
        btn.click(process_call, inputs=[audio, text], outputs=[output, chart, text])
    
    demo.launch(share=True)

if __name__ == "__main__":
    launch_app()
