import os
import gradio as gr
import torch
import librosa
import whisper
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model, RobertaTokenizer
from config import CONFIG, EMOTION_MAP
from model import DualStreamMECPE

print("⏳ Loading Models...")
DEVICE = CONFIG['device']
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', clean_up_tokenization_spaces=True)
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model_feat = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()
transcriber = whisper.load_model("base")

model = DualStreamMECPE()
path = os.path.join(CONFIG['base_path'], CONFIG['model_save_path'])
if os.path.exists(path):
    print(f"✅ Loading: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
else:
    print(f"⚠️ Warning: Model not found at {path}")
model.to(DEVICE).eval()

EMOTIONS = list(EMOTION_MAP.keys())

def process_call(audio_path, text_input):
    if not audio_path and not text_input: return "⚠️ Please provide audio/text.", {}, ""
    if not text_input and audio_path:
        try: text_input = transcriber.transcribe(audio_path)["text"].strip()
        except: text_input = "(ASR Failed)"

    try:
        if audio_path:
            y, sr = librosa.load(audio_path, sr=16000, duration=6.0)
            input_vals = audio_processor(y, sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(DEVICE)
            with torch.no_grad(): audio_vec = torch.mean(audio_model_feat(input_vals).last_hidden_state, dim=1)
        else: audio_vec = torch.zeros((1, 768)).to(DEVICE)

        ti = tokenizer(text_input, max_length=CONFIG['max_len'], padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _, out_e_metric, out_c = model(ti['input_ids'], ti['attention_mask'], audio_vec)
            p_e = torch.nn.functional.softmax(out_e_metric, dim=1)[0]
            p_c = torch.sigmoid(out_c)[0]
            causes = [f"Lag {i} ({p_c[i]:.0%})" for i in range(6) if p_c[i] > 0.4]

        report = f"### 分析结果\n\n**Emotion:** {EMOTIONS[torch.argmax(p_e).item()].upper()} ({torch.max(p_e).item():.1%})\n\n**Causes:** {', '.join(causes) if causes else 'None'}"
        return report, {EMOTIONS[i]: float(p_e[i]) for i in range(7)}, text_input
    except Exception as e: return f"Error: {e}", {}, text_input

def launch_app():
    with gr.Blocks(title="AI Call Center", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎧 AI Call Center Dashboard")
        with gr.Row():
            audio = gr.Audio(type="filepath", label="Voice Input")
            text = gr.Textbox(label="Transcript / Text Input")
        btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            output = gr.Markdown()
            chart = gr.Label(label="Emotion Confidence")
        btn.click(process_call, [audio, text], [output, chart, text])
    demo.launch(share=True)

if __name__ == "__main__": launch_app()
