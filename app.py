import os
import sys

gradio_temp = os.path.join(os.getcwd(), ".gradio_temp")
os.makedirs(gradio_temp, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = gradio_temp
os.environ["PYTHONIOENCODING"] = "utf-8"

import gradio as gr
import torch
import librosa
import whisper
import numpy as np
import subprocess
from transformers import Wav2Vec2Processor, Wav2Vec2Model, RobertaTokenizer
from config import CONFIG, EMOTION_MAP
from model import DualStreamMECPE

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    print("✅ FFmpeg initialized")
except ImportError:
    pass

print(f"⏳ Loading Models (Audio Cache: {gradio_temp})...")
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

def transcribe_audio(audio_path):
    if not audio_path: return ""
    try:
        print(f"🎙️ Transcribing: {audio_path}")
        result = transcriber.transcribe(audio_path, fp16=torch.cuda.is_available())
        return result["text"].strip()
    except Exception as e:
        print(f"❌ ASR Error: {e}")
        return f"ASR Error: {e}"

def process_call(audio_path, text_input):
    if audio_path and (not text_input or text_input.strip() == ""):
        print("💡 Text empty - Auto-transcribing audio...")
        text_input = transcribe_audio(audio_path)
    
    print(f"⚙️ Processing - Text: {len(text_input)} chars, Audio: {audio_path is not None}")
    if not audio_path and not text_input: 
        return "⚠️ Please provide audio or text.", {}, ""

    try:
        if audio_path:
            y, sr = librosa.load(audio_path, sr=16000, duration=15.0)
            input_vals = audio_processor(y, sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(DEVICE)
            with torch.no_grad(): 
                audio_vec = torch.mean(audio_model_feat(input_vals).last_hidden_state, dim=1)
        else: 
            audio_vec = torch.zeros((1, 768)).to(DEVICE)

        ti = tokenizer(text_input, max_length=CONFIG['max_len'], padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)
        
        with torch.no_grad():
            _, out_e_metric, out_c = model(ti['input_ids'], ti['attention_mask'], audio_vec)
            p_e = torch.nn.functional.softmax(out_e_metric, dim=1)[0]
            p_c = torch.sigmoid(out_c)[0]
            causes = [f"Turn {i} ({p_c[i]:.0%})" for i in range(6) if p_c[i] > 0.35]

        emo_idx = torch.argmax(p_e).item()
        confidence = p_e[emo_idx].item()
        
        report = f"### 📊 Analysis Results\n\n**Detected Emotion:** `{EMOTIONS[emo_idx].upper()}` ({confidence:.1%})\n\n**Probable Causes:** {', '.join(causes) if causes else '_None detected._'}"
        labels = {EMOTIONS[i]: float(p_e[i]) for i in range(7)}
        
        return report, labels, text_input
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        return f"⚠️ **Analysis Error:** {e}", {}, text_input

def launch_app():
    with gr.Blocks(title="AI Call Center") as demo:
        gr.Markdown("# 🎧 AI Call Center Dashboard")
        gr.Markdown("Real-time Customer Sentiment & Cause Extraction System")
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="1. Customer Voice")
                transcribe_btn = gr.Button("📝 Extract Transcript (Optional)", variant="secondary")
                text_input = gr.Textbox(label="2. Transcript (Model will use this)", lines=4, placeholder="Transcription will appear here...")
                with gr.Row():
                    analyze_btn = gr.Button("🚀 ANALYZE INTERACTION", variant="primary", scale=2)
                    reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                output_md = gr.Markdown("### 🔍 Analysis Output\n_Waiting for input..._")
                output_chart = gr.Label(label="Emotion Confidence Scores")

        transcribe_btn.click(transcribe_audio, inputs=[audio_input], outputs=[text_input])
        
        analyze_btn.click(
            process_call, 
            inputs=[audio_input, text_input], 
            outputs=[output_md, output_chart, text_input]
        ).then(
            lambda: (None, ""),
            outputs=[audio_input, text_input]
        )
        
        reset_btn.click(
            lambda: (None, "", "### 🔍 Analysis Output\n_Waiting for input..._", {}),
            outputs=[audio_input, text_input, output_md, output_chart]
        )

    print(f"📂 Audio cache initialized in: {gradio_temp}")
    demo.queue() 
    demo.launch(share=False, debug=False, theme=gr.themes.Soft())

if __name__ == "__main__": launch_app()

