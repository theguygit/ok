
import sys
import os

# Add project root to sys.path
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import librosa
import numpy as np
import pickle
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
from config import CONFIG

def extract_features(video_folder, output_pkl):
    print(f"Starting extraction from {video_folder}...")
    
    device = CONFIG['device']
    
    # Check if folder exists
    if not os.path.exists(video_folder):
        print(f"Warning: Folder {video_folder} does not exist.")
        return

    # Temp wav folder
    wav_folder = os.path.join(CONFIG['base_path'], 'temp_wavs')
    os.makedirs(wav_folder, exist_ok=True)

    print("Loading Wav2Vec 2.0 Model...")
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading Wav2Vec: {e}")
        return

    audio_features_dict = {}
    error_files = []

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} video files.")

    for video_file in tqdm(video_files):
        video_path = os.path.join(video_folder, video_file)
        wav_path = os.path.join(wav_folder, video_file.replace('.mp4', '.wav'))
        file_id = video_file.replace('.mp4', '')

        try:
            if not os.path.exists(wav_path):
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(wav_path, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
                video.close()

            audio_input, sr = librosa.load(wav_path, sr=16000, duration=6.0)

            input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
            input_values = input_values.to(device)

            with torch.no_grad():
                outputs = model(input_values)

            last_hidden_state = outputs.last_hidden_state
            pooled_output = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()

            audio_features_dict[file_id] = pooled_output

            # Cleanup
            if os.path.exists(wav_path):
                os.remove(wav_path)

        except Exception as e:
            # print(f"Error processing {video_file}: {e}")
            error_files.append(video_file)

    print(f"\nExtraction Complete! Processed {len(audio_features_dict)} files.")
    print(f"Errors: {len(error_files)}")

    print(f"Saving features to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(audio_features_dict, f)

if __name__ == "__main__":
    # 1. Training Set
    extract_features(
        os.path.join(CONFIG['base_path'], 'train_splits'), 
        os.path.join(CONFIG['base_path'], 'audio_features.pkl')
    )

    # 2. Development Set
    extract_features(
        os.path.join(CONFIG['base_path'], 'dev_splits_complete'), 
        os.path.join(CONFIG['base_path'], 'dev_audio_features.pkl')
    )

    # 3. Test Set
    extract_features(
        os.path.join(CONFIG['base_path'], 'test_splits'), 
        os.path.join(CONFIG['base_path'], 'audio_test.pkl')
    )
