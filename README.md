# Multimodal Emotion Cause Pair Extraction (MECPE)

Deep learning model for emotion recognition and causal span extraction from multimodal conversational data (text + audio).

## Project Structure

```
.
├── config.py                  # Configuration and hyperparameters
├── dataset.py                 # Dataset class with robust ID mapping
├── model.py                   # DualStreamMECPE architecture
├── feature_extraction.py      # Audio feature extraction
├── eda.py                     # Exploratory Data Analysis
├── train.py                   # Training script
├── evaluate.py                # Validation set evaluation
├── test.py                    # Test set inference
├── app.py                     # Gradio dashboard
├── train_sent_emo.csv
├── dev_sent_emo.csv
├── test_sent_emo.csv
├── Subtask_2_train.json
├── dev.json
├── test.json
├── train_splits/              # Training video clips
├── dev_splits_complete/       # Dev video clips
├── test_splits/               # Test video clips
└── requirements.txt
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install --force-reinstall -r requirements.txt
```

(if you face dependency issues)

### 2. Install Additional Libraries (if needed)

```bash
pip install wordcloud
```

## Execution Order

Follow these steps **in order** to run the complete pipeline:

---

### **Step 1: Extract Audio Features** ⏱️ (~30-60 minutes)

Extract Wav2Vec2 features from video clips for all datasets.

```bash
python feature_extraction.py
```

**Output:**

- `audio_features.pkl` (Training set)
- `dev_audio_features.pkl` (Dev set)
- `audio_test.pkl` (Test set)

---

### **Step 2: Exploratory Data Analysis (Optional)** 📊

Visualize data distributions and verify the dataset alignment.

```bash
python eda.py
```

**Output:** PNG files saved to project root:

- `eda_train_emotion_dist.png`
- `eda_train_cause_dist.png`
- `eda_train_text_length.png`
- `eda_train_wordcloud.png`
- `eda_train_transitions.png`
- (Same for Dev set)

---

### **Step 3: Train the Model** 🚀 (~2-4 hours on GPU)

Train the DualStreamMECPE model with differential learning rates and class weighting.

```bash
python train.py
```

**Output:**

- `best_model.pth` (Best model based on validation Cause F1)
- `training_results.png` (Training curves)

**What it does:**

- Loads training and dev datasets
- Trains for 15 epochs with early stopping
- Saves best model based on validation Cause F1 score
- Displays training and validation metrics per epoch

---

### **Step 4: Evaluate on Validation Set** 📈

Generate detailed metrics and confusion matrices on the validation set.

```bash
python evaluate.py
```

**Output:**

- PNG files: `emotion_cm_(val).png`, `cause_cm_(val).png`
- Classification reports printed to console
- Plot windows will popup (closes on exit)

---

### **Step 5: Test Set Inference** 🎯

Run final inference on the test set.

```bash
python test.py
```

**Output:**

- Final test metrics printed to console
- PNG files: `emotion_cm_(test_set).png`, `cause_cm_(test_set).png`
- Emotion and Cause F1 scores
- Combined F1 score

---

### **Step 6: Launch Interactive Dashboard (Optional)** 🎧

Run the Gradio app for live emotion and cause prediction.

```bash
python app.py
```

**Features:**

- Upload audio or record from microphone
- Automatic speech transcription (Whisper)
- Real-time emotion and cause prediction
- Confidence scores visualization

---

## Configuration

Edit `config.py` to modify:

- `epochs`: Number of training epochs (default: 50)
- `lr`: Base learning rate for RoBERTa (default: 5e-6)
- `head_lr`: Learning rate for classification heads (default: 1e-5)
- `batch_size`: Training batch size (default: 32)
- `max_len`: Maximum sequence length (default: 160)
- `weight_decay`: L2 regularization (default: 0.05)
- `base_path`: Root directory for data (default: `e:/dlcw`)

## Model Architecture

**DualStreamMECPE** combines:

1. **RoBERTa-base** for text encoding
2. **Wav2Vec2** for audio feature extraction
3. **Cross-Attention** for multimodal fusion
4. **Bi-LSTM** for temporal context modeling
5. **Dual Classification Heads** for emotion and cause prediction

## Key Features

✅ **Robust Text-Based Alignment**: Fixes ID mismatches between CSV and JSON files  
✅ **Differential Learning Rates**: Slow updates for RoBERTa, fast for heads  
✅ **Class Weighting**: Handles imbalanced cause labels  
✅ **Comprehensive Metrics**: F1, Accuracy for both tasks  
✅ **Training Visualization**: Loss curves and performance plots  
✅ **Production Ready**: Gradio dashboard for deployment  

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'src'`, the scripts automatically add the project root to `sys.path`. Ensure you're running from the correct directory.

### Audio Extraction Fails

- Install `ffmpeg`: Required by `moviepy` for audio extraction
- Check video file paths in `train_splits/`, `dev_splits_complete/`, `test_splits/`

### CUDA Out of Memory

- Reduce `batch_size` in `config.py`
- Use CPU by setting `device = torch.device("cpu")` in config

## Citation

If you use this code, please cite the original MECPE dataset paper and relevant model architectures (RoBERTa, Wav2Vec2).

## License

This project is for research purposes only.
