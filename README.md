# Emotion Cause Pair Extraction (ECPE) Project

This repository contains a unified multimodal framework for extracting emotion-cause pairs from conversational data. The project leverages both text and audio signals to identify not only the emotion expressed by a speaker but also the specific utterance that triggered that emotion.

## Project Overview

The goal of this project is to solve the **Emotion Cause Pair Extraction (ECPE)** task. Unlike simple emotion recognition, ECPE requires mapping an emotion to its underlying cause within a dialogue context.

### Key Features

- **Multimodal Fusion**: Combines RoBERTa-based text embeddings with Wav2Vec 2.0 audio features.
- **Contextual Understanding**: Analyzes previous utterances to identify causal relationships.
- **Deep Learning Pipeline**: End-to-end implementation including EDA, feature extraction, and model training.

## Dataset

The project utilizes the following datasets:

- **MELD (Multimodal Emotion Lines Dataset)**: For emotion labels and utterances.
- **ECAC (Emotion Cause Analysis in Conversations)**: For causal pair annotations.

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Librosa & MoviePy (for audio processing)
- Pandas, NumPy, Matplotlib, Seaborn

## Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd emotion-cause-deep-learning
   ```

2. **Mount Google Drive** (if running in Colab):
   The notebook is designed to run in a Google Colab environment with data stored in Google Drive. Ensure your `BASE_PATH` is correctly set.

3. **Data Preparation**:
   Place the MELD CSVs, ECAC JSONs, and audio/video splits in the designated data directory.

## Usage

Open the `emotion_cause_deep_learning (3).ipynb` notebook and run the cells sequentially:

1. **EDA**: Visualize causal lags and emotion distributions.
2. **Feature Extraction**: Extract audio features using the Wav2Vec 2.0 model.
3. **Training**: Train the `UnifiedMultimodalECPE` model.

## Model Architecture

The model uses a dual-encoder architecture:

- **Text Encoder**: RoBERTa-base
- **Audio Encoder**: Wav2Vec 2.0 (pre-computed features)
- **Classifier**: Multi-task heads for Emotion Classification and Causal Distance Prediction.

