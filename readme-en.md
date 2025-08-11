<div align="center">
  <h1 style="font-size:3em; font-weight:bold; letter-spacing:0.1em;">
    ✨🌐 <span style="color:#0078D4; text-shadow: 2px 2px 8px #00bfff;">ENS2ZHT</span> 🌐✨
  </h1>
  <img src="https://img.shields.io/badge/Enterprise%20Speech%20Translation-blue.svg?style=for-the-badge" alt="badge" />
</div>

<div align="center">
  <a href="./readme.md" style="font-size:1.2em; font-weight:bold;">🌏 中文</a>
</div>


🌏 Enterprise-level English Speech to Chinese Text Model

Welcome to ENS2ZHT! This project is an end-to-end English speech to Chinese text system based on the PyTorch framework, designed for enterprise scenarios.

🚀 **Technical Architecture**:
* Audio feature extraction uses Wav2Vec2 (facebook/wav2vec2-base-960h) and Wav2Vec2Processor, supporting multiple audio formats.
* Text processing uses ChineseBertTokenizer for tokenization and semantic encoding.
* The main model is a single large Transformer (12 encoder layers + 12 decoder layers, d_model=768, nhead=8, dim_feedforward=2048), enabling long-sequence modeling.
* The output layer consists of three Linear+ReLU layers, final output is a 768-dimensional vector.
* Loss function: CosineEmbeddingLoss; Optimizer: Adam.
* Inference uses autoregressive generation, token decoding via ChineseBertTokenizer.
* Supports batch training data generation and disk caching for large-scale training.

💾 **Data & Training**:
* Supports LibriSpeech format data, automatically generates aligned English-Chinese training sets.
* Training uses hard disk caching to greatly reduce memory usage, enabling large-scale datasets.
* Supports CPU, MPS (Apple Silicon), CUDA devices.

🔒 **Features & Performance**:
* Automatically transcribes English speech to Chinese text, supporting batch and real-time inference.
* Initial tests show the model achieves end-to-end speech translation on standard datasets with strong generalization.
* Extensible structure for custom pre/post-processing modules.

🎯 **Application Scenarios**:
* Customer service transcription
* Meeting archiving
* Smart subtitle generation
* Cross-border e-commerce speech translation

For detailed algorithm description, evaluation results, or custom features, please refer to the source code or open an issue.

📦 **Quick Start**:
1. Install dependencies:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Download tokenizer:
```
python Model/load_tokenizer.py
```
3. Prepare your data: Place English speech files in the specified directory
4. Run the training script and experience the efficiency and low cost of hard disk caching

✨ **More Features**:
* Supports multiple audio formats
* Extensible model architecture
* Detailed logging and monitoring

---

## 📚 Usage Guide

This project automates three main workflows: dataset generation, model training, and inference (translation), making enterprise integration easy.

### 1️⃣ Generate Training Dataset

Organize your English speech and text data in [LibriSpeech](https://www.openslr.org/12) format (e.g., `test-clean` folder), with `.flac` audio and `.trans.txt` text files in each subfolder.  
Use the multithreaded script to automatically generate training data with Chinese translations:

```bash
python Model/load-datasetM.py --fildir Model/test-clean --threads 24
```

- `--fildir`: Source data directory (e.g., `Model/test-clean`)
- `--threads`: Number of concurrent threads (recommended 8~24, adjust for API rate limits and local performance)

The script calls the API to translate English text to Chinese and packages audio and bilingual text into a JSON file in `Model/data/`.

### 2️⃣ Train the Model

Once your dataset is ready, train the model with:

```bash
python Model/train.py --dataset test-clean --batch_size 32 --epoches 10000 --device mps
```

- `--dataset`: Dataset name (e.g., `test-clean`, corresponding to `Model/data/test-clean.json`)
- `--batch_size`: Training batch size
- `--epoches`: Number of training epochs
- `--device`: Training device (`cpu`, `mps`, `cuda` supported)

During training, data is cached to disk, minimizing memory usage. Model parameters are saved to `Model/pth/en2zh_model.pth` after each epoch.

#### 🏷️ Command-line One-click Translation
You can directly use the `Model/translate.py` script to translate English audio files to Chinese text:

```bash
python Model/translate.py --audio_file <path_to_audio_file> --model_path Model/pth/en2zh_model.pth --device mps
```
- `--audio_file`: Path to the English audio file to be translated (supported formats readable by torchaudio)
- `--model_path`: Path to the trained model parameter file
- `--device`: Inference device (optional, supports cpu/mps/cuda, default mps)

The script will automatically load the model, process the audio, and output the translation result.

---

For further customization or integration, feel free to explore the source code or open an issue!

---

Feel free to star, fork, or open issues to help drive enterprise speech intelligence forward!
