<div align="center">
  <h1 style="font-size:3em; font-weight:bold; letter-spacing:0.1em;">
    ✨🌐 <span style="color:#0078D4; text-shadow: 2px 2px 8px #00bfff;">ENS2ZHT</span> 🌐✨
  </h1>
  <img src="https://img.shields.io/badge/Enterprise%20Speech%20Translation-blue.svg?style=for-the-badge" alt="badge" />
</div>

🌏 Enterprise-level English Speech to Chinese Text Model

Welcome! This project delivers a highly efficient, low-cost, and easily deployable enterprise solution for translating English speech into Chinese text. Whether for call centers, meeting transcription, or multilingual content production, ENS2ZHT is ready for your business needs.

🚀 **High Efficiency**: Built on advanced deep learning architectures and high-performance inference engines, the model processes large volumes of speech data rapidly, supporting both real-time and offline scenarios.

💾 **Low Training Cost**: By innovatively using hard disk caching for training data, memory consumption is greatly reduced, making large-scale training affordable and scalable even in resource-constrained environments.

🔒 **Strong Deployability**: The project features a clear structure, supports mainstream cloud platforms and local servers, and is easy to integrate into existing workflows. Model files and data interfaces are standardized for fast enterprise deployment.

🎯 **Rich Application Scenarios**: ENS2ZHT is ideal for customer service transcription, meeting archiving, smart subtitle generation, cross-border e-commerce translation, and more. With simple configuration, you can automate English speech to Chinese text conversion.

📦 **Quick Start**:
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your data: Place English speech files in the specified directory
3. Run the training script and experience the efficiency and low cost of hard disk caching

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

### 3️⃣ English Speech to Chinese Text Inference

For inference/translation, load the trained model and audio data, and call the model's `autoRegressor` or related interface:

```python
from Model.en2zh import en2zh
import torch

model = en2zh()
model.load_state_dict(torch.load('Model/pth/en2zh_model.pth'))
model.eval()

audio = torch.load('Model/data/audio_0.pt')  # Load preprocessed audio
for token in model.autoRegressor(audio):
    print(token, end='')
```

The model will automatically convert English speech to Chinese text, suitable for batch or real-time scenarios.

---

For further customization or integration, feel free to explore the source code or open an issue!
