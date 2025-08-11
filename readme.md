<div align="center">
  <h1 style="font-size:3em; font-weight:bold; letter-spacing:0.1em;">
    ✨🌐 <span style="color:#0078D4; text-shadow: 2px 2px 8px #00bfff;">ENS2ZHT</span> 🌐✨
  </h1>
  <img src="https://img.shields.io/badge/Enterprise%20Speech%20Translation-blue.svg?style=for-the-badge" alt="badge" />
</div>

<div align="center">
  <a href="./readme-en.md" style="font-size:1.2em; font-weight:bold;">🌐 English</a>
</div>
</div>


🌏 企业级英文语音转中文文字模型

欢迎来到本项目！ENS2ZHT 是一个基于 PyTorch 框架的端到端英文语音转中文文字系统，专为企业级场景设计。

🚀 **技术架构**：
* 语音特征提取采用 Wav2Vec2（facebook/wav2vec2-base-960h），通过 Wav2Vec2Processor 进行预处理，支持多种音频格式。
* 文本处理基于 ChineseBertTokenizer，分词与语义编码。
* 主模型结构为单个大规模 Transformer（12 层编码器 + 12 层解码器，d_model=768，nhead=8，dim_feedforward=2048），提升长序列建模能力。
* 输出层为三层线性+ReLU，最终输出 768 维向量。
* 损失函数采用 CosineEmbeddingLoss，优化器为 Adam。
* 推理采用自回归生成，token 解码用 ChineseBertTokenizer。
* 支持批量训练数据生成与硬盘缓存，便于大规模训练。

💾 **数据与训练**：
* 支持 LibriSpeech 格式数据，自动生成中英文对齐训练集。
* 训练过程采用硬盘缓存，极大降低内存消耗，支持大规模数据集。
* 支持 CPU、MPS（Apple Silicon）、CUDA 多种设备。

🔒 **功能与效果**：
* 实现英文语音自动转写为中文文本，支持批量和实时推理。
* 经过初步测试，模型在标准语音数据集上可实现端到端语音翻译，具备较强的泛化能力。
* 结构可扩展，便于集成自定义前后处理模块。

🎯 **应用场景**：
* 客服语音转写
* 会议内容归档
* 智能字幕生成
* 跨境电商语音翻译

如需详细算法说明、评测结果或定制功能，欢迎查阅源码或提交 issue。

📦 **快速上手**：
1. 安装依赖：
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
``` 
2. 下载tokenizer：
```
python Model/load_tokenizer.py
```
3. 准备数据：将英文语音文件放入指定目录
4. 运行训练脚本，体验硬盘缓存带来的高效与低成本

✨ **更多特性**：
* 支持多种音频格式
* 可扩展的模型架构
* 详细的日志与监控

欢迎 star、fork 或提交 issue，一起推动企业语音智能化！
---

## 📚 使用说明

本项目包含数据集生成、模型训练和推理（翻译）三大流程，均已自动化，便于企业快速集成。

### 1️⃣ 生成训练数据集

英文语音与文本数据需按 [LibriSpeech](https://www.openslr.org/12) 格式组织（如 `test-clean` 文件夹），每个子文件夹下包含 `.flac` 音频和 `.trans.txt` 文本。  
使用多线程脚本自动生成带有中文翻译的训练数据：

```bash
python Model/load-datasetM.py --fildir Model/test-clean --threads 24
```

- `--fildir` 指定原始数据目录（如 `Model/test-clean`）
- `--threads` 并发线程数（建议 8~24，视 API 限速和本地性能调整）

脚本会自动调用 API 翻译英文文本为中文，并将音频与中英文文本打包为 json，输出到 `Model/data/` 目录。

### 2️⃣ 训练模型

准备好数据集后，使用如下命令训练模型：

```bash
python Model/train.py --dataset test-clean --batch_size 32 --epoches 10000 --device mps
```

- `--dataset` 指定数据集名（如 `test-clean`，对应 `Model/data/test-clean.json`）
- `--batch_size` 训练批次大小
- `--epoches` 训练轮数
- `--device` 训练设备（支持 `cpu`、`mps`、`cuda`）

训练过程中，模型会自动将数据缓存到硬盘，极大降低内存消耗。每轮训练后模型参数会保存到 `Model/pth/en2zh_model.pth`。

#### 🏷️ 命令行一键翻译
你可以直接使用 `Model/translate.py` 脚本将英文音频文件翻译为中文文本：

```bash
python Model/translate.py --audio_file <音频文件路径> --model_path Model/pth/en2zh_model.pth --device mps
```
- `--audio_file`：待翻译的英文音频文件路径（支持 torchaudio 可读格式）
- `--model_path`：训练好的模型参数文件路径
- `--device`：推理设备（可选，支持 cpu/mps/cuda，默认 mps）

脚本会自动加载模型、处理音频并输出翻译结果。

---

如需进一步定制或集成，欢迎查阅各脚本源码或提交 issue 交流！
