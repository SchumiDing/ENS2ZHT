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

欢迎来到本项目！本项目致力于打造一个高效、低成本、可部署的企业级英文语音翻译为中文文字的解决方案。无论是呼叫中心、会议记录还是多语种内容生产，都能轻松应对。

🚀 **高效率**：模型采用先进的深度学习架构，结合高性能推理引擎，能够快速处理大批量语音数据，满足企业级实时或离线需求。

💾 **低训练成本**：创新性地使用硬盘作为训练数据缓存，大幅降低内存消耗，让海量数据训练变得经济可行。即使在资源有限的环境下，也能轻松扩展训练规模。

🔒 **可部署性强**：项目结构清晰，支持主流云平台和本地服务器部署，便于集成到现有业务流程。模型文件和数据接口均已标准化，助力企业快速上线。

🎯 **应用场景丰富**：
本项目适用于客服语音转写、会议内容归档、智能字幕生成、跨境电商语音翻译等多种场景。只需简单配置，即可实现英文语音到中文文字的自动转换。

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

### 3️⃣ 英文语音翻译为中文文字

推理/翻译时，加载训练好的模型和音频数据，调用模型的 `autoRegressor` 或相关接口即可：

```python
from Model.en2zh import en2zh
import torch

model = en2zh()
model.load_state_dict(torch.load('Model/pth/en2zh_model.pth'))
model.eval()

audio = torch.load('Model/data/audio_0.pt')  # 加载预处理音频
for token in model.autoRegressor(audio):
    print(token, end='')
```

模型会自动将英文语音转为中文文本，适用于批量或实时场景。

---

如需进一步定制或集成，欢迎查阅各脚本源码或提交 issue 交流！
