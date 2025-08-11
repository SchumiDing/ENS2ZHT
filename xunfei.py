#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

def local_transcribe(audio_file_path: str, model_name: str = "base") -> str:
    """
    Transcribe audio locally using OpenAI Whisper model (default: base).
    """
    try:
        import whisper  # type: ignore
    except ImportError:
        raise ImportError("请先安装 openai-whisper: pip install openai-whisper")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file_path)
    return result.get("text", "")


def main():
    parser = argparse.ArgumentParser(description='Offline English Speech-to-Text using Whisper')
    parser.add_argument('audio_file', help='Path to audio file (e.g., WAV, 16kHz mono)')
    parser.add_argument('--model', default='base', help='Whisper model name (tiny, base, small, medium, large)')
    args = parser.parse_args()
    text = local_transcribe(args.audio_file, args.model)
    print(text)


if __name__ == '__main__':
    main()
