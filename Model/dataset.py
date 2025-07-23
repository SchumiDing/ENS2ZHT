from datasets import load_dataset

ds = load_dataset("MLCommons/peoples_speech", "clean")

ds.save_to_disk("Model/data/peoples_speech_clean")