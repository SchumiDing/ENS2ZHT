from transformers import AutoTokenizer, AutoModel
model_name = "hfl/chinese-bert-wwm-ext"
local_dir = "./local_chinese_bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(local_dir)