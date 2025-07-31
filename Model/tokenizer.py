import transformers
from transformers import AutoTokenizer, AutoModel
import torch

class ChineseBertTokenizer:

    def __init__(self, model_name="./local_chinese_bert"):

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        
    def tokenize_text(self, text):

        tokens = self.tokenizer.tokenize(text)
        
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'tokens': tokens,
            'token_ids': encoded['input_ids'].squeeze().tolist(),
            'attention_mask': encoded['attention_mask'].squeeze().tolist(),
            'encoded': encoded
        }
    
    def to_vector(self, text):
        
        if not hasattr(self, 'model'):
            print(f"Loading model: {self.model_name}")
            self.model = AutoModel.from_pretrained(self.model_name, output_hidden_states=True)
            self.model.eval()
        
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            
        # 计算到stop token的长度
        input_ids = encoded['input_ids'].squeeze().tolist()
        stop_token_id = self.tokenizer.sep_token_id
        if stop_token_id in input_ids:
            stop_index = input_ids.index(stop_token_id)
        else:
            stop_index = len(input_ids)
        return {
            'last_hidden_state': outputs.hidden_states[0],  # [1, seq_len, hidden_size]
            'pooler_output': outputs.pooler_output,          # [1, hidden_size]
            'hidden_size': outputs.hidden_states[0].shape[-1],
            'sequence_length': outputs.hidden_states[0].shape[1],
            'stop_token_length': stop_index + 1
        }
    
    def batch_tokenize(self, texts, max_length=512):

        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return encoded
    
    def decode_tokens(self, token_ids):

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_info(self):
        
        vocab = self.tokenizer.get_vocab()
        return {
            'vocab_size': len(vocab),
            'special_tokens': {
                'cls_token': self.tokenizer.cls_token,
                'sep_token': self.tokenizer.sep_token,
                'pad_token': self.tokenizer.pad_token,
                'unk_token': self.tokenizer.unk_token,
                'mask_token': self.tokenizer.mask_token
            },
            'special_token_ids': {
                'cls_token_id': self.tokenizer.cls_token_id,
                'sep_token_id': self.tokenizer.sep_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'unk_token_id': self.tokenizer.unk_token_id,
                'mask_token_id': self.tokenizer.mask_token_id
            }
        }
    def vector_to_token_ids(self, hidden_states, top_k=1):

        if not hasattr(self, 'model'):
            print(f"Loading model: {self.model_name}")
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
        
        if not hasattr(self, 'embeddings'):
            self.embeddings = self.model.embeddings.word_embeddings.weight.data
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        
        flat_hidden_norm = torch.nn.functional.normalize(flat_hidden, p=2, dim=1)
        embeddings_norm = torch.nn.functional.normalize(self.embeddings, p=2, dim=1)
        similarities = torch.mm(flat_hidden_norm, embeddings_norm.t())

        
        if top_k == 1:
            predicted_ids = similarities.argmax(dim=1)
            predicted_ids = predicted_ids.view(batch_size, seq_len)
        else:
            _, predicted_ids = similarities.topk(k=top_k, dim=1)
            predicted_ids = predicted_ids.view(batch_size, seq_len, top_k)
        
        return predicted_ids

def demo_chinese_bert_tokenizer():
    tokenizer = ChineseBertTokenizer("hfl/chinese-bert-wwm-ext")
    
    sample_texts = "我喜欢自然语言处理。"

    tokenized = tokenizer.tokenize_text(sample_texts)
    
    print("Tokenization结果:")
    print(f"文本: {tokenized['text']}")
    print(f"Tokens: {tokenized['tokens']}, {len(tokenized['tokens'])}")
    
    
    vectorized = tokenizer.to_vector(sample_texts)
    print("\n向量化结果:")
    print(f"Last Hidden State Shape: {vectorized['last_hidden_state'].shape}")
    print(f"Pooler Output Shape: {vectorized['pooler_output'].shape}")
    
    token_ids = tokenizer.vector_to_token_ids(vectorized['last_hidden_state'])
    print("\n向量转换为Token IDs:")
    print(f"Token IDs: {token_ids}")
    
    text = tokenizer.decode_tokens(token_ids[0])
    print("\n解码Token IDs:")
    print(f"解码文本: {text}")
    

if __name__ == "__main__":
    demo_chinese_bert_tokenizer()

