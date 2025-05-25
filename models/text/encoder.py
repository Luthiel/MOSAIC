import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class ROPE(nn.Module):
    def __init__(self, d_model, dropout=0., max_cache_len=5000, base=1000, time_scale=1.0, time_norm=True):
        super(ROPE, self).__init__()
        assert d_model % 2 == 0
        self.dim = d_model
        self.base = base
        self.time_scale = time_scale
        self.max_cache_len = max_cache_len
        self.time_norm = time_norm
        self.dropout = nn.Dropout(p=dropout)
        self._register_cache()

    def _register_cache(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(self.max_cache_len, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, timestamps):
        """ x is sentence embeddings with shape [sentence_num, 768] """
        _, dim = x.shape
        assert dim == self.dim, \
            f"Sentence embeddings dimension {dim} doesn't match the expected time embeddings dimension {self.dim}."
        
        t = timestamps.to(self.inv_freq.dtype)
        if self.time_norm:
            t = (t - t.min()).clamp(min=1e-8) / (t.max() - t.min() + 1e-8) * self.time_scale
        else:
            t = (t - t.min()).clamp(min=1e-8)

        if torch.equal(t, t.round()) and t.max() < self.max_cache_len and t.min() >= 0:
            indices = t.long()
            cos = self.cos_cached[indices].squeeze(1)
            sin = self.sin_cached[indices].squeeze(1)
        else:
            freqs = torch.einsum('bs,d->bsd', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).squeeze(1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos.to(x.device)
        sin = sin.to(x.device)
        
        x_rot = self.dropout(x) * cos + self.dropout(self._rotate_half(x)) * sin
        
        del cos, sin, x
        torch.cuda.empty_cache()
        
        return x_rot


class MPNet(nn.Module):
    def __init__(self):
        super(MPNet, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        pre_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=4, 
            lora_alpha=8, 
            lora_dropout=0.05,
            init_lora_weights='olora',
            # target_modules=["query", "value", "key"] 
            target_modules=["output.dense", "intermediate.dense"]
        )
        self.model = get_peft_model(pre_model, lora_config).cuda()
        self.time_enc = ROPE(d_model=384)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, sentences, timestamps):
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        tokens = {key: value.to('cuda') for key, value in tokens.items()}
        outputs = self.model(**tokens)
        
        sentences_embd = self.mean_pooling(outputs, tokens['attention_mask'])
        
        del tokens, outputs
        torch.cuda.empty_cache()
        
        if timestamps is not None:
            if len(timestamps.shape) < 2:
                timestamps = timestamps.unsqueeze(1)
            embds = self.time_enc(sentences_embd, timestamps)
            sentences_embd = F.normalize(embds, p=2, dim=1)
        
        return sentences_embd