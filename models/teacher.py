import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
from transformers import AutoModel, AutoConfig, AutoTokenizer
from models.fusion import CrossAttention

from llm2vec import LLM2Vec

class LLM2CLIP(nn.Module):
    def __init__(self, fuse_dim, embd_dim, Nclass):
        super(LLM2CLIP, self).__init__()
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        config = AutoConfig.from_pretrained(
            llm_model_name, trust_remote_code=True
        )
        llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
        llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct' #  Workaround for LLM2VEC
        self.l2v = LLM2Vec(llm_model, self.tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

        self.vlm = AutoModel.from_pretrained("microsoft/LLM2CLIP-Openai-L-14-336", 
                                             torch_dtype=torch.bfloat16, 
                                             trust_remote_code=True).cuda()
        
        self.fuser = CrossAttention(1280, fuse_dim).cuda()
        self.alpha = nn.Parameter(torch.randn(1)).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(fuse_dim, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, Nclass)
        ).cuda()

    def forward(self, pixels, sentences):
        text_features = self.l2v.encode(sentences, convert_to_tensor=True).cuda()
        # pixels = self.processor(images=imgs, return_tensors="pt").pixel_values
        with torch.no_grad(), torch.amp.autocast('cuda'):
            pixels = pixels.cuda()
            image_features = self.vlm.get_image_features(pixels)
            text_features = self.vlm.get_text_features(text_features)

            # print(f'text feature 的形状是{text_features.shape}')
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        attn_a = self.fuser(image_features, text_features)
        attn_b = self.fuser(text_features, image_features)
        
        fusion_features = self.alpha * attn_a + (1 - self.alpha) * attn_b
        logits = self.mlp(fusion_features)
            
        return image_features, text_features, logits