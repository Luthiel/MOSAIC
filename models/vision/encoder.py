import torch
import torch.nn as nn
import torchvision.transforms as transforms

from transformers import AutoImageProcessor, ConvNextV2Model, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

from typing import Dict
import torchvision.transforms as T


class MambaVision(nn.Module):
    def __init__(self, version="T-1K"):
        super(MambaVision, self).__init__()
        supported = VersionWrapper.MambaVisionSupported.keys()
        assert version in supported, f"version must be one of {supported}"

        pre_model = AutoModel.from_pretrained(f"nvidia/MambaVision-{version}", trust_remote_code=True)
            
        lora_config = LoraConfig(
            # task_type=TaskType.FEATURE_EXTRACTION, # 让 peft 框架自己推断任务类型，否则会认为是语言模型，要求输入 input_ids
            r=4, 
            lora_alpha=8, 
            lora_dropout=0.05,
            init_lora_weights='olora',
            # target_modules=["dt_proj", "qkv"]
            target_modules=["mixer.proj", "mlp.fc1", "mlp.fc2"]
        )
        self.model = get_peft_model(pre_model, lora_config).cuda()

    def forward(self, pixels, return_stage=False):
        """ image should be PIL.Image.Image object """
        pixels = pixels.cuda()
        pool_features, stage_features = self.model(pixels)
        if return_stage:
            return pool_features, stage_features
        return pool_features

class ConvNeXt(nn.Module):
    def __init__(self, version="tiny-1k-224"):
        super(ConvNeXt, self).__init__()
        supported = VersionWrapper.ConvNeXtSupported.keys()
        assert version in supported.keys(), f"version must be one of {supported}"
        self.preprocessor = AutoImageProcessor.from_pretrained(f"facebook/convnextv2-{version}")
        self.model = ConvNextV2Model.from_pretrained(f"facebook/convnextv2-{version}")

    def forward(self, img):
        model = get_peft_model(self.model, self.config)
        outputs = model(**img)
        return outputs.last_hidden_state


class VersionWrapper:
    MambaVisionSupported: Dict = {'T-1K': {'Params': '31.8M', 'FLOPs': '4.4G'},
                                  "T2-1K": {'Params': '35.1M', 'FLOPs': '5.1G'}, 
                                  "S-1K": {'Params': '50.1M', 'FLOPs': '7.5G'},
                                  "B-1K": {'Params': '97.7M', 'FLOPs': '15.0G'}, 
                                  "L-1K": {'Params': '227.9M', 'FLOPs': '34.9G'}, 
                                  "L2-1K": {'Params': '241.5M', 'FLOPs': '37.5G'}}
    
    ConvNeXtSupported: Dict = {'nano-22k-224': {'Params': '15.6M', 'FLOPs': '2.45G'},
                                'nano-22k-384': {'Params': '15.6M', 'FLOPs': '7.21G'},
                                'tiny-22k-224': {'Params': '28.6M', 'FLOPs': '4.47G'},
                                'tiny-22k-384': {'Params': '28.6M', 'FLOPs': '13.1G'},
                                'base-22k-224': {'Params': '89M', 'FLOPs': '15.4G'},
                                'base-22k-384': {'Params': '89M', 'FLOPs': '45.2G'},
                                'large-22k-224': {'Params': '198M', 'FLOPs': '34.4G'},
                                'large-22k-384': {'Params': '198M', 'FLOPs': '101.1G'},
                                'huge-22k-384': {'Params': '660M', 'FLOPs': '337.9G'},
                                'huge-22k-512': {'Params': '660M', 'FLOPs': '600.8G'},

                                'atto-1k-224': {'Params': '3.7M', 'FLOPs': '0.55G'},
                                'femto-1k-224': {'Params': '5.2M', 'FLOPs': '0.78G'},
                                'pico-1k-224': {'Params': '9.1M', 'FLOPs': '1.37G'},
                                'nano-1k-224': {'Params': '15.6M', 'FLOPs': '2.45G'},
                                'tiny-1k-224': {'Params': '28.6M', 'FLOPs': '4.47G'},
                                'base-1k-224': {'Params': '89M', 'FLOPs': '15.4G'},
                                'large-1k-224': {'Params': '198M', 'FLOPs': '34.4G'},
                                'huge-1k-224': {'Params': '660M', 'FLOPs': '115G'}}


