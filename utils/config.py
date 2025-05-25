import os

class Config:
    
    lr: int = 5e-5 
    epochs: int = 100
    batch_size: int = 128 # todo weibo - 64, pheme - 128
    dataname: str = 'pheme' # todo 'pheme'
    fuse_dim: int = 512
    hidden_dim: int = 256
    state_dim: int = 256
    Nclass: int = 2 # todo weibo - 4, pheme - 2
    weight_decay: float = 1e-4
    model_path: str = os.path.join(os.getcwd(), 'ckpt', dataname + '_mosaic_without_kd.pt')
    # peft_path: str = os.path.join(os.getcwd(), 'ckpt', dataname + '_kd_mvp.pt')
    peft_path: str = None
    use_vae: bool = False # todo if False, use the linear combination as fuser module
    need_macro: bool = True # todo ablation 
    use_dum: bool = True # todo ablation
