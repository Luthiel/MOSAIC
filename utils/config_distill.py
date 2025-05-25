import os

class Config:
    
    lr: int = 0.0001
    epochs: int = 100
    batch_size: int = 64
    dataname: str = 'pheme'
    hidden_dim: int = 256
    fuse_dim: int = 512
    Nclass: int = 4
    weight_decay: float = 1e-4
    model_path: str = os.path.join(os.getcwd(), 'ckpt', dataname + '_kd_mvp.pt')
