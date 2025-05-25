import os
from utils.config import Config as conf
from utils.dataset import load_data
from train import Trainer
import multiprocessing

if __name__ == '__main__':
    # 设置启动方法为 'spawn'
    # multiprocessing.set_start_method('spawn')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    train_iter, dev_iter, test_iter = load_data(conf.batch_size, conf.dataname)
    trainer = Trainer(conf)
    trainer.train(train_iter, dev_iter)
    # trainer.evaluate(test_iter)