import os
from utils.config_distill import Config as conf
from utils.dataset import load_vision_text
from train_distill import DistillationTrainer
import multiprocessing

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 设置启动方法为 'spawn'
    multiprocessing.set_start_method('spawn')
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    train_iter, dev_iter, test_iter = load_vision_text(conf.dataname, conf.batch_size)
    trainer = DistillationTrainer(conf)
    trainer.train(train_iter, dev_iter)