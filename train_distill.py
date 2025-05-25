import torch
import torch.nn as nn
from models.student import MVP as s_model
from models.teacher import LLM2CLIP as t_model
from utils.lossfn import KDLoss
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from models.linear import Proj as fuse_model
from utils.metrics import evaluate_model
import logging
from early_stopping_pytorch import EarlyStopping
from sklearn.metrics import classification_report
from torch.amp import autocast, GradScaler


def get_logger(log_name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_name), logging.StreamHandler()]
    )
    return logging.getLogger('info_recorder')


class DistillationTrainer:
    def __init__(self, conf):
        in_dim, hidden_dim = conf.fuse_dim, conf.hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("distill.log")
        self.epochs = conf.epochs

        self.s_model = s_model(in_dim, hidden_dim, conf.Nclass)
        self.t_model = t_model(in_dim, hidden_dim, conf.Nclass)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.s_model.parameters()),
            lr=conf.lr,
            weight_decay=conf.weight_decay
        )

        self.model_path = conf.model_path
        self.criterion = KDLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3)
        self.elsp = EarlyStopping(patience=5, verbose=True, path=self.model_path)
        self.scaler = GradScaler()

    def train(self, train_iter, dev_iter):
        for epoch in range(self.epochs):
            self.s_model.train(), self.t_model.eval()
            train_losses, train_preds, train_labels = [], [], []

            pbar = tqdm(train_iter, desc=f"Epoch {epoch + 1} / {self.epochs}", unit="batch")
            for batch in train_iter:
                img, post, labels = batch

                self.optimizer.zero_grad()
                
                with autocast('cuda'):
                    t_v_feat, t_t_feat, t_logit = self.t_model(img, post)
                    s_v_feat, s_t_feat, s_logit = self.s_model(img, post)
                    
                    loss = self.criterion(s_v_feat, t_v_feat, s_t_feat, t_t_feat, s_logit, t_logit)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.s_model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_losses.append(loss.item())
                
                s_logit = s_logit.squeeze(1)
                # print(f'student logit shape is {s_logit.shape}')
                train_preds.extend(s_logit.argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.argmax(dim=1).cpu().numpy())

                pbar.set_postfix(loss=loss.item())

            train_loss = np.mean(train_losses)
            self._log_metrics(epoch, "Train", train_loss, train_preds, train_labels)

            val_loss = self.evaluate(dev_iter, epoch)
            self.scheduler.step(val_loss)

            self.elsp(val_loss, self.s_model)
            if self.elsp.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    def evaluate(self, dev_iter, epoch):
        self.s_model.eval(), self.t_model.eval()
        dev_losses, dev_preds, dev_labels = [], [], []
        pbar = tqdm(dev_iter, desc=f"[Eval] Epoch {epoch + 1}", unit="batch")

        with torch.no_grad():
            for batch in dev_iter:
                img, post, labels = batch
                t_v_feat, t_t_feat, s_logit = self.t_model(img, post)
                s_v_feat, s_t_feat, t_logit = self.s_model(img, post)

                loss = self.criterion(s_v_feat, t_v_feat, s_t_feat, t_t_feat, s_logit, t_logit)
                dev_losses.append(loss.item())
                
                s_logit = s_logit.squeeze(1)
                dev_preds.extend(s_logit.argmax(dim=1).cpu().numpy())
                dev_labels.extend(labels.argmax(dim=1).cpu().numpy())
                pbar.set_postfix(loss=loss.item())

        dev_loss = np.mean(dev_losses)
        self._log_metrics(epoch, "Valid", dev_loss, dev_preds, dev_labels)
        return dev_loss

    def _log_metrics(self, epoch, phase, loss, preds, labels):
        acc, precision, recall, f1, macro = evaluate_model(preds, labels)
        self.logger.info(f"[{phase}] Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, Macro-F1: {macro:.4f}")
        self.logger.info(f"[{phase}] Classification Report:\n{classification_report(labels, preds, digits=3)}")
