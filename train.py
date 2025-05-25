import torch
import torch.nn as nn
import numpy as np
import logging

from tqdm import tqdm
from utils.lossfn import FinalLoss, FocalLoss
from utils.metrics import evaluate_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping_pytorch import EarlyStopping
from sklearn.metrics import classification_report
from models.e2e import Mosaic
from sklearn.utils.class_weight import compute_class_weight

def get_logger(log_name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_name), logging.StreamHandler()]
    )
    return logging.getLogger('info_recorder')


class Trainer:
    def __init__(self, conf):
        in_dim, hidden_dim, state_dim, fuse_dim = \
            conf.fuse_dim, conf.hidden_dim, conf.state_dim, conf.fuse_dim
        peft_path = conf.peft_path
        need_macro = conf.need_macro
        use_dum = conf.use_dum
        
        self.model_path = conf.model_path
        self.num_classes = conf.Nclass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("e2e.log")
        self.epochs = conf.epochs
        self.use_vae = conf.use_vae
        
        self.model = Mosaic(in_dim, hidden_dim, state_dim, self.num_classes, fuse_dim, peft_path, if_rec=self.use_vae, need_macro=need_macro, use_dum=use_dum)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=conf.lr,
            weight_decay=conf.weight_decay
        )

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3, threshold=0.001)
        
        self.elsp = EarlyStopping(patience=10, verbose=True, path=self.model_path)

    def compute_class_weights(self, labels_all):
        labels_np = np.array(labels_all)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(self.num_classes), y=labels_np)
        weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        return weights

    def train(self, train_iter, dev_iter):
        all_train_labels = []
        for batch in train_iter:
            _, _, _, _, labels = batch
            all_train_labels.extend(torch.argmax(labels, dim=1).tolist())
        class_weights = self.compute_class_weights(all_train_labels)
        
        if self.use_vae:
            self.criterion = FinalLoss(alpha=class_weights)    
        else:
            self.criterion = FocalLoss(alpha=class_weights)

        for epoch in range(self.epochs):
            self.model.train()
            train_losses, train_preds, train_labels = [], [], []

            for batch in tqdm(train_iter, desc=f"Epoch {epoch + 1} / {self.epochs}", unit="batch"):
                graphs, macro_graphs, images, posts, labels = batch

                graphs = [g.to(self.device) for g in graphs]
                macro_graphs = [g.to(self.device) for g in macro_graphs]
                labels = labels.to(self.device)
                labels = torch.argmax(labels, dim=1)

                self.optimizer.zero_grad()
                if self.use_vae:
                    T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits = \
                        self.model(posts, images, macro_graphs, graphs)
                    loss = self.criterion(T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits, labels)
                else:
                    logits = self.model(posts, images, macro_graphs, graphs)
                    loss = self.criterion(logits, labels)
                    
                preds = torch.argmax(logits, dim=1)
                
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
                train_preds.append(preds.cpu().numpy())
                train_labels.append(labels.cpu().numpy())
            
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            train_loss = np.mean(train_losses)
            
            self._log_metrics(epoch, "Train", train_loss, train_preds, train_labels)

            val_loss = self.validate(dev_iter, epoch)
            self.scheduler.step(val_loss)

            self.elsp(val_loss, self.model)
            if self.elsp.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    def validate(self, dev_iter, epoch):
        self.model.eval()
        dev_losses, dev_preds, dev_labels = [], [], []

        with torch.no_grad():
            for batch in tqdm(dev_iter, desc=f"[Eval] Epoch {epoch + 1}", unit="batch"):
                graphs, macro_graphs, images, posts, labels = batch
                
                graphs = [g.to(self.device) for g in graphs]
                macro_graphs = [g.to(self.device) for g in macro_graphs]
                labels = labels.to(self.device)
                labels = torch.argmax(labels, dim=1)

                if self.use_vae:
                    T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits = \
                        self.model(posts, images, macro_graphs, graphs)
                    loss = self.criterion(T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits, labels)
                else:
                    logits = self.model(posts, images, macro_graphs, graphs)
                    loss = self.criterion(logits, labels)
                
                preds = torch.argmax(logits, dim=1)
                
                dev_losses.append(loss.item())
                dev_preds.append(preds.cpu().numpy())
                dev_labels.append(labels.cpu().numpy())
            
        dev_loss = np.mean(dev_losses)
        self._log_metrics(epoch, "Valid", dev_loss, dev_preds, dev_labels)
        return dev_loss

    def evaluate(self, test_iter):
        # 加载模型
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        test_losses, test_preds, test_labels = [], [], []
        
        all_test_labels = []
        for batch in test_iter:
            _, _, _, _, labels = batch
            all_test_labels.extend(torch.argmax(labels, dim=1).tolist())
        class_weights = self.compute_class_weights(all_test_labels)
        
        if self.use_vae:
            self.criterion = FinalLoss(alpha=class_weights)    
        else:
            self.criterion = FocalLoss(alpha=class_weights)
        
        with torch.no_grad():
            for batch in test_iter:
                graphs, macro_graphs, images, posts, labels = batch
                
                graphs = [g.to(self.device) for g in graphs]
                macro_graphs = [g.to(self.device) for g in macro_graphs]
                labels = labels.to(self.device)
                labels = torch.argmax(labels, dim=1)

                if self.use_vae:
                    T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits = \
                        self.model(posts, images, macro_graphs, graphs)
                    loss = self.criterion(T_recon, T, P_recon, P, mu_T, logvar_T, mu_P, logvar_P, logits, labels)
                else:
                    logits = self.model(posts, images, macro_graphs, graphs)
                    loss = self.criterion(logits, labels)
                    
                preds = torch.argmax(logits, dim=1)

                test_losses.append(loss.item())
                test_preds.append(preds.cpu().numpy())
                test_labels.append(labels.cpu().numpy())

            test_loss = np.mean(test_losses)
            self._log_metrics(0, "Test", test_loss, test_preds, test_labels)

    def _log_metrics(self, epoch, phase, loss, preds, labels):
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        acc, precision, recall, f1, macro = evaluate_model(preds, labels)
        self.logger.info(f"[{phase}] Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, Macro-F1: {macro:.4f}")
        self.logger.info(f"[{phase}] Classification Report:\n{classification_report(labels, preds, digits=3)}")
