import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np

class Trainer:

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {"train": [], "val": []}
        self.min_val_loss = float("inf")
        self._counter = 0

    def early_stopping(self, patience, min_delta, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self._counter = 0
        elif val_loss > (self.min_val_loss + min_delta):
            self._counter += 1
            if self._counter == patience:           
                return True
        return False
        
    def fit(self,train_loader, val_loader, max_epochs, patience, min_delta):
         
         for epoch in range(1, max_epochs+1): 
             loss = self._train_step(train_loader)
             val_loss = self._eval_step(val_loader)
             self.history["train"].append(loss)
             self.history["val"].append(val_loss)

             if self.early_stopping(patience, min_delta, val_loss):
                 print(f"Training stopped in Epoch {epoch}, (patience reached)")
                 break

             print(f"[Epoch {epoch:03d}] Train: {loss:.4f}, Val: {val_loss:.4f}")

    def _train_step(self, train_loader,):
        loss_all = list()
        self.model.train()  
        for batch in train_loader:
            batch = batch.to(self.device)
            z = self.model.encode(batch.x, batch.edge_index)
            logits = self.model.decode(z, batch.edge_label_index)
            y = batch.edge_label.float()
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_all.append(loss.item())
        return np.mean(loss_all)
    
    def evaluate_model(self, test_loader,):
        if not self.history["train"]:
            raise Exception(f"The Model has not been trained yet.")
        batch_all = self._forward_scores(test_loader)
        scores = np.concatenate(batch_all["scores"])
        y = np.concatenate(batch_all["y"])
        edges = np.vstack(batch_all["edges"])


        roc_auc = roc_auc_score(y_true = y, y_score= scores)
        ave_prec = average_precision_score(y_true=y, y_score=scores)

        print(f"[ROC-AUC: {roc_auc:.3f}] [AVE-PREC: {ave_prec:.3f}]")

    @torch.no_grad()
    def _forward_scores(self, test_loader):        
        batch_all = {"scores": [],
                     "y": [],
                     "edges": []}
        self.model.eval()
        for batch in test_loader:
            batch = batch.to(self.device)
            z = self.model.encode(batch.x, batch.edge_index)
            logits = self.model.decode(z, batch.edge_label_index)
            scores = torch.sigmoid(logits).cpu().numpy()
            y = batch.edge_label.cpu().numpy()
            edges = batch.edge_label_index.cpu().T.numpy()
            batch_all["scores"].append(scores)
            batch_all["y"].append(y)
            batch_all["edges"].append(edges)

        return batch_all
            

    @torch.no_grad()
    def _eval_step(self, val_loader,):
        self.model.eval()
        loss_all = list()
        for batch in val_loader:
            batch = batch.to(self.device)
            z = self.model.encode(batch.x, batch.edge_index)
            logits = self.model.decode(z, batch.edge_label_index)
            y = batch.edge_label.float()
            loss = self.criterion(logits, y)
            loss_all.append(loss.item())
        return np.mean(loss_all)


