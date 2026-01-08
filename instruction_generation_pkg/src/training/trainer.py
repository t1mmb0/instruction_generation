import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np


class Trainer:

    def __init__(self, model, optimizer, criterion, device, Regularizer = None, LR_Scheduler_Builder=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler_builder = LR_Scheduler_Builder
        self.criterion = criterion
        self.device = device
        self.history = {"train": [], "val": []}
        self.metrics = {"roc-auc":None, "ave-prec":None}
        self.Regularizer = Regularizer
        self.best_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = None

    def fit(self,train_loader, val_loader, max_epochs,):
         
        #1 Setup for Train Run (Best-Tracking Reset, History Reset, Build Scheduler)
        scheduler = self._setup_run()

        for epoch in range(1, max_epochs+1): 

            #2 Training + Validation (One Epoch)
            train_loss, val_loss = self._run_one_epoch(train_loader, val_loader)

            #3 Update History, Update Best-State, Update Scheduler
            self._after_epoch(epoch, train_loss, val_loss, scheduler)

            #4 Check Early Stopping
            if self._should_stop(epoch=epoch, val_loss=val_loss):
                break

        #5 Return and Summarize best Model
        self._finalize_run()

    def evaluate_model(self, test_loader,):
        batch_all = self._forward_scores(test_loader)
        scores = np.concatenate(batch_all["scores"])
        y = np.concatenate(batch_all["y"])
        edges = np.vstack(batch_all["edges"])


        roc_auc = roc_auc_score(y_true = y, y_score= scores)
        ave_prec = average_precision_score(y_true=y, y_score=scores)
        self.metrics["roc-auc"] = roc_auc
        self.metrics["ave-prec"] = ave_prec
        print(f"[ROC-AUC: {roc_auc:.3f}] [AVE-PREC: {ave_prec:.3f}]")

    def _run_one_epoch(self, train_loader, val_loader):
        train_loss = self._train_step(train_loader)
        val_loss = self._eval_step(val_loader)
        
        return (train_loss, val_loss)
    
    def _after_epoch(self, epoch, train_loss, val_loss, scheduler):
        if scheduler is not None:
            scheduler.step()
        self.history["train"].append(train_loss)
        self.history["val"].append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.best_epoch = epoch

    def _should_stop(self, epoch, val_loss):
        if self.Regularizer and self.Regularizer.check_early_stopping(val_loss):
            print("Early stopping triggered")
            print(f"Stopped at Epoch : {epoch}")
            return True
        return False
    
    def _setup_run(self, ):
        if self.Regularizer:
            self.Regularizer.reset()
        self.best_state = None
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.history = {"train": [], "val": []}
        if self.scheduler_builder is not None:            
            scheduler = self.scheduler_builder.build(self.optimizer)
        else:
            scheduler = None

        return scheduler

    def _finalize_run(self, ):
        self.model.load_state_dict(self.best_state)
        print("\n[ Training Finished ]")
        print(f"[Epoch {self.best_epoch:03d}] , Best_Val: {self.best_val_loss:.4f}")

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
    
    def _return_history(self,):
        return self.history, self.metrics["roc-auc"], self.metrics["ave-prec"]