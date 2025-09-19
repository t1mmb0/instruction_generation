import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score

class GCN(torch.nn.Module):
    """
    A Graph Convolutional Network (GCN) model for link prediction tasks.

    This model computes node embeddings via two stacked GCN layers and
    uses a dot-product decoder to predict the existence of edges between
    pairs of nodes. The design follows the common link prediction pipeline
    in PyTorch Geometric, where `forward` produces embeddings and the
    `decode` function maps candidate edges to scalar scores.

    Parameters
    ----------
    in_channels : int
        Dimensionality of the input node features.
    hidden_channels : int
        Number of hidden units in the first GCN layer.
    out_channels : int
        Dimensionality of the output embeddings (typically used for decoding).

    Methods
    -------
    encode(x, edge_index)
        Computes node embeddings by applying two GCN layers with ReLU
        activation after the first layer.
    decode(z, edge_label_index)
        Computes edge scores for given pairs of nodes using a dot product
        decoder. Returns a tensor of shape [num_edges].
    forward(data)
        Runs the encoder on the input graph data and returns node embeddings.
        The decoder must be called separately on the embeddings and candidate
        edge indices.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x   # Knoteneinbettungen z

    def decode(self, z, edge_label_index):
        # Dot Product Decoder
        z = z[edge_label_index[0]] * z[edge_label_index[1]]
        return z.sum(dim=-1)

    def forward(self, data):
        # liefert nur z (Embeddings), Decoder separat aufrufen
        return self.encode(data.x, data.edge_index)


class Trainer:
    """
    A training utility class for link prediction models in PyTorch Geometric.

    This class encapsulates the training, evaluation, and early stopping logic
    for graph neural network models. It separates model architecture from the
    optimization pipeline, making it easier to reuse the same training loop
    across different models (e.g., GCN, GAT, GraphSAGE).

    Attributes
    ----------
    model : torch.nn.Module
        The graph neural network model to be trained.
    optimizer : torch.optim.Optimizer
        Optimizer used for gradient updates.
    criterion : torch.nn.Module
        Loss function for link prediction (e.g., BCEWithLogitsLoss).
    patience : int
        Number of epochs without improvement on the validation loss before
        early stopping is triggered.

    Methods
    -------
    fit(train_data, val_data, max_epochs=500)
        Runs the training loop with early stopping. Returns the best validation loss
        and restores the model weights to the best epoch.
    train_step(data)
        Performs a single training step (forward, backward, optimizer update).
    eval_step(data)
        Evaluates the model on the given data and returns the loss.
    evaluate(data, k_list=[10])
        Evaluates the model on a dataset split using standard link prediction
        metrics. Returns ROC-AUC, Average Precision, and Hits@K for each K
        in `k_list`.
    """

    def __init__(self, model, optimizer, criterion, patience = 20):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.patience = patience

    def fit(self, train_data, val_data, max_epochs=500):
        best, counter = float("inf"), 0
        best_state = None

        print("\n[ Training Started ]")
        print(f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'Best Val Loss':<15}{'Patience':<10}")
        print("-" * 60)

        for epoch in range(1, max_epochs + 1):
            loss = self.train_step(train_data)
            val_loss = self.eval_step(val_data)

            # Update best state
            if val_loss < best:
                best, counter = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                counter += 1

            # Print every 20 epochs or on improvement
            if epoch % 20 == 0 or val_loss == best:
                print(f"{epoch:<8}{loss:<15.4f}{val_loss:<15.4f}{best:<15.4f}{counter}/{self.patience}")

            if counter >= self.patience:
                print("\n[ Early stopping triggered ]")
                break

        # restore best weights
        self.model.load_state_dict(best_state)
        print("\n[ Training Finished ]")
        print(f"Best Validation Loss: {best:.4f}")
        return best

    
    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model(data)
        logits = self.model.decode(z, data.edge_label_index)
        y = data.edge_label.float()
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_step(self, data):
        self.model.eval()
        z = self.model(data)
        logits = self.model.decode(z, data.edge_label_index)
        y = data.edge_label.float()
        loss = self.criterion(logits, y)
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, k_list=[10]):
        """
        Evaluate the model on a given dataset split.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data object containing `edge_label_index` and `edge_label`.
        k_list : list of int, optional (default=[10])
            List of K values for computing Hits@K.

        Returns
        -------
        metrics : dict
            Dictionary with the following keys:
            - 'roc_auc' : float
                ROC-AUC score of the predictions.
            - 'average_precision' : float
                Average precision (AP) score of the predictions.
            - 'hits@K' : dict
                Hits@K scores for each specified K in `k_list`.
        """
        self.model.eval()
        z = self.model(data)
        logits = self.model.decode(z, data.edge_label_index)
        scores = torch.sigmoid(logits).cpu().numpy()
        y = data.edge_label.cpu().numpy()

        # Standard metrics
        roc_auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)

        # Hits@K
        hits_at_k = {}
        for k in k_list:
            # sort by score, take top-k
            top_k_idx = scores.argsort()[::-1][:k]
            hits_at_k[k] = y[top_k_idx].mean()  # fraction of positives in top-k

        return {
            "roc_auc": roc_auc,
            "average_precision": ap,
            "hits@k": hits_at_k,
        }