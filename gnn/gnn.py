import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

class GNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)
 
    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        return x
    

class GNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gnn1 = GNNLayer(dim_in, dim_h)
        self.gnn2 = GNNLayer(dim_h, dim_h)
        self.gnn3 = GNNLayer(dim_h, dim_out)
        self.layers = torch.nn.ModuleList([self.gnn1, self.gnn2, self.gnn3])
 
    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return F.log_softmax(x, dim=1)
 
    def fit(self, data, epochs, adj, verbose=0):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=1e-4)
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self(data.x, adj)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = Accuracy(out[data.train_mask].argmax(dim=1),
                          data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if verbose:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = Accuracy(out[data.val_mask].argmax(dim=1),
                                  data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                      f' {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')
 
    @torch.no_grad()
    def test(self, data, adj):
        self.eval()
        out = self(data.x, adj)
        acc = Accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc