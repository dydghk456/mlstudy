import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
 
'''
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 20],
    batch_size=64,
    shuffle=True,
    input_nodes=data.train_mask,
)
'''
 
class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)
        self.layers = torch.nn.ModuleList([self.sage1, self.sage2])
 
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=1)
 
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0
            for batch in data:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss.item()
                acc += Accuracy(out[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_acc += Accuracy(out[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])
 
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = Accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc