import dgl
import dgl.data
from dgl.dataloading import GraphDataLoader
from GatModel import GATregression
import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(model, train_loader, loss_fcn, optimizer):
    model.train()
    batch_loss = 0

    for batched_graph, labels in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        logits = model(
            g=batched_graph,
            node_feats=batched_graph.ndata["feat"],
            edge_feats=batched_graph.edata["feat"],
        )
        loss = loss_fcn(logits, labels.view(-1, 1))

        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        batch_loss += loss.item()
    total_loss = batch_loss / len(train_loader)
    return total_loss


def evaluate(model, test_loader):
    model.eval()
    batch_loss = 0

    for batched_graph, labels in tqdm(test_loader, desc="Evaluating"):
        logits = model(
            g=batched_graph,
            node_feats=batched_graph.ndata["feat"],
            edge_feats=batched_graph.edata["feat"],
        )
        loss = loss_fcn(logits, labels.view(-1, 1))
        batch_loss += loss.item()
    total_loss = batch_loss / len(test_loader)
    return total_loss


if __name__ == "__main__":
    train_dataset = dgl.data.ZINCDataset(mode="train")
    valid_dataset = dgl.data.ZINCDataset(mode="valid")
    test_dataset = dgl.data.ZINCDataset(mode="test")

    train_loader = GraphDataLoader(train_dataset, batch_size=500, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=500, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=500, shuffle=False)

    model = GATregression(
        num_atoms=28,
        num_bonds=4,
        hidden_dim=60,
        out_dim=1,
        heads=[4,4,4,4,4,4],
        num_layers=6,
    )
    summary(model)
    loss_fcn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=2e-5)

    for epoch in range(800):
        train_loss = train(model, train_loader, loss_fcn, optimizer)
        valid_loss = evaluate(model, valid_loader)
        scheduler.step(valid_loss)
        # test_loss = evaluate(model, test_loader)
        print(
            f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}"
        )
    test_loss = evaluate(model, test_loader)
