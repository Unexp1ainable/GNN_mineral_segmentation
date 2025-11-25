
import lightning
import torch
from torch_geometric.nn import GAT


class GATmodel(lightning.LightningModule):
    """Model used for EDS + BSE fusion."""
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, lr, batch_size, **kwargs) -> None:
        super().__init__()
        kwargs.pop("concat", None)

        if num_layers > 1:
            self.features = GAT(in_channels, hidden_channels, num_layers - 1, hidden_channels, concat=True, **kwargs)
        else:
            self.features = lambda x: x

        self.classification = GAT(hidden_channels, hidden_channels, 1, out_channels, concat=False, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self, x, edge_index, edge_attr):
        out = self.features(x, edge_index, edge_attr=edge_attr)
        return self.classification(out, edge_index, edge_attr=edge_attr)

    def training_step(self, batch, batch_idx):
        graphs = batch
        out = self.forward(graphs.x, graphs.edge_index, graphs.edge_attr)
        loss = self.criterion(out, graphs.y.squeeze().long())
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        graphs = batch
        out = self.forward(graphs.x, graphs.edge_index, graphs.edge_attr)
        loss = self.criterion(out, graphs.y.squeeze().long())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
