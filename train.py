import lightning
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

import models
from graph_fusion_eds import ToGraphEDS

# prepare data and transformations
transform = ToGraphEDS(False, (0.,0.7))
dataset = ... # TODO! load you own data here


BATCH_SIZE = 16
train_idx, validation_idx = train_test_split(np.arange(len(dataset)),
                                             test_size=0.1,
                                             random_state=42,
                                             shuffle=True)


train_dataset = Subset(dataset, train_idx)
validation_dataset = Subset(dataset, validation_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

# prepare training
NUM_CLASSES = ... # TODO! this depends on your data
NUM_FEATURES = dataset[0].num_node_features


checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    every_n_train_steps=1,
    save_last=True,
    auto_insert_metric_name=True
)

model = models.GATmodel(NUM_FEATURES, 56, 3, NUM_CLASSES, lr=0.01, edge_dim=1, heads=4, fill_value=0.0 , batch_size=BATCH_SIZE)
trainer = lightning.Trainer(
    accelerator="gpu",
    log_every_n_steps=25,
    max_epochs=200,
    callbacks=[
        checkpoint_callback
    ],
    logger=TensorBoardLogger("mineralogy_logs", name="best_hyperparams")
)

# train
trainer.fit(model, train_loader, validation_loader)



