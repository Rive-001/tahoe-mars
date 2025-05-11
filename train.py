import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as L
from torchmetrics import MeanSquaredError
from pytorch_lightning.callbacks import ModelCheckpoint
from modules import Mars, MarsDataset, GeneSetAdaptor
from torch.utils.data import DataLoader
from utils import _init_weights

import argparse
parser = argparse.ArgumentParser(description='Train MARS')
parser.add_argument('--name', type=str, default='', help='name of the experiment')
parser.add_argument('--train-data', type=str, default='data/train.h5', help='path to training data')
parser.add_argument('--val-data', type=str, default='data/val.h5', help='path to validation data')
parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading')
parser.add_argument('--max-epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--adaptor', type=int, default='gene_set', help='Adaptor type')
parser.add_argument('--criterion', type=int, default='mse', help='Criterion type')
args = parser.parse_args()

# Load Datasets
train_dataset = MarsDataset(args.train_data)
val_dataset = MarsDataset(args.val_data)

# Load DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Load Adaptor
ADAPTOR_DICT = {
    'gene_set': GeneSetAdaptor(256, 7467),
}

adaptor = ADAPTOR_DICT[args.adaptor]
adaptor.apply_weights(_init_weights)


# Load Criterion
CRITERION_DICT = {
    'mse': MeanSquaredError()
}
criterion = CRITERION_DICT[args.criterion]


# Load Model
mars = Mars(
    cell_embedding_dim=1280,
    smiles_embedding_dim=784,
    dosage_embedding=64,
    backbone_embedding_dim=512,
    adaptor=adaptor,
    training=True,
    lr=1e-3,
    weight_decay=1e-5,
    max_epochs=args.max_epochs,
    criterion=criterion,
)

logger = TensorBoardLogger(save_dir="training_logs", name=f'mars_{args.name}')

n_gpus = torch.cuda.device_count()

checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"mars_{args.name}" + "-epoch{epoch:02d}",
        save_top_k=-1,              
        every_n_epochs=10,          
    )

trainer = L.Trainer(
    max_epochs=20,
    gpus=1 if torch.cuda.is_available() else 0,
    logger=logger,
    progress_bar_refresh_rate=20,
    accelerator="gpu" if n_gpus > 0 else "cpu",
    devices=n_gpus if n_gpus > 0 else None,
    strategy="ddp" if n_gpus > 1 else None,
    precision=16,
    callbacks=[checkpoint_callback],
)
model = Mars()
trainer.fit(model, train_dataloader, val_dataloader)