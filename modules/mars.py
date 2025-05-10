from torch import nn
import torch
import pytorch_lightning as L
from torchmetrics import MeanSquaredError

from modules import ConcatEmbeddings, EmbeddingProjection, Backbone
from utils import _init_weights

class Mars(L.LightningModule):

    def __init__(self, cell_embedding_dim, smiles_embedding_dim, backbone_embedding_dim, adaptor, criterion, 
        training=True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100):

        super().__init__()

        self.save_hyperparameters()
        self.training = training

        self.cell_embedding_dim = cell_embedding_dim
        self.smiles_embedding_dim = smiles_embedding_dim
        self.concat_embedding_dim = self.cell_embedding_dim + self.smiles_embedding_dim
        self.backbone_embedding_dim = backbone_embedding_dim
        self.output_dimension = self.backbone_embedding_dim // 2

        self.criterion = criterion

        self.cell_projection = EmbeddingProjection(self.cell_embedding_dim)
        self.smiles_projection = EmbeddingProjection(self.smiles_embedding_dim)
        # self.mutaion_projection = Projection(self.mutation_embedding_dim)
        
        self.concat = ConcatEmbeddings(self.concat_embedding_dim)
        self.backbone = Backbone(self.backbone_embedding_dim)

        if self.training:
            self.apply(_init_weights)

        self.adaptor = adaptor

        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, cell_id, cell_name, cell_repr, smile_repr, dosage):

        cell_embed = self.cell_projection(cell_repr)
        smiles_embed = self.smiles_projection(smile_repr)

        concat_embed = self.concat([cell_embed, smiles_embed])
        backbone_embed = self.backbone(concat_embed)

        output = self.adaptor(backbone_embed)

        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # 6. log to TensorBoard
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        mse  = self.val_mse(logits, y.float())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse",  mse,  on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        self.train_mse.reset()

    def on_validation_epoch_end(self):
        self.val_mse.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }
