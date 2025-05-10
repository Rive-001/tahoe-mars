from torch import nn
import torch

from modules import ConcatEmbeddings, EmbeddingProjection, Backbone

class Mars(nn.Module):

    def __init__(self, cell_embedding_dim, smiles_embedding_dim, backbone_embedding_dim, adaptor):

        super().__init__()

        self.cell_embedding_dim = cell_embedding_dim
        self.smiles_embedding_dim = smiles_embedding_dim
        self.concat_embedding_dim = self.cell_embedding_dim + self.smiles_embedding_dim
        self.backbone_embedding_dim = backbone_embedding_dim
        self.output_dimension = self.backbone_embedding_dim // 2

        self.cell_projection = EmbeddingProjection(self.cell_embedding_dim)
        self.smiles_projection = EmbeddingProjection(self.smiles_embedding_dim)
        # self.mutaion_projection = Projection(self.mutation_embedding_dim)
        
        self.concat = ConcatEmbeddings(self.concat_embedding_dim)
        self.backbone = Backbone(self.backbone_embedding_dim)

        self.adaptor = adaptor

    def forward(self, cell_repr, smiles_repr):

        cell_embed = self.cell_projection(cell_repr)
        smiles_embed = self.smiles_projection(smiles_repr)

        concat_embed = self.concat([cell_embed, smiles_embed])
        backbone_embed = self.backbone(concat_embed)

        output = self.adaptor(backbone_embed)

        return output
