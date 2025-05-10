from torch import nn
import torch

class ConcatEmbeddings(nn.Module):

    def __init__(self, embedding_dim, output_dim=512):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, self.output_dim),
            nn.GELU(),
        )

    def forward(self, embedding_list):

        # Concatenate the embeddings
        x = [embedding(x) for embedding in embedding_list]
        x = torch.cat(x, dim=1)

        # Project to a lower dimension
        x = self.projection(x)

        return x