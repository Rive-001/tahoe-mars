from torch import nn
import torch

class ConcatEmbeddings(nn.Module):

    def __init__(self, embedding_list):

        super().__init__()


        self.embedding_list = embedding_list
        self.embedding_dim = sum([len(embedding) for embedding in self.embedding_list])

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
        )

    def forward(self, x):

        # Concatenate the embeddings
        x = [embedding(x) for embedding in self.embedding_list]
        x = torch.cat(x, dim=1)

        # Project to a lower dimension
        x = self.projection(x)

        return x