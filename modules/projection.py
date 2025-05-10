from torch import nn

class EmbeddingProjection(nn.Module):

    def __init__(self, embedding_dim, output_dim=256):

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.output_dim),
            nn.GELU()
        )

        self.downsampling = nn.Sequential(
            nn.Linear(self.embedding_dim, self.output_dim),
            nn.GELU()
        )

    def forward(self, x):

        x = self.downsampling(x) + self.projection(x)