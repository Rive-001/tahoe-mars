from torch import nn

class GeneSetAdaptor(nn.Module):

    def __init__(self, embedding_dim, output_dim=7467):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, self.output_dim),
            nn.GELU()
        )

        self.upsampling = nn.Sequential(
            nn.Linear(self.embedding_dim, self.output_dim),
            nn.GELU()
        )

    def forward(self, x):

        x = self.upsampling(x) + self.projection(x)
        return x