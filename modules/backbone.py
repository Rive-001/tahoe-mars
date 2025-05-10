from torch import nn

class Residual(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.res = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU()
        )

    def forward(self, x):

        return self.res(x)

class Backbone(nn.Module):

    def __init__(self, embedding_dimension):

        self.embedding_dimension = embedding_dimension
        self.output_dimension = self.embedding_dimension//2

        self.l1 = nn.Sequential(
            nn.Linear(self.embedding_dimension, 1024),
            nn.GELU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU()
        )
        self.l4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU()
        )
        self.l5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU()
        )

        self.res1 = Residual(self.embedding_dimension, 1024)
        self.res2 = Residual(1024, 20248)
        self.res3 = Residual(2048, 1024)
        self.res4 = Residual(1024, 512)
        self.res5 = Residual(512, 256)

    def forward(self, x):

        x = self.l1(x) + self.res1(x)
        x = self.l2(x) + self.res2(x)
        x = self.l3(x) + self.res3(x)
        x = self.l4(x) + self.res4(x)
        x = self.l5(x) + self.res5(x)

        return x
        


    