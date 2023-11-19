from torch import nn

from config import dropout

class FeedForward(nn.Module):
    # On a per-token basis, the feed-forward network is applied to the output of each attention head
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
