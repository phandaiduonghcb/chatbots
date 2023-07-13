from torch import nn
from torch.nn import functional as F

class EmbeddingClassifier(nn.Module):
    def __init__(self, max_words, embed_len, num_classes):
        super(EmbeddingClassifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(max_words*embed_len, 256),
            # nn.ReLU(),

            # nn.Linear(256,128),
            # nn.ReLU(),

            nn.Linear(256,64),
            nn.ReLU(),
            # nn.RNN(input_size=256, hidden_size=64, num_layers=1, batch_first=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, X_batch):
        return self.seq(X_batch)