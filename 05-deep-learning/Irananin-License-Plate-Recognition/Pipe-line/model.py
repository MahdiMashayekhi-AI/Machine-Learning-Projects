import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, nc, nclass, nh=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)), nn.Dropout(0.2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)), nn.Dropout(0.2),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU()
        )

        self.rnn = nn.LSTM(
            512, nh, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.3
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(nh * 2, nclass)

        self.fc.bias.data[0] = -2.0

    def forward(self, x):
        x = self.cnn(x)
        x = F.adaptive_avg_pool2d(x, (1, x.size(3)))
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x