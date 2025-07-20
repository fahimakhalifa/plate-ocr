import torch
import torch.nn as nn

class PlateRecognitionModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super(PlateRecognitionModel, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # [B, 64, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [B, 64, H/2, W/2]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [B, 128, H/4, W/4]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),                         # [B, 256, H/8, W/4]

            nn.AdaptiveAvgPool2d((1, None))            # [B, 256, 1, W']
        )

        # RNN decoder
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.cnn(x)            # [B, 256, 1, W]
        x = x.squeeze(2)           # [B, 256, W]
        x = x.permute(0, 2, 1)     # [B, W, 256]
        x, _ = self.rnn(x)         # [B, W, 2H]
        x = self.fc(x)             # [B, W, vocab_size]
        return x.permute(1, 0, 2)  # [T, B, vocab_size]
