import torch
import torch.nn as nn


class MultiHeadCNN(nn.Module):
    def __init__(self):
        super(MultiHeadCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def load_model():
    model = MultiHeadCNN()
    model.load_state_dict(
        torch.load("snow_model_csv.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model