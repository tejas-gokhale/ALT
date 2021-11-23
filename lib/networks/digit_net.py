import torch.nn as nn

__all__ = ['DigitNet']

class DigitNet(nn.Module):

    def __init__(self, num_classes=1000, **kwargs):
        super(DigitNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x, return_feat=False):
        feat = self.features(x)
        y = self.classifier(feat.view(feat.size(0), -1))

        if return_feat:
            return feat, y
        return y



