"""Policy network: a small CNN over a stack of recent frames."""
import torch.nn as nn

STACK = 4  # frames per observation; one frame cannot show speed


class PolicyNet(nn.Module):
    """(B, 4, 40, 160) frames in [0, 1] -> (B, 2) logits for [no-op, jump]."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STACK, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # -> 16x20x80
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),     # -> 32x10x40
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),     # -> 32x5x20
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 20, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.head(self.conv(x))
