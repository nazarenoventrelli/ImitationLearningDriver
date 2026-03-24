import torch
import torch.nn as nn


class DrivingNet(nn.Module):
    """
    Backbone CNN + 2 heads:
    - keys_head: logits para [w, a, s, d]
    - mouse_head: regresion normalizada para [dx, dy] en [-1, 1]
    """

    def __init__(self, model_size: str = "base") -> None:
        super().__init__()
        specs = {
            "base": {
                "channels": (32, 64, 128, 192),
                "shared_dim": 256,
                "mouse_hidden_dim": 64,
                "dropout": 0.20,
            },
            "plus": {
                "channels": (48, 96, 192, 256),
                "shared_dim": 320,
                "mouse_hidden_dim": 96,
                "dropout": 0.25,
            },
            "xl": {
                "channels": (64, 128, 256, 384),
                "shared_dim": 512,
                "mouse_hidden_dim": 128,
                "dropout": 0.30,
            },
        }
        if model_size not in specs:
            raise ValueError(f"model_size invalido: {model_size}. Opciones: {sorted(specs)}")
        self.model_size = model_size
        cfg = specs[model_size]
        c1, c2, c3, c4 = cfg["channels"]

        self.backbone = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, cfg["shared_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg["dropout"]),
        )

        self.keys_head = nn.Linear(cfg["shared_dim"], 4)
        self.mouse_head = nn.Sequential(
            nn.Linear(cfg["shared_dim"], cfg["mouse_hidden_dim"]),
            nn.ReLU(inplace=True),
            nn.Linear(cfg["mouse_hidden_dim"], 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features = self.shared(features)
        key_logits = self.keys_head(features)
        mouse_pred = self.mouse_head(features)
        return key_logits, mouse_pred
