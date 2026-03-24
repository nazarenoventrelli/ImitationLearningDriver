import torch
import torch.nn as nn


class DrivingNet(nn.Module):
    """
    Backbone CNN + 2 heads:
    - keys_head: logits para [w, a, s, d]
    - mouse_head: regresion normalizada para [dx, dy] en [-1, 1]
    """

    def __init__(self, model_size: str = "base", seq_len: int = 1) -> None:
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
        self.seq_len = max(1, int(seq_len))
        self.use_temporal = self.seq_len > 1
        cfg = specs[model_size]
        c1, c2, c3, c4 = cfg["channels"]

        self.frame_backbone = nn.Sequential(
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

        self.frame_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, cfg["shared_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg["dropout"]),
        )

        if self.use_temporal:
            self.temporal = nn.GRU(
                input_size=cfg["shared_dim"],
                hidden_size=cfg["shared_dim"],
                num_layers=1,
                batch_first=True,
            )
        else:
            self.temporal = None

        self.keys_head = nn.Linear(cfg["shared_dim"], 4)
        self.mouse_head = nn.Sequential(
            nn.Linear(cfg["shared_dim"], cfg["mouse_hidden_dim"]),
            nn.ReLU(inplace=True),
            nn.Linear(cfg["mouse_hidden_dim"], 2),
            nn.Tanh(),
        )
        self.mode_head = nn.Linear(cfg["shared_dim"], 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
        if x.dim() != 5:
            raise ValueError("Entrada invalida. Se espera [B,C,H,W] o [B,T,C,H,W].")

        batch, timesteps, channels, height, width = x.shape
        frames = x.reshape(batch * timesteps, channels, height, width)
        frame_features = self.frame_backbone(frames)
        frame_features = self.frame_projector(frame_features).reshape(batch, timesteps, -1)

        if self.temporal is not None:
            temporal_out, _ = self.temporal(frame_features)
            features = temporal_out[:, -1]
        else:
            features = frame_features[:, -1]

        key_logits = self.keys_head(features)
        mouse_pred = self.mouse_head(features)
        mode_logits = self.mode_head(features)
        return key_logits, mouse_pred, mode_logits
