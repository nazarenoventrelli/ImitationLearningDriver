import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

KEY_ORDER = ("w", "a", "s", "d")


@dataclass
class SampleRecord:
    frame_path: Path
    key_target: np.ndarray
    mouse_target: np.ndarray
    session_name: str


def _parse_pressed_keys(pressed_keys: str) -> set[str]:
    if not pressed_keys:
        return set()
    return {token.strip().lower() for token in pressed_keys.split("+") if token.strip()}


def _build_key_target(pressed_keys: str) -> np.ndarray:
    active_keys = _parse_pressed_keys(pressed_keys)
    return np.array([1.0 if key in active_keys else 0.0 for key in KEY_ORDER], dtype=np.float32)


def _build_mouse_target(row: dict[str, str], mouse_scale: float) -> np.ndarray:
    dx = float(row.get("mouse_dx", 0.0) or 0.0)
    dy = float(row.get("mouse_dy", 0.0) or 0.0)
    normalized = np.array([dx / mouse_scale, dy / mouse_scale], dtype=np.float32)
    return np.clip(normalized, -1.0, 1.0)


def load_records(captures_dir: Path, mouse_scale: float) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    if not captures_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de capturas: {captures_dir}")

    sessions = sorted(
        [p for p in captures_dir.iterdir() if p.is_dir() and (p / "labels.csv").exists()],
        key=lambda p: p.name,
    )
    if not sessions:
        raise RuntimeError(f"No se encontraron sesiones con labels.csv en: {captures_dir}")

    for session_dir in sessions:
        labels_path = session_dir / "labels.csv"
        with labels_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_name = row.get("filename")
                if not frame_name:
                    continue

                frame_path = session_dir / "frames" / frame_name
                if not frame_path.exists():
                    continue

                records.append(
                    SampleRecord(
                        frame_path=frame_path,
                        key_target=_build_key_target(row.get("pressed_keys", "")),
                        mouse_target=_build_mouse_target(row, mouse_scale=mouse_scale),
                        session_name=session_dir.name,
                    )
                )

    if not records:
        raise RuntimeError("No se pudieron construir samples validos a partir del dataset.")
    return records


def split_records(
    records: list[SampleRecord], val_ratio: float, seed: int
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio debe estar entre 0 y 1.")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(len(records) * val_ratio))

    val_indices = set(indices[:val_count])
    train_records = [records[i] for i in indices[val_count:]]
    val_records = [records[i] for i in indices[:val_count]]

    if not train_records or not val_records:
        raise RuntimeError("Split invalido: train o val quedo vacio.")
    return train_records, val_records


class DrivingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[SampleRecord],
        image_size: tuple[int, int],
        augment: bool = False,
        jitter_strength: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        self.records = records
        self.image_size = image_size  # (width, height)
        self.augment = augment
        self.jitter_strength = max(0.0, float(jitter_strength))
        self.noise_std = max(0.0, float(noise_std))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        with Image.open(record.frame_path) as img:
            img = img.convert("RGB")
            img = img.resize(self.image_size, Image.BILINEAR)
            if self.augment:
                img = self._apply_augmentation(img)
            image_np = np.asarray(img, dtype=np.float32) / 255.0
            if self.augment and self.noise_std > 0:
                noise = np.random.normal(loc=0.0, scale=self.noise_std, size=image_np.shape).astype(
                    np.float32
                )
                image_np = np.clip(image_np + noise, 0.0, 1.0)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - 0.5) / 0.5

        key_tensor = torch.from_numpy(record.key_target.copy())
        mouse_tensor = torch.from_numpy(record.mouse_target.copy())

        return {
            "image": image_tensor,
            "key_target": key_tensor,
            "mouse_target": mouse_tensor,
            "frame_path": str(record.frame_path),
            "session_name": record.session_name,
        }

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        if self.jitter_strength <= 0:
            return image
        low = max(0.4, 1.0 - self.jitter_strength)
        high = 1.0 + self.jitter_strength
        brightness = random.uniform(low, high)
        contrast = random.uniform(low, high)
        color = random.uniform(low, high)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(color)
        return image
