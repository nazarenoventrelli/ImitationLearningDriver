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
MODE_ORDER = ("normal", "turn", "correction", "reverse")


@dataclass
class SampleRecord:
    frame_path: Path
    key_target: np.ndarray
    mouse_target: np.ndarray
    session_name: str
    frame_id: int


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


def _extract_frame_id(frame_name: str) -> int:
    stem = Path(frame_name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return -1
    return int(digits)


def _derive_mode_target(
    key_target: np.ndarray,
    mouse_target: np.ndarray,
    correction_threshold: float = 0.40,
    turn_threshold: float = 0.22,
) -> int:
    if key_target[2] > 0.5:
        return MODE_ORDER.index("reverse")
    abs_steer = float(abs(mouse_target[0]))
    if abs_steer >= correction_threshold:
        return MODE_ORDER.index("correction")
    if abs_steer >= turn_threshold or key_target[1] > 0.5 or key_target[3] > 0.5:
        return MODE_ORDER.index("turn")
    return MODE_ORDER.index("normal")


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
                        frame_id=_extract_frame_id(frame_name),
                    )
                )

    if not records:
        raise RuntimeError("No se pudieron construir samples validos a partir del dataset.")
    records.sort(key=lambda r: (r.session_name, r.frame_id))
    return records


def split_records(
    records: list[SampleRecord],
    val_ratio: float,
    seed: int,
    split_mode: str = "segment",
    segment_length: int = 240,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio debe estar entre 0 y 1.")

    if split_mode == "random":
        indices = list(range(len(records)))
        random.Random(seed).shuffle(indices)
        val_count = max(1, int(len(records) * val_ratio))
        val_indices = set(indices[:val_count])
        train_records = [records[i] for i in indices[val_count:]]
        val_records = [records[i] for i in indices[:val_count]]
    elif split_mode == "session":
        sessions: dict[str, list[SampleRecord]] = {}
        for record in records:
            sessions.setdefault(record.session_name, []).append(record)
        for session_records in sessions.values():
            session_records.sort(key=lambda r: r.frame_id)
        session_names = list(sessions.keys())
        random.Random(seed).shuffle(session_names)
        val_target = max(1, int(len(records) * val_ratio))
        val_records = []
        train_records = []
        val_count = 0
        for session_name in session_names:
            block = sessions[session_name]
            if val_count < val_target:
                val_records.extend(block)
                val_count += len(block)
            else:
                train_records.extend(block)
    elif split_mode == "segment":
        if segment_length <= 1:
            raise ValueError("segment_length debe ser > 1 para split_mode=segment.")
        segments: list[list[SampleRecord]] = []
        by_session: dict[str, list[SampleRecord]] = {}
        for record in records:
            by_session.setdefault(record.session_name, []).append(record)
        for session_records in by_session.values():
            session_records.sort(key=lambda r: r.frame_id)
            for start in range(0, len(session_records), segment_length):
                chunk = session_records[start : start + segment_length]
                if chunk:
                    segments.append(chunk)
        random.Random(seed).shuffle(segments)
        val_target = max(1, int(len(records) * val_ratio))
        val_records = []
        train_records = []
        val_count = 0
        for chunk in segments:
            if val_count < val_target:
                val_records.extend(chunk)
                val_count += len(chunk)
            else:
                train_records.extend(chunk)
    else:
        raise ValueError("split_mode invalido. Opciones: random, session, segment.")

    if not train_records or not val_records:
        raise RuntimeError("Split invalido: train o val quedo vacio.")
    return train_records, val_records


def smooth_mouse_targets(records: list[SampleRecord], window_size: int, deadzone: float) -> list[SampleRecord]:
    if window_size <= 1 and deadzone <= 0:
        return records

    updated: list[SampleRecord] = []
    by_session: dict[str, list[SampleRecord]] = {}
    for record in records:
        by_session.setdefault(record.session_name, []).append(record)

    for session_name, session_records in by_session.items():
        session_records.sort(key=lambda r: r.frame_id)
        mouse_arr = np.stack([r.mouse_target for r in session_records], axis=0)
        smoothed = mouse_arr.copy()
        if window_size > 1:
            radius = window_size // 2
            padded = np.pad(mouse_arr, ((radius, radius), (0, 0)), mode="edge")
            for i in range(len(mouse_arr)):
                smoothed[i] = padded[i : i + window_size].mean(axis=0)
        if deadzone > 0:
            smoothed[np.abs(smoothed) < deadzone] = 0.0

        for idx, record in enumerate(session_records):
            updated.append(
                SampleRecord(
                    frame_path=record.frame_path,
                    key_target=record.key_target.copy(),
                    mouse_target=smoothed[idx].astype(np.float32),
                    session_name=session_name,
                    frame_id=record.frame_id,
                )
            )
    return updated


class DrivingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[SampleRecord],
        image_size: tuple[int, int],
        seq_len: int = 1,
        frame_stride: int = 1,
        augment: bool = False,
        jitter_strength: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        self.records = records
        self.image_size = image_size  # (width, height)
        self.seq_len = max(1, int(seq_len))
        self.frame_stride = max(1, int(frame_stride))
        self.augment = augment
        self.jitter_strength = max(0.0, float(jitter_strength))
        self.noise_std = max(0.0, float(noise_std))
        self.sequence_indices = self._build_sequence_indices()

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_indices = self.sequence_indices[idx]
        seq_records = [self.records[i] for i in seq_indices]
        endpoint = seq_records[-1]

        aug_params = self._sample_aug_params() if self.augment else None
        frame_tensors = []
        for record in seq_records:
            with Image.open(record.frame_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.image_size, Image.BILINEAR)
                if aug_params is not None:
                    img = self._apply_augmentation(img, aug_params)
                image_np = np.asarray(img, dtype=np.float32) / 255.0
                if self.augment and self.noise_std > 0:
                    noise = np.random.normal(loc=0.0, scale=self.noise_std, size=image_np.shape).astype(
                        np.float32
                    )
                    image_np = np.clip(image_np + noise, 0.0, 1.0)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
            image_tensor = (image_tensor - 0.5) / 0.5
            frame_tensors.append(image_tensor)

        if self.seq_len == 1:
            image = frame_tensors[0]
        else:
            image = torch.stack(frame_tensors, dim=0)

        mode_target = _derive_mode_target(endpoint.key_target, endpoint.mouse_target)
        key_tensor = torch.from_numpy(endpoint.key_target.copy())
        mouse_tensor = torch.from_numpy(endpoint.mouse_target.copy())

        return {
            "image": image,
            "key_target": key_tensor,
            "mouse_target": mouse_tensor,
            "mode_target": torch.tensor(mode_target, dtype=torch.long),
            "frame_path": str(endpoint.frame_path),
            "session_name": endpoint.session_name,
        }

    def get_endpoint_records(self) -> list[SampleRecord]:
        return [self.records[idxs[-1]] for idxs in self.sequence_indices]

    def _build_sequence_indices(self) -> list[list[int]]:
        by_session: dict[str, list[tuple[int, int]]] = {}
        for i, record in enumerate(self.records):
            by_session.setdefault(record.session_name, []).append((record.frame_id, i))

        all_sequences: list[list[int]] = []
        for session_items in by_session.values():
            session_items.sort(key=lambda item: item[0])
            session_indices = [item[1] for item in session_items]
            session_frame_ids = [item[0] for item in session_items]
            if len(session_indices) < self.seq_len:
                continue
            for pos in range((self.seq_len - 1) * self.frame_stride, len(session_indices)):
                current = []
                valid = True
                for back in range(self.seq_len - 1, -1, -1):
                    source_pos = pos - back * self.frame_stride
                    if source_pos < 0:
                        valid = False
                        break
                    current.append(session_indices[source_pos])
                if valid and all(fid >= 0 for fid in session_frame_ids):
                    expected_gap = self.frame_stride
                    for j in range(1, len(current)):
                        curr_fid = self.records[current[j]].frame_id
                        prev_fid = self.records[current[j - 1]].frame_id
                        if (curr_fid - prev_fid) != expected_gap:
                            valid = False
                            break
                if valid:
                    all_sequences.append(current)
        if not all_sequences:
            raise RuntimeError("No hay secuencias validas. Ajusta seq_len/frame_stride o dataset.")
        return all_sequences

    def _sample_aug_params(self) -> tuple[float, float, float]:
        if self.jitter_strength <= 0:
            return (1.0, 1.0, 1.0)
        low = max(0.4, 1.0 - self.jitter_strength)
        high = 1.0 + self.jitter_strength
        brightness = random.uniform(low, high)
        contrast = random.uniform(low, high)
        color = random.uniform(low, high)
        return brightness, contrast, color

    def _apply_augmentation(self, image: Image.Image, params: tuple[float, float, float]) -> Image.Image:
        brightness, contrast, color = params
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(color)
        return image
