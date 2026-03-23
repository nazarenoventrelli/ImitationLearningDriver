import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.dataset import DrivingDataset, load_records, split_records
from training.model import DrivingNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train imitation model for GTA SA driving.")
    parser.add_argument("--captures-dir", type=Path, default=Path("captures"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=144)
    parser.add_argument("--mouse-scale", type=float, default=30.0)
    parser.add_argument("--mouse-loss-weight", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def config_to_dict(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_metrics(
    key_logits: torch.Tensor,
    key_target: torch.Tensor,
    mouse_pred: torch.Tensor,
    mouse_target: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        key_pred = (torch.sigmoid(key_logits) > 0.5).float()
        key_acc = (key_pred == key_target).float().mean().item()
        mouse_mae = torch.abs(mouse_pred - mouse_target).mean().item()
    return {"key_acc": key_acc, "mouse_mae": mouse_mae}


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "??:??:??"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_epoch(
    model: DrivingNet,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    bce_loss: nn.Module,
    mse_loss: nn.Module,
    mouse_loss_weight: float,
    progress_prefix: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_steps = len(dataloader)
    if total_steps == 0:
        raise RuntimeError("Dataloader vacio, no se pudo procesar ninguna iteracion.")

    total_loss = 0.0
    total_key_loss = 0.0
    total_mouse_loss = 0.0
    total_key_acc = 0.0
    total_mouse_mae = 0.0
    epoch_start = time.perf_counter()
    update_every = max(1, total_steps // 20)

    for step_idx, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        key_target = batch["key_target"].to(device)
        mouse_target = batch["mouse_target"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        key_logits, mouse_pred = model(images)
        key_loss = bce_loss(key_logits, key_target)
        mouse_loss = mse_loss(mouse_pred, mouse_target)
        loss = key_loss + mouse_loss_weight * mouse_loss

        if is_train:
            loss.backward()
            optimizer.step()

        metrics = batch_metrics(key_logits, key_target, mouse_pred, mouse_target)

        total_loss += loss.item()
        total_key_loss += key_loss.item()
        total_mouse_loss += mouse_loss.item()
        total_key_acc += metrics["key_acc"]
        total_mouse_mae += metrics["mouse_mae"]
        if step_idx == 1 or step_idx % update_every == 0 or step_idx == total_steps:
            elapsed = time.perf_counter() - epoch_start
            avg_step_time = elapsed / step_idx
            eta = avg_step_time * (total_steps - step_idx)
            pct = (100.0 * step_idx) / total_steps
            print(
                f"\r{progress_prefix} {pct:6.2f}% ({step_idx}/{total_steps}) "
                f"ETA epoca: {format_duration(eta)}",
                end="",
                flush=True,
            )
    print()

    epoch_duration = time.perf_counter() - epoch_start

    return {
        "loss": total_loss / total_steps,
        "key_loss": total_key_loss / total_steps,
        "mouse_loss": total_mouse_loss / total_steps,
        "key_acc": total_key_acc / total_steps,
        "mouse_mae": total_mouse_mae / total_steps,
        "duration_sec": epoch_duration,
    }


def main() -> None:
    args = parse_args()
    config = config_to_dict(args)
    set_seed(args.seed)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Cargando dataset...")
    records = load_records(args.captures_dir, mouse_scale=args.mouse_scale)
    train_records, val_records = split_records(records, val_ratio=args.val_ratio, seed=args.seed)

    print(f"[INFO] Samples totales: {len(records)}")
    print(f"[INFO] Train: {len(train_records)} | Val: {len(val_records)}")

    train_ds = DrivingDataset(
        train_records, image_size=(args.image_width, args.image_height)
    )
    val_ds = DrivingDataset(
        val_records, image_size=(args.image_width, args.image_height)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrivingNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    training_start = time.perf_counter()

    print(f"[INFO] Entrenando en device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            bce_loss=bce_loss,
            mse_loss=mse_loss,
            mouse_loss_weight=args.mouse_loss_weight,
            progress_prefix=f"[TRAIN {epoch:02d}/{args.epochs:02d}]",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                bce_loss=bce_loss,
                mse_loss=mse_loss,
                mouse_loss_weight=args.mouse_loss_weight,
                progress_prefix=f"[VAL   {epoch:02d}/{args.epochs:02d}]",
            )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_key_loss": train_metrics["key_loss"],
            "train_mouse_loss": train_metrics["mouse_loss"],
            "train_key_acc": train_metrics["key_acc"],
            "train_mouse_mae": train_metrics["mouse_mae"],
            "val_loss": val_metrics["loss"],
            "val_key_loss": val_metrics["key_loss"],
            "val_mouse_loss": val_metrics["mouse_loss"],
            "val_key_acc": val_metrics["key_acc"],
            "val_mouse_mae": val_metrics["mouse_mae"],
            "epoch_train_sec": train_metrics["duration_sec"],
            "epoch_val_sec": val_metrics["duration_sec"],
        }
        history.append(row)

        elapsed_total = time.perf_counter() - training_start
        avg_epoch_sec = elapsed_total / epoch
        eta_total = avg_epoch_sec * (args.epochs - epoch)
        estimated_total = avg_epoch_sec * args.epochs

        print(
            f"[EPOCH {epoch:02d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"train_key_acc={row['train_key_acc']:.3f} "
            f"val_key_acc={row['val_key_acc']:.3f} "
            f"train_mouse_mae={row['train_mouse_mae']:.3f} "
            f"val_mouse_mae={row['val_mouse_mae']:.3f} "
            f"tiempo_epoca={format_duration(row['epoch_train_sec'] + row['epoch_val_sec'])}"
        )
        print(
            f"[TIME] transcurrido={format_duration(elapsed_total)} "
            f"restante_estimado={format_duration(eta_total)} "
            f"total_estimado={format_duration(estimated_total)}"
        )

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": row["val_loss"],
            "key_order": ["w", "a", "s", "d"],
        }
        torch.save(checkpoint_payload, run_dir / "last.pt")

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            torch.save(checkpoint_payload, run_dir / "best.pt")

    summary = {
        "config": config,
        "device": str(device),
        "total_samples": len(records),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "best_val_loss": best_val_loss,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Entrenamiento finalizado. Artefactos en: {run_dir}")
    print(f"[OK] Best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
