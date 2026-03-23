import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from training.dataset import KEY_ORDER, DrivingDataset, load_records, split_records
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
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--model-size", type=str, default="plus", choices=["base", "plus"])
    parser.add_argument("--mouse-scale", type=float, default=30.0)
    parser.add_argument("--mouse-loss-weight", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--train-jitter-strength",
        type=float,
        default=0.18,
        help="Random brightness/contrast/saturation jitter strength for train images.",
    )
    parser.add_argument(
        "--train-noise-std",
        type=float,
        default=0.01,
        help="Gaussian noise std added to train images in [0, 1] space.",
    )
    parser.add_argument("--balance-keys", dest="balance_keys", action="store_true")
    parser.add_argument("--no-balance-keys", dest="balance_keys", action="store_false")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(balance_keys=True, use_amp=True)
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


def compute_key_pos_weight(records: list, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    targets = np.stack([sample.key_target for sample in records], axis=0)
    pos = targets.sum(axis=0)
    neg = float(len(records)) - pos
    raw = neg / np.clip(pos, 1.0, None)
    clipped = np.clip(raw, 0.5, 10.0).astype(np.float32)
    ratios = {
        key_name: float(pos_i / max(1.0, float(len(records))))
        for key_name, pos_i in zip(KEY_ORDER, pos, strict=False)
    }
    pos_weight = torch.tensor(clipped, dtype=torch.float32, device=device)
    return pos_weight, ratios


def run_epoch(
    model: DrivingNet,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    bce_loss: nn.Module,
    mse_loss: nn.Module,
    mouse_loss_weight: float,
    progress_prefix: str,
    scaler: GradScaler | None,
    use_amp: bool,
    grad_clip_norm: float,
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
        images = batch["image"].to(device, non_blocking=True)
        key_target = batch["key_target"].to(device, non_blocking=True)
        mouse_target = batch["mouse_target"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            key_logits, mouse_pred = model(images)
            key_loss = bce_loss(key_logits, key_target)
            mouse_loss = mse_loss(mouse_pred, mouse_target)
            loss = key_loss + mouse_loss_weight * mouse_loss

        if is_train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
        train_records,
        image_size=(args.image_width, args.image_height),
        augment=True,
        jitter_strength=args.train_jitter_strength,
        noise_std=args.train_noise_std,
    )
    val_ds = DrivingDataset(
        val_records,
        image_size=(args.image_width, args.image_height),
        augment=False,
        jitter_strength=0.0,
        noise_std=0.0,
    )

    using_cuda = torch.cuda.is_available()
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=using_cuda,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=using_cuda,
        persistent_workers=persistent_workers,
    )

    device = torch.device("cuda" if using_cuda else "cpu")
    model = DrivingNet(model_size=args.model_size).to(device)
    model_param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr
    )
    key_pos_weight = None
    key_ratios = {}
    if args.balance_keys:
        key_pos_weight, key_ratios = compute_key_pos_weight(train_records, device=device)
        print("[INFO] Balanceo de teclas activo (pos_weight BCE):")
        print("       " + ", ".join(f"{k}={v:.3f}" for k, v in zip(KEY_ORDER, key_pos_weight.tolist())))
    else:
        targets = np.stack([sample.key_target for sample in train_records], axis=0)
        key_ratios = {
            key_name: float(pos / max(1, len(train_records)))
            for key_name, pos in zip(KEY_ORDER, targets.sum(axis=0), strict=False)
        }
    print("[INFO] Distribucion de teclas train:")
    print("       " + ", ".join(f"{k}={key_ratios[k]:.3f}" for k in KEY_ORDER))

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=key_pos_weight)
    mse_loss = nn.MSELoss()
    scaler = GradScaler(enabled=bool(args.use_amp and device.type == "cuda"))

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    training_start = time.perf_counter()

    print(
        f"[INFO] Entrenando en device: {device} | amp={scaler.is_enabled()} | "
        f"model_size={args.model_size} | params={model_param_count:,}"
    )
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
            scaler=scaler,
            use_amp=args.use_amp,
            grad_clip_norm=args.grad_clip_norm,
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
                scaler=None,
                use_amp=args.use_amp,
                grad_clip_norm=0.0,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
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
            "lr": current_lr,
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
            f"lr={row['lr']:.6f} "
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
            "key_pos_weight": key_pos_weight.detach().cpu().tolist() if key_pos_weight is not None else None,
            "key_ratios": key_ratios,
            "model_size": args.model_size,
            "model_param_count": model_param_count,
        }
        torch.save(checkpoint_payload, run_dir / "last.pt")

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            torch.save(checkpoint_payload, run_dir / "best.pt")

        scheduler.step()

    summary = {
        "config": config,
        "device": str(device),
        "total_samples": len(records),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "best_val_loss": best_val_loss,
        "key_ratios": key_ratios,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Entrenamiento finalizado. Artefactos en: {run_dir}")
    print(f"[OK] Best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
