import argparse
import csv
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mss
import numpy as np
import torch
from PIL import Image
from pynput import keyboard, mouse

from training.model import DrivingNet

try:
    import winsound
except ImportError:
    winsound = None


@dataclass
class InferenceConfig:
    checkpoint: Path
    fps: int = 12
    monitor_index: int = 1
    image_width: int = 512
    image_height: int = 288
    mouse_scale: float = 30.0
    key_threshold: float = 0.5
    reverse_threshold: float = 0.78
    reverse_hold_steps: int = 4
    reverse_margin_vs_forward: float = 0.12
    mouse_deadzone: float = 0.08
    mouse_smoothing: float = 0.35
    log_dir: Path = Path("artifacts/inference_logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run imitation model in real time with Ctrl+1 start and Ctrl+2 stop."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--fps", type=int, default=12, help="Inference/control loop FPS")
    parser.add_argument("--monitor-index", type=int, default=1, help="Monitor index used by mss")
    parser.add_argument("--image-width", type=int, default=None, help="Override image width from checkpoint")
    parser.add_argument("--image-height", type=int, default=None, help="Override image height from checkpoint")
    parser.add_argument("--mouse-scale", type=float, default=None, help="Override mouse scale from checkpoint")
    parser.add_argument("--key-threshold", type=float, default=0.5, help="Sigmoid threshold for W/A/S/D")
    parser.add_argument(
        "--reverse-threshold",
        type=float,
        default=0.78,
        help="Threshold for allowing reverse (S). Higher value = less accidental reverse.",
    )
    parser.add_argument(
        "--reverse-hold-steps",
        type=int,
        default=4,
        help="Consecutive steps S must stay above threshold before pressing reverse.",
    )
    parser.add_argument(
        "--reverse-margin-vs-forward",
        type=float,
        default=0.12,
        help="Extra margin required for S over W to allow reverse.",
    )
    parser.add_argument("--mouse-deadzone", type=float, default=0.08, help="Ignore tiny mouse predictions")
    parser.add_argument("--mouse-smoothing", type=float, default=0.35, help="EMA smoothing in [0, 1]")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("artifacts/inference_logs"),
        help="Directory where inference CSV logs are written",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, checkpoint_config: dict[str, object]) -> InferenceConfig:
    image_width = args.image_width or int(checkpoint_config.get("image_width", 512))
    image_height = args.image_height or int(checkpoint_config.get("image_height", 288))
    mouse_scale = args.mouse_scale or float(checkpoint_config.get("mouse_scale", 30.0))
    return InferenceConfig(
        checkpoint=args.checkpoint,
        fps=args.fps,
        monitor_index=args.monitor_index,
        image_width=image_width,
        image_height=image_height,
        mouse_scale=mouse_scale,
        key_threshold=args.key_threshold,
        reverse_threshold=args.reverse_threshold,
        reverse_hold_steps=max(1, int(args.reverse_hold_steps)),
        reverse_margin_vs_forward=max(0.0, float(args.reverse_margin_vs_forward)),
        mouse_deadzone=args.mouse_deadzone,
        mouse_smoothing=max(0.0, min(1.0, args.mouse_smoothing)),
        log_dir=args.log_dir,
    )


class InferenceDriver:
    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self.running = False
        self._interval = 1.0 / max(1, cfg.fps)
        self._pressed: set[str] = set()
        self._smoothed_mouse = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._loop_started_at = 0.0
        self._thread: threading.Thread | None = None
        self._log_file = None
        self._log_writer: csv.writer | None = None
        self._log_path: Path | None = None
        self._prob_sums = np.zeros(4, dtype=np.float64)
        self._reverse_streak = 0

        payload = torch.load(cfg.checkpoint, map_location="cpu")
        self.key_order = [str(k).lower() for k in payload.get("key_order", ["w", "a", "s", "d"])]
        self.model_size = str(payload.get("model_size") or payload.get("config", {}).get("model_size", "base"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DrivingNet(model_size=self.model_size).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()

    def start_inference(self) -> None:
        if self.running:
            self._play_feedback("already_running")
            print("[INFO] La inferencia ya esta activa.")
            return

        self.running = True
        self._steps = 0
        self._loop_started_at = time.time()
        self._smoothed_mouse = np.zeros(2, dtype=np.float32)
        self._prob_sums = np.zeros(4, dtype=np.float64)
        self._reverse_streak = 0
        self._open_log()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

        self._play_feedback("start")
        print("[OK] Inferencia iniciada.")
        print("[INFO] Mantene el foco en el juego. Ctrl+2 para detener.")

    def stop_inference(self) -> None:
        if not self.running:
            self._play_feedback("already_stopped")
            print("[INFO] La inferencia no esta activa.")
            return

        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._release_all_keys()
        self._close_log()

        self._play_feedback("stop")
        if self._steps > 0:
            avg_probs = self._prob_sums / self._steps
            print(
                "[INFO] Promedio probs teclas: "
                f"W={avg_probs[0]:.3f} A={avg_probs[1]:.3f} S={avg_probs[2]:.3f} D={avg_probs[3]:.3f}"
            )
        if self._log_path:
            print(f"[INFO] Log guardado en: {self._log_path}")
        print("\n[OK] Inferencia detenida.")

    def _inference_loop(self) -> None:
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if self.cfg.monitor_index >= len(monitors):
                    print(
                        f"[WARN] monitor_index={self.cfg.monitor_index} no existe. "
                        "Se usa monitor principal (1)."
                    )
                    monitor = monitors[1]
                else:
                    monitor = monitors[self.cfg.monitor_index]

                while self.running:
                    tick = time.time()
                    raw = sct.grab(monitor)
                    pil_img = Image.frombytes("RGB", raw.size, raw.rgb)
                    image_tensor = self._preprocess_image(pil_img).to(self.device)

                    with torch.no_grad():
                        key_logits, mouse_pred = self.model(image_tensor)
                        key_probs = torch.sigmoid(key_logits[0]).detach().cpu().numpy()
                        mouse_vec = mouse_pred[0].detach().cpu().numpy()

                    desired_pressed, allow_reverse = self._apply_keys(key_probs)
                    mx, my, dx, dy = self._apply_mouse(mouse_vec)
                    self._steps += 1
                    self._prob_sums += key_probs.astype(np.float64)

                    elapsed = time.time() - self._loop_started_at
                    mean_hz = self._steps / elapsed if elapsed > 0 else 0.0
                    pressed_preview = "+".join(sorted(self._pressed)) if self._pressed else "-"
                    print(
                        f"\r[RUN] step={self._steps} keys={pressed_preview} "
                        f"mouse=({self._smoothed_mouse[0]:+.2f},{self._smoothed_mouse[1]:+.2f}) "
                        f"avg_hz={mean_hz:.1f}",
                        end="",
                        flush=True,
                    )
                    self._log_step(
                        elapsed=elapsed,
                        key_probs=key_probs,
                        desired_pressed=desired_pressed,
                        raw_mouse=mouse_vec,
                        mx=mx,
                        my=my,
                        dx=dx,
                        dy=dy,
                        mean_hz=mean_hz,
                        allow_reverse=allow_reverse,
                    )

                    delta = time.time() - tick
                    sleep_time = self._interval - delta
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except Exception as exc:
            self.running = False
            self._release_all_keys()
            self._close_log()
            self._play_feedback("error")
            print(f"\n[ERROR] Fallo en inferencia: {exc}")

    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.cfg.image_width, self.cfg.image_height), Image.BILINEAR).convert("RGB")
        image_np = np.asarray(img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - 0.5) / 0.5
        return image_tensor.unsqueeze(0)

    def _apply_keys(self, key_probs: np.ndarray) -> tuple[set[str], bool]:
        key_prob_map = {str(key_name): float(prob) for key_name, prob in zip(self.key_order, key_probs, strict=False)}
        p_w = key_prob_map.get("w", 0.0)
        p_a = key_prob_map.get("a", 0.0)
        p_s = key_prob_map.get("s", 0.0)
        p_d = key_prob_map.get("d", 0.0)

        desired_pressed: set[str] = set()
        if p_w >= self.cfg.key_threshold:
            desired_pressed.add("w")
        if p_a >= self.cfg.key_threshold:
            desired_pressed.add("a")
        if p_d >= self.cfg.key_threshold:
            desired_pressed.add("d")

        if p_s >= self.cfg.reverse_threshold:
            self._reverse_streak += 1
        else:
            self._reverse_streak = 0
        allow_reverse = (
            self._reverse_streak >= self.cfg.reverse_hold_steps
            and p_s >= (p_w + self.cfg.reverse_margin_vs_forward)
        )
        if allow_reverse:
            desired_pressed.add("s")
            desired_pressed.discard("w")

        for key_name in ("w", "a", "s", "d"):
            if key_name in desired_pressed and key_name not in self._pressed:
                self.keyboard_controller.press(key_name)
                self._pressed.add(key_name)
            elif key_name not in desired_pressed and key_name in self._pressed:
                self.keyboard_controller.release(key_name)
                self._pressed.discard(key_name)
        return desired_pressed, allow_reverse

    def _apply_mouse(self, raw_mouse: np.ndarray) -> tuple[float, float, int, int]:
        alpha = self.cfg.mouse_smoothing
        self._smoothed_mouse = (1.0 - alpha) * self._smoothed_mouse + alpha * raw_mouse
        mx = float(self._smoothed_mouse[0])
        my = float(self._smoothed_mouse[1])
        if abs(mx) < self.cfg.mouse_deadzone:
            mx = 0.0
        if abs(my) < self.cfg.mouse_deadzone:
            my = 0.0

        dx = int(round(mx * self.cfg.mouse_scale))
        dy = int(round(my * self.cfg.mouse_scale))
        if dx != 0 or dy != 0:
            self.mouse_controller.move(dx, dy)
        return mx, my, dx, dy

    def _release_all_keys(self) -> None:
        for key_name in list(self._pressed):
            try:
                self.keyboard_controller.release(key_name)
            except Exception:
                pass
            self._pressed.discard(key_name)

    @staticmethod
    def _play_feedback(event: str) -> None:
        if winsound is None:
            return
        sounds = {
            "start": [(880, 120), (1200, 140)],
            "stop": [(700, 120), (450, 160)],
            "already_running": [(500, 120)],
            "already_stopped": [(500, 120)],
            "error": [(350, 180), (280, 220)],
        }
        for freq, dur in sounds.get(event, []):
            winsound.Beep(freq, dur)

    def _open_log(self) -> None:
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self.cfg.log_dir / f"inference_{run_stamp}.csv"
        self._log_file = self._log_path.open("w", newline="", encoding="utf-8")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            [
                "step",
                "wall_time",
                "elapsed_sec",
                "mean_hz",
                "prob_w",
                "prob_a",
                "prob_s",
                "prob_d",
                "desired_keys",
                "pressed_keys",
                "raw_mouse_x",
                "raw_mouse_y",
                "smoothed_mouse_x",
                "smoothed_mouse_y",
                "applied_dx",
                "applied_dy",
                "reverse_streak",
                "allow_reverse",
            ]
        )

    def _close_log(self) -> None:
        if self._log_file:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None
            self._log_writer = None

    def _log_step(
        self,
        elapsed: float,
        key_probs: np.ndarray,
        desired_pressed: set[str],
        raw_mouse: np.ndarray,
        mx: float,
        my: float,
        dx: int,
        dy: int,
        mean_hz: float,
        allow_reverse: bool,
    ) -> None:
        if self._log_writer is None:
            return
        ordered_probs = []
        for key_name in ("w", "a", "s", "d"):
            if key_name in self.key_order:
                idx = self.key_order.index(key_name)
                ordered_probs.append(float(key_probs[idx]))
            else:
                ordered_probs.append(float("nan"))

        self._log_writer.writerow(
            [
                self._steps,
                datetime.now().isoformat(timespec="milliseconds"),
                f"{elapsed:.3f}",
                f"{mean_hz:.2f}",
                f"{ordered_probs[0]:.6f}",
                f"{ordered_probs[1]:.6f}",
                f"{ordered_probs[2]:.6f}",
                f"{ordered_probs[3]:.6f}",
                "+".join(sorted(desired_pressed)),
                "+".join(sorted(self._pressed)),
                f"{float(raw_mouse[0]):.6f}",
                f"{float(raw_mouse[1]):.6f}",
                f"{mx:.6f}",
                f"{my:.6f}",
                dx,
                dy,
                self._reverse_streak,
                int(allow_reverse),
            ]
        )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = payload.get("config", {})
    cfg = build_config(args, checkpoint_config)
    driver = InferenceDriver(cfg)

    print("[INFO] Herramienta de inferencia lista.")
    print(f"[INFO] Checkpoint: {cfg.checkpoint}")
    print(f"[INFO] Device: {driver.device}")
    print(f"[INFO] Model size: {driver.model_size}")
    print(f"[INFO] Input: {cfg.image_width}x{cfg.image_height} | mouse_scale={cfg.mouse_scale}")
    print(f"[INFO] Logs: {cfg.log_dir}")
    print("[INFO] Ctrl+1: iniciar inferencia")
    print("[INFO] Ctrl+2: detener inferencia")
    print("[INFO] Ctrl+C: salir")

    hotkeys = keyboard.GlobalHotKeys(
        {
            "<ctrl>+1": driver.start_inference,
            "<ctrl>+2": driver.stop_inference,
        }
    )

    try:
        hotkeys.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Cerrando herramienta de inferencia...")
    finally:
        driver.stop_inference()
        hotkeys.stop()


if __name__ == "__main__":
    main()
