import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mss
import numpy as np
import torch
from PIL import Image
from pynput import keyboard, mouse

from training.model import DrivingNet


@dataclass
class PlayConfig:
    checkpoint: Path
    fps: int
    monitor_index: int
    image_width: int
    image_height: int
    mouse_scale: float
    key_threshold: float
    mouse_deadzone: float
    mouse_smoothing: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run imitation model to control the game in real time.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--fps", type=int, default=12, help="Inference/control loop FPS")
    parser.add_argument("--monitor-index", type=int, default=1, help="Monitor index used by mss")
    parser.add_argument("--image-width", type=int, default=None, help="Override image width from checkpoint")
    parser.add_argument("--image-height", type=int, default=None, help="Override image height from checkpoint")
    parser.add_argument("--mouse-scale", type=float, default=None, help="Override mouse scale from checkpoint")
    parser.add_argument("--key-threshold", type=float, default=0.5, help="Sigmoid threshold for W/A/S/D")
    parser.add_argument(
        "--mouse-deadzone",
        type=float,
        default=0.08,
        help="Ignore tiny normalized mouse predictions in [-1, 1]",
    )
    parser.add_argument(
        "--mouse-smoothing",
        type=float,
        default=0.35,
        help="EMA smoothing for normalized mouse prediction (0-1, higher = more reactive)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, checkpoint_config: dict[str, object]) -> PlayConfig:
    image_width = args.image_width or int(checkpoint_config.get("image_width", 256))
    image_height = args.image_height or int(checkpoint_config.get("image_height", 144))
    mouse_scale = args.mouse_scale or float(checkpoint_config.get("mouse_scale", 30.0))
    return PlayConfig(
        checkpoint=args.checkpoint,
        fps=args.fps,
        monitor_index=args.monitor_index,
        image_width=image_width,
        image_height=image_height,
        mouse_scale=mouse_scale,
        key_threshold=args.key_threshold,
        mouse_deadzone=args.mouse_deadzone,
        mouse_smoothing=max(0.0, min(1.0, args.mouse_smoothing)),
    )


class ImitationDriver:
    def __init__(self, cfg: PlayConfig) -> None:
        self.cfg = cfg
        self.running = False
        self._interval = 1.0 / max(1, cfg.fps)
        self._pressed: set[str] = set()
        self._smoothed_mouse = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._loop_started_at = 0.0

        payload = torch.load(cfg.checkpoint, map_location="cpu")
        self.key_order = [str(k).lower() for k in payload.get("key_order", ["w", "a", "s", "d"])]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DrivingNet().to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()

    def start(self) -> None:
        if self.running:
            print("[INFO] Autopilot ya esta activo.")
            return
        self.running = True
        self._steps = 0
        self._loop_started_at = time.time()
        self._smoothed_mouse = np.zeros(2, dtype=np.float32)
        print("[OK] Autopilot iniciado.")
        print("[INFO] Mantene el foco en el juego. Ctrl+9 para detener.")
        self._loop()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._release_all_keys()
        print("\n[OK] Autopilot detenido.")

    def _loop(self) -> None:
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

                self._apply_keys(key_probs)
                self._apply_mouse(mouse_vec)
                self._steps += 1

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

                delta = time.time() - tick
                sleep_time = self._interval - delta
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.cfg.image_width, self.cfg.image_height), Image.BILINEAR).convert("RGB")
        image_np = np.asarray(img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - 0.5) / 0.5
        return image_tensor.unsqueeze(0)

    def _apply_keys(self, key_probs: np.ndarray) -> None:
        desired_pressed = {
            key_name
            for key_name, prob in zip(self.key_order, key_probs, strict=False)
            if key_name in {"w", "a", "s", "d"} and float(prob) >= self.cfg.key_threshold
        }
        for key_name in ("w", "a", "s", "d"):
            if key_name in desired_pressed and key_name not in self._pressed:
                self.keyboard_controller.press(key_name)
                self._pressed.add(key_name)
            elif key_name not in desired_pressed and key_name in self._pressed:
                self.keyboard_controller.release(key_name)
                self._pressed.discard(key_name)

    def _apply_mouse(self, raw_mouse: np.ndarray) -> None:
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

    def _release_all_keys(self) -> None:
        for key_name in list(self._pressed):
            try:
                self.keyboard_controller.release(key_name)
            except Exception:
                pass
            self._pressed.discard(key_name)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = payload.get("config", {})
    cfg = build_config(args, checkpoint_config)
    driver = ImitationDriver(cfg)

    print("[INFO] Fase 3 - Conduccion autonoma por imitation learning")
    print(f"[INFO] Checkpoint: {cfg.checkpoint}")
    print(f"[INFO] Device: {driver.device}")
    print(f"[INFO] Input: {cfg.image_width}x{cfg.image_height} | mouse_scale={cfg.mouse_scale}")
    print("[INFO] Hotkeys globales:")
    print("       Ctrl+8 -> iniciar autopilot")
    print("       Ctrl+9 -> detener autopilot")
    print("       Ctrl+C -> salir")

    hotkeys = keyboard.GlobalHotKeys(
        {
            "<ctrl>+8": driver.start,
            "<ctrl>+9": driver.stop,
        }
    )

    try:
        hotkeys.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Cerrando Fase 3...")
    finally:
        driver.stop()
        hotkeys.stop()


if __name__ == "__main__":
    main()
