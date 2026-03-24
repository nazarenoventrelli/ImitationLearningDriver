import argparse
import time
import ctypes
from ctypes import Structure, Union, byref, c_long, c_ulong
from ctypes import sizeof as c_sizeof
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys

import mss
import numpy as np
import torch
from PIL import Image
from pynput import keyboard, mouse

from training.model import DrivingNet

try:
    from interception.constants import MouseFlag as InterceptionMouseFlag
    from interception.interception import Interception
    from interception.strokes import MouseStroke as InterceptionMouseStroke
except ImportError:
    Interception = None
    InterceptionMouseStroke = None
    InterceptionMouseFlag = None


@dataclass
class PlayConfig:
    checkpoint: Path
    fps: int
    monitor_index: int
    image_width: int
    image_height: int
    mouse_scale: float
    mouse_backend: str
    seq_len: int
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
    parser.add_argument(
        "--mouse-backend",
        type=str,
        default="auto",
        choices=["auto", "pynput", "sendinput", "interception"],
        help="Mouse injection backend. interception is driver-level and may work better in games.",
    )
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
    image_width = args.image_width or int(checkpoint_config.get("image_width", 512))
    image_height = args.image_height or int(checkpoint_config.get("image_height", 288))
    mouse_scale = args.mouse_scale or float(checkpoint_config.get("mouse_scale", 30.0))
    seq_len = max(1, int(checkpoint_config.get("seq_len", 1)))
    return PlayConfig(
        checkpoint=args.checkpoint,
        fps=args.fps,
        monitor_index=args.monitor_index,
        image_width=image_width,
        image_height=image_height,
        mouse_scale=mouse_scale,
        mouse_backend=args.mouse_backend,
        seq_len=seq_len,
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
        self._mouse_carry = np.zeros(2, dtype=np.float32)
        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=max(1, cfg.seq_len))
        self._steps = 0
        self._loop_started_at = 0.0

        payload = torch.load(cfg.checkpoint, map_location="cpu")
        self.key_order = [str(k).lower() for k in payload.get("key_order", ["w", "a", "s", "d"])]
        self.model_size = str(payload.get("model_size") or payload.get("config", {}).get("model_size", "base"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = max(1, int(payload.get("seq_len", payload.get("config", {}).get("seq_len", cfg.seq_len))))
        self.model = DrivingNet(model_size=self.model_size, seq_len=self.seq_len).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=False)
        self.model.eval()

        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        self.mouse_backend = self._resolve_mouse_backend(cfg.mouse_backend)
        if self.mouse_backend == "sendinput" and not _sendinput_available():
            print("[WARN] SendInput no disponible. Se usa backend pynput.")
            self.mouse_backend = "pynput"
        if self.mouse_backend == "interception" and not _interception_available():
            print("[WARN] Interception no disponible/driver ausente. Se usa backend sendinput.")
            self.mouse_backend = "sendinput" if _sendinput_available() else "pynput"

    def start(self) -> None:
        if self.running:
            print("[INFO] Autopilot ya esta activo.")
            return
        self.running = True
        self._steps = 0
        self._loop_started_at = time.time()
        self._smoothed_mouse = np.zeros(2, dtype=np.float32)
        self._mouse_carry = np.zeros(2, dtype=np.float32)
        self._frame_buffer.clear()
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
                frame_tensor = self._preprocess_image(pil_img)
                image_tensor = self._build_model_input(frame_tensor).to(self.device)

                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        key_logits = outputs[0]
                        mouse_pred = outputs[1]
                    else:
                        raise RuntimeError("Salida invalida del modelo durante inferencia.")
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
        return image_tensor

    def _build_model_input(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        self._frame_buffer.append(frame_tensor)
        while len(self._frame_buffer) < self.seq_len:
            self._frame_buffer.appendleft(frame_tensor)
        if self.seq_len == 1:
            return self._frame_buffer[-1].unsqueeze(0)
        seq_tensor = torch.stack(list(self._frame_buffer), dim=0)
        return seq_tensor.unsqueeze(0)

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

        scaled_dx = mx * self.cfg.mouse_scale
        scaled_dy = my * self.cfg.mouse_scale
        self._mouse_carry[0] += scaled_dx
        self._mouse_carry[1] += scaled_dy
        dx = int(np.trunc(self._mouse_carry[0]))
        dy = int(np.trunc(self._mouse_carry[1]))
        self._mouse_carry[0] -= dx
        self._mouse_carry[1] -= dy
        if dx != 0 or dy != 0:
            self._move_mouse(dx, dy)

    def _release_all_keys(self) -> None:
        for key_name in list(self._pressed):
            try:
                self.keyboard_controller.release(key_name)
            except Exception:
                pass
            self._pressed.discard(key_name)

    def _move_mouse(self, dx: int, dy: int) -> None:
        if self.mouse_backend == "sendinput":
            _sendinput_move(dx, dy)
        elif self.mouse_backend == "interception":
            _interception_move(dx, dy)
        else:
            self.mouse_controller.move(dx, dy)

    @staticmethod
    def _resolve_mouse_backend(requested: str) -> str:
        if requested == "auto":
            if _interception_available():
                return "interception"
            return "sendinput" if sys.platform == "win32" else "pynput"
        return requested


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
    print(f"[INFO] Model size: {driver.model_size}")
    print(f"[INFO] Temporal frames: {driver.seq_len}")
    print(f"[INFO] Input: {cfg.image_width}x{cfg.image_height} | mouse_scale={cfg.mouse_scale}")
    print(f"[INFO] Mouse backend: {driver.mouse_backend}")
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


# Windows SendInput backend
MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0


class MOUSEINPUT(Structure):
    _fields_ = [
        ("dx", c_long),
        ("dy", c_long),
        ("mouseData", c_ulong),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", c_ulong),
    ]


class INPUTUNION(Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(Structure):
    _fields_ = [("type", c_ulong), ("union", INPUTUNION)]


def _sendinput_available() -> bool:
    return sys.platform == "win32"


def _sendinput_move(dx: int, dy: int) -> None:
    if not _sendinput_available():
        return
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0)
    ctypes.windll.user32.SendInput(1, byref(inp), c_sizeof(INPUT))


def _interception_available() -> bool:
    if Interception is None or InterceptionMouseStroke is None or InterceptionMouseFlag is None:
        return False
    try:
        ctx = _get_interception_context()
        return ctx is not None and bool(ctx.valid)
    except Exception:
        return False


_INTERCEPTION_CONTEXT = None


def _get_interception_context():
    global _INTERCEPTION_CONTEXT
    if _INTERCEPTION_CONTEXT is None and Interception is not None:
        try:
            _INTERCEPTION_CONTEXT = Interception()
        except Exception:
            _INTERCEPTION_CONTEXT = None
    return _INTERCEPTION_CONTEXT


def _interception_move(dx: int, dy: int) -> None:
    ctx = _get_interception_context()
    if ctx is None or not ctx.valid:
        return
    stroke = InterceptionMouseStroke(InterceptionMouseFlag.MOUSE_MOVE_RELATIVE, 0, 0, dx, dy)
    ctx.send(ctx.mouse, stroke)


if __name__ == "__main__":
    main()
