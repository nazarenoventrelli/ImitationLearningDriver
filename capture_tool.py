import csv
import json
import threading
import time
import winsound
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import mss
from PIL import Image
from pynput import keyboard, mouse


@dataclass
class CaptureConfig:
    fps: int = 10
    output_root: Path = Path("captures")
    monitor_index: int = 1
    jpeg_quality: int = 85


class DrivingDataCapture:
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.running = False
        self.current_session_path: Optional[Path] = None
        self.frames_path: Optional[Path] = None
        self.csv_file = None
        self.csv_writer: Optional[csv.writer] = None
        self.capture_thread: Optional[threading.Thread] = None

        self._pressed_keys: set[str] = set()
        self._keys_lock = threading.Lock()
        self._mouse_lock = threading.Lock()
        self._pressed_mouse_buttons: set[str] = set()
        self._mouse_position = (0, 0)
        self._last_frame_mouse_position: Optional[tuple[int, int]] = None
        self._scroll_dx = 0
        self._scroll_dy = 0
        self._frame_count = 0
        self._session_start = 0.0

        try:
            self._mouse_position = mouse.Controller().position
        except Exception:
            self._mouse_position = (0, 0)

    def start_capture(self) -> None:
        if self.running:
            self._play_feedback("already_running")
            print("[INFO] La captura ya esta activa.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_path = self.config.output_root / f"session_{timestamp}"
        self.frames_path = self.current_session_path / "frames"
        self.frames_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "fps": self.config.fps,
            "monitor_index": self.config.monitor_index,
            "jpeg_quality": self.config.jpeg_quality,
        }
        (self.current_session_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        csv_path = self.current_session_path / "labels.csv"
        self.csv_file = csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "frame_id",
                "timestamp",
                "relative_time_sec",
                "filename",
                "pressed_keys",
                "mouse_x",
                "mouse_y",
                "mouse_rel_x",
                "mouse_rel_y",
                "mouse_dx",
                "mouse_dy",
                "mouse_buttons",
                "scroll_dx",
                "scroll_dy",
            ]
        )

        self.running = True
        self._frame_count = 0
        self._session_start = time.time()
        self._last_frame_mouse_position = None

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self._play_feedback("start")
        print(f"[OK] Captura iniciada en: {self.current_session_path}")

    def stop_capture(self) -> None:
        if not self.running:
            self._play_feedback("already_stopped")
            print("[INFO] La captura no esta activa.")
            return

        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)

        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None

        self._play_feedback("stop")
        duration = time.time() - self._session_start if self._session_start else 0
        print(f"[OK] Captura detenida. Frames guardados: {self._frame_count}. Duracion: {duration:.1f}s")

    def on_key_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        key_name = self._normalize_key(key)
        if key_name:
            with self._keys_lock:
                self._pressed_keys.add(key_name)

    def on_key_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        key_name = self._normalize_key(key)
        if key_name:
            with self._keys_lock:
                self._pressed_keys.discard(key_name)

    def on_mouse_move(self, x: int, y: int) -> None:
        with self._mouse_lock:
            self._mouse_position = (x, y)

    def on_mouse_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        button_name = self._normalize_mouse_button(button)
        with self._mouse_lock:
            self._mouse_position = (x, y)
            if button_name:
                if pressed:
                    self._pressed_mouse_buttons.add(button_name)
                else:
                    self._pressed_mouse_buttons.discard(button_name)

    def on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        with self._mouse_lock:
            self._mouse_position = (x, y)
            self._scroll_dx += dx
            self._scroll_dy += dy

    def _capture_loop(self) -> None:
        interval = 1.0 / self.config.fps

        with mss.mss() as sct:
            monitors = sct.monitors
            if self.config.monitor_index >= len(monitors):
                print(
                    f"[WARN] monitor_index={self.config.monitor_index} no existe. "
                    "Se usa monitor principal (1)."
                )
                monitor = monitors[1]
            else:
                monitor = monitors[self.config.monitor_index]

            while self.running:
                loop_start = time.time()

                img = sct.grab(monitor)
                pil_img = Image.frombytes("RGB", img.size, img.rgb)

                with self._keys_lock:
                    keys_snapshot = sorted(
                        key_name for key_name in self._pressed_keys if self._should_store_key(key_name)
                    )
                with self._mouse_lock:
                    mouse_x, mouse_y = self._mouse_position
                    mouse_buttons_snapshot = sorted(self._pressed_mouse_buttons)
                    scroll_dx, scroll_dy = self._scroll_dx, self._scroll_dy
                    self._scroll_dx = 0
                    self._scroll_dy = 0

                if self._last_frame_mouse_position is None:
                    mouse_dx = 0
                    mouse_dy = 0
                else:
                    prev_x, prev_y = self._last_frame_mouse_position
                    mouse_dx = mouse_x - prev_x
                    mouse_dy = mouse_y - prev_y
                self._last_frame_mouse_position = (mouse_x, mouse_y)

                mouse_rel_x = mouse_x - monitor["left"]
                mouse_rel_y = mouse_y - monitor["top"]

                self._frame_count += 1
                frame_name = f"frame_{self._frame_count:06d}.jpg"
                frame_path = self.frames_path / frame_name
                pil_img.save(frame_path, format="JPEG", quality=self.config.jpeg_quality)

                timestamp = datetime.now().isoformat(timespec="milliseconds")
                relative_time = time.time() - self._session_start
                pressed_keys = "+".join(keys_snapshot)
                pressed_mouse_buttons = "+".join(mouse_buttons_snapshot)

                if self.csv_writer:
                    self.csv_writer.writerow(
                        [
                            self._frame_count,
                            timestamp,
                            f"{relative_time:.3f}",
                            frame_name,
                            pressed_keys,
                            mouse_x,
                            mouse_y,
                            mouse_rel_x,
                            mouse_rel_y,
                            mouse_dx,
                            mouse_dy,
                            pressed_mouse_buttons,
                            scroll_dx,
                            scroll_dy,
                        ]
                    )

                elapsed = time.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    @staticmethod
    def _normalize_key(key: keyboard.Key | keyboard.KeyCode) -> Optional[str]:
        if isinstance(key, keyboard.KeyCode):
            if key.char:
                char = key.char.lower()
                if not char.isprintable():
                    return None
                if ord(char) < 32:
                    return None
                return char
            return None

        special_map = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
            keyboard.Key.space: "space",
            keyboard.Key.shift: "shift",
            keyboard.Key.shift_l: "shift_l",
            keyboard.Key.shift_r: "shift_r",
            keyboard.Key.ctrl: "ctrl",
            keyboard.Key.ctrl_l: "ctrl_l",
            keyboard.Key.ctrl_r: "ctrl_r",
            keyboard.Key.alt: "alt",
            keyboard.Key.alt_l: "alt_l",
            keyboard.Key.alt_r: "alt_r",
            keyboard.Key.enter: "enter",
            keyboard.Key.esc: "esc",
            keyboard.Key.tab: "tab",
            keyboard.Key.backspace: "backspace",
        }

        return special_map.get(key)

    @staticmethod
    def _should_store_key(key_name: str) -> bool:
        blocked_keys = {
            "ctrl",
            "ctrl_l",
            "ctrl_r",
            "alt",
            "alt_l",
            "alt_r",
            "1",
            "2",
        }
        return key_name not in blocked_keys

    @staticmethod
    def _normalize_mouse_button(button: mouse.Button) -> Optional[str]:
        button_map = {
            mouse.Button.left: "left",
            mouse.Button.right: "right",
            mouse.Button.middle: "middle",
            mouse.Button.x1: "x1",
            mouse.Button.x2: "x2",
        }
        return button_map.get(button)

    @staticmethod
    def _play_feedback(event: str) -> None:
        sounds = {
            "start": [(880, 120), (1200, 140)],
            "stop": [(700, 120), (450, 160)],
            "already_running": [(500, 120)],
            "already_stopped": [(500, 120)],
        }
        for freq, dur in sounds.get(event, []):
            winsound.Beep(freq, dur)


def main() -> None:
    config = CaptureConfig()
    capturer = DrivingDataCapture(config)

    print("Herramienta de captura lista.")
    print("Ctrl+1: iniciar captura")
    print("Ctrl+2: detener captura")
    print("Ctrl+C en esta consola para salir")

    key_listener = keyboard.Listener(
        on_press=capturer.on_key_press,
        on_release=capturer.on_key_release,
    )
    key_listener.start()
    mouse_listener = mouse.Listener(
        on_move=capturer.on_mouse_move,
        on_click=capturer.on_mouse_click,
        on_scroll=capturer.on_mouse_scroll,
    )
    mouse_listener.start()

    hotkeys = keyboard.GlobalHotKeys(
        {
            "<ctrl>+1": capturer.start_capture,
            "<ctrl>+2": capturer.stop_capture,
        }
    )

    try:
        hotkeys.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Cerrando herramienta...")
    finally:
        capturer.stop_capture()
        hotkeys.stop()
        key_listener.stop()
        mouse_listener.stop()


if __name__ == "__main__":
    main()
