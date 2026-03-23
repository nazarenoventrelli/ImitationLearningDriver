# Imitation Learning Driver - Fase 1 (Captura de Datos)

Este proyecto empieza con una herramienta para capturar:
- Frames de pantalla del juego.
- Teclas presionadas en cada frame.
- Actividad del mouse (posicion, desplazamiento, botones y scroll) en cada frame.

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
python capture_tool.py
```

Atajos globales:
- `Ctrl + 1`: inicia captura
- `Ctrl + 2`: detiene captura

## Salida de datos

Se crea una carpeta por sesion en `captures/`:

- `captures/session_YYYYMMDD_HHMMSS/metadata.json`
- `captures/session_YYYYMMDD_HHMMSS/labels.csv`
- `captures/session_YYYYMMDD_HHMMSS/frames/frame_000001.jpg`

`labels.csv` contiene:
- `frame_id`
- `timestamp`
- `relative_time_sec`
- `filename`
- `pressed_keys` (ej: `w+a`, `space`, etc)
- `mouse_x`, `mouse_y`: posicion global del cursor.
- `mouse_rel_x`, `mouse_rel_y`: posicion relativa al monitor capturado.
- `mouse_dx`, `mouse_dy`: movimiento del mouse respecto al frame anterior.
- `mouse_buttons`: botones presionados (`left`, `right`, `middle`, etc).
- `scroll_dx`, `scroll_dy`: desplazamiento de rueda acumulado por frame.

## Nota

Si usas mas de un monitor y el juego esta en otro, cambia `monitor_index` en `CaptureConfig` dentro de `capture_tool.py`.

## Fase 2 - Entrenamiento

Entrena un modelo de imitation learning con:
- Clasificacion multi-etiqueta de teclado: `W`, `A`, `S`, `D`.
- Regresion de mouse: `dx`, `dy` normalizados.

Ejemplo de entrenamiento:

```bash
python train.py --captures-dir captures --epochs 10 --batch-size 64
```

Artefactos generados por corrida:
- `artifacts/run_YYYYMMDD_HHMMSS/best.pt`
- `artifacts/run_YYYYMMDD_HHMMSS/last.pt`
- `artifacts/run_YYYYMMDD_HHMMSS/metrics.json`
