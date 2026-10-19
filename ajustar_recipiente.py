#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ajustar_recipiente.py
=====================
Herramienta interactiva para **calibrar px/mm sin video de calibración**: se
ajusta a mano un círculo sobre la zona de interés (por ejemplo, el recipiente)
y el programa deduce la escala.

    px/mm = diametro_px / diametro_mm

Flujo de trabajo
----------------
1. Abrí un cuadro del video (o el fondo temporal, que se ve más limpio):

       python3 ajustar_recipiente.py --video ruta/al/video.mp4 --bg

2. Ajustá el círculo hasta que coincida con el borde de la zona que sabés que
   mide un diámetro conocido (el recipiente / la superficie del fluido).

3. Guardá con **s** → escribe un JSON con el centro, el diámetro y el px/mm,
   más un PNG de verificación.

4. Usá ese JSON en la detección y el análisis:

       python3 detectar_glitter.py --video ... --roi-json resultados/<nombre>/recipiente.json ...
       python3 analizar_glitter.py --dir resultados/<nombre>

Controles
---------
- Arrastrar cerca del borde : cambiar el radio.
- Arrastrar adentro         : mover el centro.
- Rueda del mouse          : radio (paso grueso).
- Flechas ← → ↑ ↓           : mover el centro 1 px (con Shift, 10 px).
- + / -                     : radio ±1 px (con Shift, ±10 px).
- g                         : centrar el círculo en el cuadro.
- s                         : guardar JSON + PNG.
- q / Esc                   : salir.

Opciones
--------
--video PATH     video de entrada
--image PATH     imagen de entrada (alternativa)
--frame N        cuadro a mostrar (por defecto: el del medio)
--bg             usa la mediana temporal (fondo limpio) en vez de un cuadro
--n-bg N         cuántos cuadros muestrear para la mediana (por defecto 30)
--diametro-mm F  diámetro real de la zona (por defecto 90.0)
--scale F        factor de visualización (por defecto 0.4)
--salida PATH    JSON de salida (por defecto: recipiente.json junto al video)
--overlay PATH   PNG de verificación
--cerrar-al-guardar  cerrar la ventana automáticamente después de guardar con 's'
                     (útil para encadenar la calibración de varios videos)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def cargar_imagen(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"No se pudo leer {args.image}")
        return img, os.path.basename(args.image)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.bg:
        n = max(2, args.n_bg)
        idx = np.unique(np.linspace(0, total - 1, n).astype(int))
        stack = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                stack.append(f)
        cap.release()
        if not stack:
            sys.exit("No se pudieron leer cuadros.")
        print(f"[info] mediana temporal de {len(stack)} cuadros (fondo limpio)")
        return (np.median(np.stack(stack, 0), 0).astype(np.uint8),
                os.path.basename(args.video) + " (fondo)")
    fr = args.frame if args.frame is not None else total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
    ok, f = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"No se pudo leer el cuadro {fr}.")
    print(f"[info] cuadro {fr}")
    return f, f"{os.path.basename(args.video)} [frame {fr}]"


def main():
    p = argparse.ArgumentParser(description="Calibrar px/mm ajustando un círculo a mano.")
    p.add_argument("--video", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--bg", action="store_true")
    p.add_argument("--n-bg", type=int, default=30)
    p.add_argument("--diametro-mm", type=float, default=90.0)
    p.add_argument("--diametro-mm-err", type=float, default=0.5,
                   help="incertidumbre del diámetro real [mm]")
    p.add_argument("--scale", type=float, default=0.4)
    p.add_argument("--salida", default=None)
    p.add_argument("--overlay", default=None)
    p.add_argument("--cerrar-al-guardar", action="store_true",
                   help="cerrar la ventana al guardar con 's' (para encadenar videos)")
    args = p.parse_args()
    if not args.video and not args.image:
        p.error("Indicá --video o --image.")

    # En Wayland el plugin xcb de Qt falla; forzamos wayland.
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

    import importlib
    import matplotlib
    for nombre, modulo in [
        ("GTK3Agg", "matplotlib.backends.backend_gtk3agg"),
        ("GTK4Agg", "matplotlib.backends.backend_gtk4agg"),
        ("QtAgg", "matplotlib.backends.backend_qtagg"),
        ("TkAgg", "matplotlib.backends.backend_tkagg"),
    ]:
        try:
            importlib.import_module(modulo)
            matplotlib.use(nombre)
            print(f"[info] backend matplotlib = {nombre}")
            break
        except Exception:
            continue
    else:
        sys.exit("No hay backend gráfico. Instalá PyQt5: python3 -m pip install --user PyQt5")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    img, nombre = cargar_imagen(args)
    s = args.scale
    disp = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    Hd, Wd = disp.shape[:2]

    # estado en píxeles ORIGINALES
    estado = {"cx": W / 2.0, "cy": H / 2.0, "r": 0.28 * min(W, H)}
    drag = {"modo": None, "dx": 0.0, "dy": 0.0}

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    ax.set_xlim(0, Wd)
    ax.set_ylim(Hd, 0)
    circ = Circle((estado["cx"] * s, estado["cy"] * s), estado["r"] * s,
                  fill=False, edgecolor="red", lw=2)
    ax.add_patch(circ)
    cruz = ax.plot([estado["cx"] * s], [estado["cy"] * s], "+",
                   color="lime", ms=14, mew=2)[0]
    txt = ax.text(0.01, 0.02, "", transform=ax.transAxes, color="yellow",
                  fontsize=12, fontweight="bold",
                  bbox=dict(fc="black", alpha=0.6, ec="none"), va="bottom")

    def refrescar():
        circ.center = (estado["cx"] * s, estado["cy"] * s)
        circ.radius = estado["r"] * s
        cruz.set_data([estado["cx"] * s], [estado["cy"] * s])
        d_px = 2 * estado["r"]
        ppm = d_px / args.diametro_mm
        txt.set_text(f"centro=({estado['cx']:.0f},{estado['cy']:.0f}) px  "
                     f"D={d_px:.0f} px  ->  {ppm:.3f} px/mm")
        ax.set_title(f"{nombre}\n{args.diametro_mm:.0f} mm = {d_px:.0f} px "
                     f"({ppm:.3f} px/mm)   |   arrastrar=borde/centro, rueda=radio, "
                     "flechas=mover, +/-=radio, s=guardar, q=salir", fontsize=9)
        fig.canvas.draw_idle()

    def on_press(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        d = np.hypot(ev.xdata - estado["cx"] * s, ev.ydata - estado["cy"] * s)
        if abs(d - estado["r"] * s) < 12:
            drag["modo"] = "radio"
        else:
            drag["modo"] = "mover"
            drag["dx"] = ev.xdata - estado["cx"] * s
            drag["dy"] = ev.ydata - estado["cy"] * s

    def on_motion(ev):
        if drag["modo"] is None or ev.inaxes != ax or ev.xdata is None:
            return
        if drag["modo"] == "radio":
            estado["r"] = max(5.0, np.hypot(ev.xdata - estado["cx"] * s,
                                            ev.ydata - estado["cy"] * s) / s)
        else:
            estado["cx"] = (ev.xdata - drag["dx"]) / s
            estado["cy"] = (ev.ydata - drag["dy"]) / s
        refrescar()

    def on_release(ev):
        drag["modo"] = None

    def on_scroll(ev):
        if ev.inaxes != ax:
            return
        estado["r"] = max(5.0, estado["r"] * (1.03 if ev.step > 0 else 1 / 1.03))
        refrescar()

    def on_key(ev):
        k = ev.key
        paso = 10.0 if "shift" in (k or "") else 1.0
        if k in ("left", "right", "up", "down"):
            if k == "left":
                estado["cx"] -= paso
            elif k == "right":
                estado["cx"] += paso
            elif k == "up":
                estado["cy"] -= paso
            elif k == "down":
                estado["cy"] += paso
        elif k in ("+", "="):
            estado["r"] += paso
        elif k in ("-", "_"):
            estado["r"] = max(5.0, estado["r"] - paso)
        elif k == "g":
            estado["cx"], estado["cy"] = W / 2.0, H / 2.0
        elif k == "s":
            guardar()
            return
        elif k in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        refrescar()

    def guardar():
        d_px = 2 * estado["r"]
        ppm = d_px / args.diametro_mm
        salida = args.salida
        if salida is None:
            base = args.video or args.image
            salida = os.path.join(os.path.dirname(os.path.abspath(base)),
                                  "recipiente.json")
        data = {"cx": float(estado["cx"]), "cy": float(estado["cy"]),
                "a": float(d_px), "b": float(d_px), "ang": 0.0,
                "diam_px": float(d_px), "px_per_mm": float(ppm),
                "diametro_mm": float(args.diametro_mm),
                "diametro_mm_err": float(args.diametro_mm_err),
                "fuente": "manual",
                "video": os.path.normpath(args.video) if args.video else None}
        os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
        with open(salida, "w") as fh:
            json.dump(data, fh, indent=2)
        overlay = args.overlay or os.path.splitext(salida)[0] + ".png"
        refrescar()
        fig.savefig(overlay, dpi=130, bbox_inches="tight")
        print(f"[ok] {salida}")
        print(f"     centro=({estado['cx']:.1f},{estado['cy']:.1f}) px  "
              f"D={d_px:.1f} px  ->  {ppm:.4f} px/mm")
        print(f"[ok] {overlay}")
        if args.cerrar_al_guardar:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refrescar()
    print("  Ajustá el círculo al borde de la zona de 9 cm y apretá 's'.")
    plt.show()


if __name__ == "__main__":
    main()
