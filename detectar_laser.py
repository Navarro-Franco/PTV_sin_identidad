#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detectar_laser.py
=================
Detección de la sección de líquido iluminada por un láser rojo y estimación
del campo de velocidades interno mediante flujo óptico denso.

A diferencia de `detectar_glitter.py`, acá no se buscan partículas puntuales:
la señal es una región roja difusa (el láser) con moteado adentro.  El pipeline
es:

1. **Máscara del láser** por exceso de rojo `R − max(G,B)`: umbral robusto
   dentro del ROI, morfología y eliminación de componentes chicos.  La máscara
   restringe todo el cálculo a la zona realmente iluminada.
2. **Flujo óptico denso** (Farneback) sobre la imagen de exceso de rojo con
   paso alto (se quita el halo difuso y queda el moteado).
3. **Muestreo en grilla**: cada celda de la grilla aporta una posición y una
   velocidad media.  Se escriben `detecciones.csv` y `vinculos.csv` con el
   mismo formato que consume `analizar_glitter.py`.

Salidas:
    --output        detecciones.csv
    --links         vinculos.csv  (xmid_mm, ymid_mm, vx_mm_s, vy_mm_s)
    --debug-video   video con la máscara y las flechas del flujo
    --summary-png   mapa promedio de velocidades + perfil vθ(r) descriptivo

Uso:
    python3 detectar_laser.py \
        --video ruta/al/video.mp4 \
        --roi-json resultados/<nombre>/recipiente.json \
        --output resultados/<nombre>_laser/detecciones.csv \
        --links  resultados/<nombre>_laser/vinculos.csv \
        --debug-video resultados/<nombre>_laser/deteccion_laser.mp4 \
        --summary-png resultados/<nombre>_laser/resumen_laser.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Imagen
# ---------------------------------------------------------------------------
def exceso_rojo(img_bgr: np.ndarray) -> np.ndarray:
    """Devuelve R − max(G,B) como float32.  El láser rojo queda muy alto y el
    líquido rosado (R≈G≈B) queda cerca de cero."""
    f = img_bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    return r - np.maximum(g, b)


def sigma_mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def umbral_laser(vals: np.ndarray, pct: float, k: float) -> float:
    """Umbral = el mayor entre un percentil y mediana + k·MAD.

    Con k=0 manda el percentil (recomendado: la región iluminada es difusa y
    un piso alto en MAD la recorta demasiado).  k>0 sirve para exigir un
    contraste mínimo sobre el fondo rosado.
    """
    med = float(np.median(vals))
    mad = max(sigma_mad(vals), 1e-6)
    return max(float(np.percentile(vals, pct)), med + k * mad)


def mascara_roi(shape, roi, escala: float, margen: float) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.mgrid[:h, :w]
    cx = roi["cx"] * escala
    cy = roi["cy"] * escala
    r = roi["a"] / 2.0 * escala * margen
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def mascara_laser(img, roi, escala, pct, k, min_area_px, margen,
                  min_comp_ratio=0.15):
    """Segmenta la región iluminada por el láser en un cuadro de trabajo."""
    e = exceso_rojo(img)
    roi_m = mascara_roi(img.shape, roi, escala, margen)
    vals = e[roi_m]
    if vals.size == 0:
        return np.zeros(e.shape, np.uint8), e, 0.0
    th = umbral_laser(vals, pct, k)
    m = ((e > th) & roi_m).astype(np.uint8)

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = [st[i, cv2.CC_STAT_AREA] for i in range(1, n)]
    amax = max(areas) if areas else 0
    umbral_area = max(min_area_px, min_comp_ratio * amax)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= umbral_area:
            keep[lab == i] = 1

    cont, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep)
    if cont:
        cv2.drawContours(filled, cont, -1, 1, thickness=cv2.FILLED)
    return filled, e, th


def preparar_flujo(img, roi, escala, sigma, margen):
    """Exceso de rojo + paso alto + normalización robusta a uint8.

    Se conserva la parte positiva del paso alto (el moteado brillante) y se
    reemplaza el exterior del ROI por un valor constante para que el fondo no
    aporte movimiento falso.
    """
    e = exceso_rojo(img)
    hp = e - cv2.GaussianBlur(e, (0, 0), sigma)
    hp = np.clip(hp, 0, None)
    roi_m = mascara_roi(img.shape, roi, escala, margen)
    vals = hp[roi_m]
    if vals.size == 0:
        return np.zeros(e.shape, np.uint8)
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((hp - lo) / (hi - lo), 0.0, 1.0) * 255.0
    out = cv2.GaussianBlur(out.astype(np.uint8), (5, 5), 0)
    out[~roi_m] = 127
    return out


# ---------------------------------------------------------------------------
# Grilla de muestreo
# ---------------------------------------------------------------------------
def construir_grilla(roi, ppm, grid_mm, margen):
    """Celdas (x_mm, y_mm) relativas al centro del ROI, dentro del círculo."""
    r_mm = roi["a"] / 2.0 / ppm * margen
    ejes = np.arange(-r_mm, r_mm + 1e-9, grid_mm)
    celdas = []
    for x in ejes:
        for y in ejes:
            cx = x + grid_mm / 2.0
            cy = y + grid_mm / 2.0
            if cx * cx + cy * cy <= r_mm * r_mm:
                celdas.append((cx, cy))
    return celdas


def muestrear_flujo(flow, mask, roi, escala, ppm, fps, stride,
                    celdas, grid_mm, min_frac=0.4):
    h, w = mask.shape[:2]
    cx = roi["cx"] * escala
    cy = roi["cy"] * escala
    half = max(1, int(round(grid_mm * ppm * escala / 2.0)))
    out = []
    for x_mm, y_mm in celdas:
        px = int(round(cx + x_mm * ppm * escala))
        py = int(round(cy + y_mm * ppm * escala))
        x0, x1 = max(0, px - half), min(w, px + half + 1)
        y0, y1 = max(0, py - half), min(h, py + half + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        m = mask[y0:y1, x0:x1] > 0
        frac = float(m.mean())
        if frac < min_frac:
            continue
        fx = flow[y0:y1, x0:x1, 0][m]
        fy = flow[y0:y1, x0:x1, 1][m]
        if fx.size < 3:
            continue
        vx = float(np.median(fx)) / (escala * ppm) * (fps / stride)
        vy = float(np.median(fy)) / (escala * ppm) * (fps / stride)
        out.append((x_mm, y_mm, vx, vy, frac))
    return out


def suavizar_campo(puntos, grid_mm, r_mm, sigma_mm):
    """Suaviza vx/vy entre celdas vecinas con una Gaussiana de ancho sigma_mm.

    El flujo del moteado es ruidoso; promediar espacialmente recupera el campo
    de gran escala sin imponer ninguna forma funcional.
    """
    if not puntos or sigma_mm <= 0:
        return puntos
    n = int(np.ceil(2.0 * r_mm / grid_mm)) + 1
    vx = np.zeros((n, n), np.float32)
    vy = np.zeros((n, n), np.float32)
    w = np.zeros((n, n), np.float32)
    idx = []
    for (x, y, u, v, _) in puntos:
        i = int(round((y + r_mm) / grid_mm))
        j = int(round((x + r_mm) / grid_mm))
        if not (0 <= i < n and 0 <= j < n):
            idx.append(None)
            continue
        vx[i, j] = u
        vy[i, j] = v
        w[i, j] = 1.0
        idx.append((i, j))
    sig = max(sigma_mm / grid_mm, 1e-6)
    k = int(round(2.0 * sig)) * 2 + 1
    wb = cv2.GaussianBlur(w, (k, k), sig)
    vxb = cv2.GaussianBlur(vx * w, (k, k), sig) / np.maximum(wb, 1e-6)
    vyb = cv2.GaussianBlur(vy * w, (k, k), sig) / np.maximum(wb, 1e-6)
    out = []
    for (x, y, u, v, f), ij in zip(puntos, idx):
        if ij is not None and wb[ij] > 0.2:
            out.append((x, y, float(vxb[ij]), float(vyb[ij]), f))
        else:
            out.append((x, y, u, v, f))
    return out


# ---------------------------------------------------------------------------
# Resumen visual (sin ajustes estadísticos)
# ---------------------------------------------------------------------------
def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    })
    return plt


def escribir_resumen(path, acum, muestras, roi, ppm, titulo,
                     escala, grid_mm):
    """`acum`: dict celda -> [sum_vx, sum_vy, n]. `muestras`: lista de
    (x_mm, y_mm, vx, vy).  Dos paneles: campo promedio y perfil vθ(r)."""
    plt = _estilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

    xs, ys, vx, vy = [], [], [], []
    for (x, y), (sx, sy, n) in acum.items():
        if n <= 0:
            continue
        xs.append(x)
        ys.append(y)
        vx.append(sx / n)
        vy.append(sy / n)
    if xs:
        xs = np.array(xs)
        ys = np.array(ys)
        vx = np.array(vx)
        vy = np.array(vy)
        sp = np.hypot(vx, vy)
        q = ax1.quiver(xs, ys, vx, vy, sp, angles="xy",
                       scale_units="xy", scale=None, cmap="viridis",
                       width=0.004)
        fig.colorbar(q, ax=ax1, label="|v| [mm/s]")
        ax1.set_xlim(-roi["a"] / 2 / ppm, roi["a"] / 2 / ppm)
        ax1.set_ylim(roi["a"] / 2 / ppm, -roi["a"] / 2 / ppm)
        ax1.set_aspect("equal")
    ax1.set_title("Campo de velocidades promedio (grilla %.1f mm)" % grid_mm)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")

    if muestras:
        arr = np.array(muestras, float)
        x, y, vx, vy = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        vt = -vx * np.sin(th) + vy * np.cos(th)
        vr = vx * np.cos(th) + vy * np.sin(th)
        signo = -1.0 if np.median(vt) < 0 else 1.0
        vt *= signo
        bins = np.arange(0.0, roi["a"] / 2 / ppm + 1e-9, 2.0)
        centros, med, lo, hi = [], [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            s = (r >= a) & (r < b)
            if s.sum() < 30:
                continue
            centros.append((a + b) / 2)
            med.append(np.median(vt[s]))
            lo.append(np.percentile(vt[s], 25))
            hi.append(np.percentile(vt[s], 75))
        if centros:
            ax2.fill_between(centros, lo, hi, color="C0", alpha=0.25,
                             label="IQR")
            ax2.plot(centros, med, "o-", color="C0", label="mediana vθ")
            ax2.axhline(0, color="k", lw=0.8)
            ax2.set_xlabel("r [mm]")
            ax2.set_ylabel("vθ [mm/s] (sentido de giro)")
            ax2.legend()
    ax2.set_title("Perfil tangencial descriptivo (sin ajustes)")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Detección del láser rojo + flujo óptico interno.")
    p.add_argument("--video", required=True)
    p.add_argument("--roi-json", required=True)
    p.add_argument("--output", required=True, help="CSV de detecciones")
    p.add_argument("--links", default=None, help="CSV de vínculos/velocidades")
    p.add_argument("--debug-video", default=None)
    p.add_argument("--summary-png", default=None)
    p.add_argument("--titulo", default=None)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--grid-mm", type=float, default=1.5,
                   help="tamaño de la celda de muestreo [mm]")
    p.add_argument("--flow-stride", type=int, default=4,
                   help="cuántos cuadros separa el flujo (aumenta la señal "
                        "cuando el movimiento es subpíxel)")
    p.add_argument("--suavizado-mm", type=float, default=3.0,
                   help="suavizado espacial del campo [mm]; 0 lo desactiva")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="guardar vínculos cada N cuadros")
    p.add_argument("--laser-pct", type=float, default=85.0,
                   help="percentil de exceso de rojo para la máscara")
    p.add_argument("--laser-k", type=float, default=0.0,
                   help="piso opcional mediana+k*MAD; 0 = usar sólo el percentil")
    p.add_argument("--min-area-mm2", type=float, default=0.5)
    p.add_argument("--min-comp-ratio", type=float, default=0.15,
                   help="descartar componentes menores a esta fracción del mayor")
    p.add_argument("--highpass-sigma", type=float, default=5.0)
    p.add_argument("--roi-margin", type=float, default=0.95)
    p.add_argument("--win-size", type=int, default=41)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--debug-scale", type=float, default=0.4)
    p.add_argument("--debug-paso", type=int, default=3,
                   help="dibujar una flecha cada N celdas")
    p.add_argument("--debug-cada", type=int, default=1,
                   help="escribir un cuadro de debug cada N")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--sin-debug", action="store_true")
    args = p.parse_args()

    with open(args.roi_json) as fh:
        rj = json.load(fh)
    a = float(rj.get("a", rj.get("diam_px", 0)))
    b = float(rj.get("b", rj.get("diam_px", 0)))
    roi = {"cx": float(rj["cx"]), "cy": float(rj["cy"]), "a": a, "b": b,
           "ang": float(rj.get("ang", 0.0)),
           "diam_px": float(rj.get("diam_px", (a + b) / 2))}
    ppm = float(rj["px_per_mm"])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.links is None:
        args.links = os.path.join(os.path.dirname(args.output), "vinculos.csv")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir {args.video}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    if args.end_frame is not None:
        total = min(total, args.end_frame + 1)
    ini = args.start_frame
    stride = max(1, args.flow_stride)

    celdas = construir_grilla(roi, ppm, args.grid_mm, args.roi_margin)
    min_area_px = max(4.0, args.min_area_mm2 * (ppm * args.scale) ** 2)
    print(f"[info] {W}x{H} @ {fps:.2f} fps, frames {ini}..{total-1}")
    print(f"[info] ROI {roi['diam_px']:.0f} px, {ppm:.3f} px/mm, "
          f"{len(celdas)} celdas de {args.grid_mm:g} mm")

    debug_w = int(W * args.debug_scale)
    debug_h = int(H * args.debug_scale)
    writer = None
    if args.debug_video and not args.sin_debug:
        os.makedirs(os.path.dirname(os.path.abspath(args.debug_video)),
                    exist_ok=True)
        for fourcc in ("avc1", "H264", "mp4v"):
            writer = cv2.VideoWriter(args.debug_video,
                                     cv2.VideoWriter_fourcc(*fourcc),
                                     fps, (debug_w, debug_h))
            if writer.isOpened():
                if fourcc != "avc1":
                    print(f"[aviso] video de debug en {fourcc}")
                break
        if not writer.isOpened():
            sys.exit("No se pudo crear el video de debug.")

    dets, links = [], []
    acum = {}
    muestras = []
    prev_prep = None
    prev_mask = None
    prev_f = None
    t0 = time.time()
    nframes = 0
    contador_debug = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, ini)
    f = ini
    while f < total:
        ok, frame = cap.read()
        if not ok:
            break
        work = cv2.resize(frame, None, fx=args.scale, fy=args.scale,
                          interpolation=cv2.INTER_AREA)
        mask, _, _ = mascara_laser(work, roi, args.scale, args.laser_pct,
                                   args.laser_k, min_area_px, args.roi_margin,
                                   args.min_comp_ratio)
        prep = preparar_flujo(work, roi, args.scale, args.highpass_sigma,
                              args.roi_margin)

        puntos = []
        if prev_prep is None:
            prev_prep, prev_mask, prev_f = prep, mask, f
        elif (f - prev_f) == stride:
            flow = cv2.calcOpticalFlowFarneback(
                prev_prep, prep, None, 0.5, args.levels, args.win_size, 3,
                5, 1.2, 0)
            m_inter = cv2.bitwise_and(prev_mask, mask)
            if m_inter.sum() > 0:
                m_inter = cv2.erode(m_inter, np.ones((3, 3), np.uint8))
            puntos = muestrear_flujo(flow, m_inter, roi, args.scale, ppm,
                                     fps, stride, celdas, args.grid_mm)
            puntos = suavizar_campo(puntos, args.grid_mm,
                                    roi["a"] / 2 / ppm * args.roi_margin,
                                    args.suavizado_mm)
            if (f % args.frame_stride) == 0:
                rex_mm_area = args.grid_mm * args.grid_mm * ppm * ppm
                for (x_mm, y_mm, vx, vy, frac) in puntos:
                    x_px = roi["cx"] + x_mm * ppm
                    y_px = roi["cy"] + y_mm * ppm
                    dets.append((f, f / fps, x_px, y_px, rex_mm_area, frac))
                    links.append((f, x_mm, y_mm, vx, vy))
                    k = (round(x_mm, 3), round(y_mm, 3))
                    s = acum.setdefault(k, [0.0, 0.0, 0])
                    s[0] += vx
                    s[1] += vy
                    s[2] += 1
                    muestras.append((x_mm, y_mm, vx, vy))
            prev_prep, prev_mask, prev_f = prep, mask, f

        if writer is not None and puntos and (f % args.debug_cada) == 0:
            vis = cv2.resize(frame, (debug_w, debug_h),
                             interpolation=cv2.INTER_AREA)
            cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cont = [c.astype(np.float32) * (1.0 / args.scale) * args.debug_scale
                    for c in cont]
            cv2.drawContours(vis, [c.astype(np.int32) for c in cont], -1,
                             (0, 255, 255), 2)
            if puntos:
                sp = np.array([np.hypot(x[2], x[3]) for x in puntos])
                ref = max(float(np.percentile(sp, 90)), 1e-6)
                hv = args.debug_scale
                for ip, (x_mm, y_mm, vx, vy, _) in enumerate(puntos):
                    if (ip % max(1, args.debug_paso)) != 0:
                        continue
                    x0 = (roi["cx"] + x_mm * ppm) * hv
                    y0 = (roi["cy"] + y_mm * ppm) * hv
                    v = float(np.hypot(vx, vy))
                    largo = min(45.0, v / ref * 45.0)
                    if v > 0:
                        x1 = x0 + vx / v * largo
                        y1 = y0 + vy / v * largo
                    else:
                        x1, y1 = x0, y0
                    cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                                    (0, 255, 0), 1, tipLength=0.35)
            roi_px = max(1.0, float(mascara_roi(mask.shape, roi, args.scale,
                                                args.roi_margin).sum()))
            cov = 100.0 * mask.sum() / roi_px
            vmed = float(np.median([np.hypot(x[2], x[3]) for x in puntos])) \
                if puntos else 0.0
            etq = (f"frame {f}  celdas={len(puntos)}  "
                   f"mascara={cov:.1f}%  v_med={vmed:.1f} mm/s")
            cv2.putText(vis, etq, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 0), 2)
            for _ in range(max(1, stride)):
                writer.write(vis)

        nframes += 1
        f += 1
        if nframes % 100 == 0:
            print(f"    frame {f}/{total}  ({time.time()-t0:.0f}s)")

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(dets, columns=["frame", "t_s", "x_px", "y_px",
                                     "area_px", "pico"])
    df.to_csv(args.output, index=False)
    dl = pd.DataFrame(links, columns=["frame", "xmid_mm", "ymid_mm",
                                      "vx_mm_s", "vy_mm_s"])
    dl.to_csv(args.links, index=False)
    print(f"[ok] {len(df)} detecciones -> {args.output}")
    print(f"[ok] {len(dl)} vínculos -> {args.links}")

    sigma_pos_px = 0.3
    sigma_v = float(np.sqrt(2) * sigma_pos_px / (args.scale * ppm)
                    * fps / stride)
    roi_out = {**roi, "px_per_mm": ppm, "fps": fps,
               "filtro": "laser", "modo": "laser_flow",
               "diametro_mm": float(rj.get("diametro_mm", 90.0)),
               "sigma_v_mm_s": sigma_v,
               "laser": {"pct": args.laser_pct, "k": args.laser_k,
                         "min_area_mm2": args.min_area_mm2,
                         "highpass_sigma": args.highpass_sigma,
                         "grid_mm": args.grid_mm,
                         "flow_stride": stride,
                         "scale": args.scale},
               "video": os.path.normpath(args.video)}
    roi_json = os.path.splitext(args.output)[0] + "_roi.json"
    with open(roi_json, "w") as fh:
        json.dump(roi_out, fh, indent=2)
    print(f"[ok] ROI/calibración -> {roi_json}")

    if args.summary_png:
        titulo = args.titulo or os.path.basename(args.video)
        escribir_resumen(args.summary_png, acum, muestras, roi, ppm, titulo,
                         args.scale, args.grid_mm)
        print(f"[ok] resumen -> {args.summary_png}")


if __name__ == "__main__":
    main()
