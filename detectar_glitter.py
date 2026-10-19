#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detectar_glitter.py
===================
Detección y vinculación *frame a frame* de MUCHAS partículas de glitter, a
partir de un video.

Idea general
------------
Hay cientos de partículas chiquitas e indistinguibles entre sí que se mueven
rápido.  Por eso:

* **No** se fija un número N de partículas: se detectan *todas* las que pasan
  los filtros en cada cuadro.
* **No** se sigue a cada una todo el video (frame a frame otra partícula puede
  ocupar su lugar).  Alcanza con vincular cada detección con la más parecida del
  cuadro siguiente (vecino mutuo) para estimar velocidades.  Como el campo del
  vórtice es estacionario y suave, aunque el vínculo cambie de partícula, la
  velocidad medida sigue siendo representativa del punto.

Filtro de imagen
----------------
    amarillo = (G+R)/2 − B        menos      fondo temporal (mediana)

porque el glitter es amarillo y el fondo (mesa, vidrio, reflejos) es neutro o
está quieto.  El "fondo temporal" es la mediana por píxel: el glitter se mueve
y desaparece de la mediana; lo estático queda.  Para trazadores de otro color
usar `--filtro` (o `--filtro auto`).

Calibración (px/mm) y ROI
-------------------------
Se toman del JSON que genera `ajustar_recipiente.py` (un círculo ajustado a
mano sobre la zona de diámetro conocido) con `--roi-json`.  Si no se pasa, se
intenta detectar automáticamente el borde del fluido y se calibra con
`--diametro-mm`.

Salidas:
    --output        detecciones.csv : frame,t_s,x_px,y_px,area_px,pico[,x_mm,y_mm]
    --links         vinculos.csv    : frame,x0_px,...,vx_mm_s,vy_mm_s,...
    (auto)          <output>_roi.json con el centro/ROI/calibración usados
    --debug-video   video con las detecciones (círculos verdes) y las flechas
                    verdes de los vínculos cuadro anterior -> cuadro actual

Uso:
    python3 detectar_glitter.py \
        --video ruta/al/video.mp4 \
        --roi-json resultados/<nombre>/recipiente.json \
        --output resultados/<nombre>/detecciones.csv \
        --links  resultados/<nombre>/vinculos.csv \
        --bg-cache resultados/<nombre>/bg.png \
        --debug-video resultados/<nombre>/deteccion.mp4
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
# Filtros de imagen
# ---------------------------------------------------------------------------
FILTROS = ["gray", "green", "red", "blue", "yellow", "magenta", "greenx2",
           "white", "maxc", "sat"]


def filtro(img_bgr: np.ndarray, nombre: str) -> np.ndarray:
    """Aplica un filtro de la lista FILTROS y devuelve float32 (H, W)."""
    f = img_bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    nombre = nombre.lower()
    if nombre in ("gray", "gris"):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if nombre == "green":
        return g
    if nombre == "red":
        return r
    if nombre == "blue":
        return b
    if nombre == "yellow":
        return 0.5 * (g + r) - b
    if nombre in ("magenta", "pink"):
        return 0.5 * (r + b) - g
    if nombre == "greenx2":
        return 2.0 * g - r - b
    if nombre == "white":
        return np.minimum(np.minimum(b, g), r)
    if nombre == "maxc":
        return np.maximum(np.maximum(b, g), r)
    if nombre == "sat":
        return (np.maximum(np.maximum(b, g), r)
                - np.minimum(np.minimum(b, g), r))
    raise ValueError(f"Filtro desconocido: {nombre}")


# ---------------------------------------------------------------------------
# Fondo temporal
# ---------------------------------------------------------------------------
def construir_fondo(video: str, n: int = 40, dst_scale: float = 0.25,
                    max_frames: int | None = None, cache: str | None = None):
    """Fondo = mediana por píxel (y por canal) de n cuadros repartidos en todo
    el video, a escala reducida.  Como el glitter se mueve, en cada píxel la
    mediana es el fondo estático (mesa, vidrio, reflejos fijos).

    Devuelve una imagen BGR uint8 a escala `dst_scale`.
    """
    if cache and os.path.exists(cache):
        return cv2.imread(cache, cv2.IMREAD_COLOR)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    idx = np.unique(np.linspace(0, total - 1, max(3, n)).astype(int))
    stack = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        stack.append(cv2.resize(fr, None, fx=dst_scale, fy=dst_scale,
                                interpolation=cv2.INTER_AREA))
    cap.release()
    if not stack:
        raise RuntimeError("No se pudieron leer cuadros para el fondo.")
    med = np.median(np.stack(stack, 0), axis=0).astype(np.uint8)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        cv2.imwrite(cache, med)
    return med


# ---------------------------------------------------------------------------
# Umbral robusto y detección de blobs
# ---------------------------------------------------------------------------
def sigma_mad(x: np.ndarray) -> float:
    """Desvío estándar robusto (MAD) de un arreglo."""
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def umbral_robusto(diff: np.ndarray, k: float = 4.0, minimo: float = 3.0) -> float:
    """Umbral = k * sigma_robusto del ruido de la resta de fondo."""
    return max(minimo, k * sigma_mad(diff))


def detectar(diff: np.ndarray, thresh: float, min_area: float, max_area: float,
             escala: float = 1.0):
    """Umbraliza `diff` (float32, alto=glitter) y devuelve una lista de
    detecciones (x_px, y_px, area_px, pico) en coordenadas del frame original.

    `escala` es el factor con el que se redujo el cuadro (para volver a px
    reales).
    """
    m = (diff > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, lab, st, ce = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        a = float(st[k, cv2.CC_STAT_AREA])
        if a < min_area or a > max_area:
            continue
        x0 = int(st[k, cv2.CC_STAT_LEFT])
        y0 = int(st[k, cv2.CC_STAT_TOP])
        w = int(st[k, cv2.CC_STAT_WIDTH])
        h = int(st[k, cv2.CC_STAT_HEIGHT])
        # centroide subpíxel pesado por intensidad (mejor que el de la máscara)
        sub = diff[y0:y0 + h, x0:x0 + w] * (lab[y0:y0 + h, x0:x0 + w] == k)
        tot = sub.sum()
        if tot <= 0:
            cx, cy = ce[k]
            pico = float(diff[int(round(cy)), int(round(cx))])
        else:
            ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w]
            cx = float((sub * xs).sum() / tot)
            cy = float((sub * ys).sum() / tot)
            pico = float(sub.max())
        out.append((cx / escala, cy / escala, a / (escala * escala), pico))
    return out


# ---------------------------------------------------------------------------
# ROI automático (respaldo si no se pasa --roi-json)
# ---------------------------------------------------------------------------
def estimar_roi_fluido(bg_bgr: np.ndarray, escala: float = 1.0):
    """Ajusta una elipse al borde fluido↔vidrio del fondo temporal.

    Desde el centro se trazan rayos y en cada uno se busca la ÚLTIMA transición
    oscuro→claro (donde el fluido termina y empieza la pared de vidrio).  Luego
    se ajusta una elipse.  Devuelve centro, ejes (longitudes completas), ángulo
    y diámetro equivalente, en píxeles del frame original.
    """
    g = cv2.GaussianBlur(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32),
                         (5, 5), 0)
    H, W = g.shape
    cx, cy = W / 2.0, H / 2.0
    a = b = 0.0
    ang = 0.0
    for _ in range(3):
        pts = []
        rr = np.arange(int(0.25 * W), int(0.52 * W))
        for th in np.linspace(0, 2 * np.pi, 720, endpoint=False):
            xs = cx + rr * np.cos(th)
            ys = cy + rr * np.sin(th)
            keep = (xs >= 1) & (xs < W - 1) & (ys >= 1) & (ys < H - 1)
            if keep.sum() < 10:
                continue
            xi = xs[keep].astype(np.float32)
            yi = ys[keep].astype(np.float32)
            vals = cv2.remap(g, xi.reshape(1, -1), yi.reshape(1, -1),
                             cv2.INTER_LINEAR).ravel()
            v = cv2.GaussianBlur(vals.reshape(-1, 1), (1, 15), 0).ravel()
            d = np.gradient(v)
            kk = np.where(d > 2.0)[0]
            if len(kk):
                k = kk[-1]
                pts.append((float(xi[k]), float(yi[k])))
        pts = np.array(pts)
        if len(pts) < 30:
            break
        (ex, ey), (a, b), ang = cv2.fitEllipse(pts.astype(np.float32).reshape(-1, 1, 2))
        cx, cy = ex, ey
    if a == 0 or b == 0:
        raise RuntimeError("No se pudo detectar el borde del fluido.")
    ex, ey, a, b = ex / escala, ey / escala, a / escala, b / escala
    return {"cx": float(ex), "cy": float(ey), "a": float(a), "b": float(b),
            "ang": float(ang), "diam_px": float(np.sqrt(a * b))}


def dentro_elipse(pts, roi, margen=0.98):
    if roi is None:
        return np.ones(len(pts), bool)
    cx, cy = roi["cx"], roi["cy"]
    a, b = roi["a"] / 2 * margen, roi["b"] / 2 * margen
    ang = np.radians(roi["ang"])
    ca, sa = np.cos(ang), np.sin(ang)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


# ---------------------------------------------------------------------------
# Selección automática de filtro de color
# ---------------------------------------------------------------------------
FILTROS_AUTO = ["yellow", "magenta", "green", "red", "greenx2", "sat",
                "maxc", "white"]


def elegir_filtro_auto(video, bg, roi, escala, k, min_area, max_area,
                       n=8, start_frame=0, max_frames=None, candidatos=None):
    """Prueba varios filtros sobre unos cuadros repartidos en el video y elige
    el que da más detecciones (mediana), restringido al ROI.

    Devuelve (mejor_filtro, {filtro: score}).
    """
    candidatos = candidatos or FILTROS_AUTO
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return "yellow", {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    idx = np.unique(np.linspace(start_frame, max(1, total - 1), max(3, n)).astype(int))
    con = {f: [] for f in candidatos}
    bgf_cache = {}
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        work = cv2.resize(frame, None, fx=escala, fy=escala,
                          interpolation=cv2.INTER_AREA)
        for f in candidatos:
            if f not in bgf_cache:
                bgf_cache[f] = cv2.resize(filtro(bg, f),
                                          (work.shape[1], work.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
            diff = filtro(work, f) - bgf_cache[f]
            d = detectar(diff, umbral_robusto(diff, k), min_area, max_area,
                         escala=escala)
            if d:
                arr = np.array(d, float)
                if roi is not None:
                    arr = arr[dentro_elipse(arr[:, :2], roi)]
                con[f].append(len(arr))
            else:
                con[f].append(0)
    cap.release()
    score = {f: float(np.median(v)) if v else 0.0 for f, v in con.items()}
    mejor = max(score, key=score.get)
    return mejor, score


# ---------------------------------------------------------------------------
# Vínculos
# ---------------------------------------------------------------------------
def vincular(p0, p1, gate_px):
    """Vecino mutuo entre las detecciones de dos cuadros consecutivos.

    p0, p1: (N,2) posiciones.  Devuelve lista de (i, j) mutuamente más cercanos
    dentro de `gate_px`.
    """
    if len(p0) == 0 or len(p1) == 0:
        return []
    d = np.linalg.norm(p0[:, None, :] - p1[None, :, :], axis=2)
    j_of_i = d.argmin(1)
    i_of_j = d.argmin(0)
    pares = []
    for i, j in enumerate(j_of_i):
        if i_of_j[j] == i and d[i, j] <= gate_px:
            pares.append((i, j))
    return pares


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Detección + vínculos de glitter.")
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True, help="CSV de detecciones")
    p.add_argument("--links", default=None, help="CSV de vínculos (velocidades)")
    p.add_argument("--debug-video", default=None,
                   help="video con las detecciones y las flechas verdes de los vínculos")
    p.add_argument("--debug-scale", type=float, default=0.4)
    p.add_argument("--sin-flechas", dest="dibujar_flechas", action="store_false",
                   help="no dibujar las flechas verdes de los vínculos "
                        "(cuadro anterior -> actual) en el video de debug")
    p.add_argument("--flecha-ganancia", type=float, default=5.0,
                   help="factor de exageración del largo de las flechas en el video "
                        "de debug (1 = desplazamiento real; por defecto 5, como en PIV)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="escala de procesamiento respecto del video original")
    p.add_argument("--filtro", default="yellow",
                   help="filtro de imagen (yellow, green, gray, white, ...)")
    p.add_argument("--k", type=float, default=6.0,
                   help="umbral = k × sigma_robusto del ruido de la resta")
    p.add_argument("--min-area", type=float, default=3.0,
                   help="área mínima del blob (px² a escala de proceso)")
    p.add_argument("--max-area", type=float, default=300.0,
                   help="área máxima del blob (px² a escala de proceso)")
    p.add_argument("--bg-frames", type=int, default=40)
    p.add_argument("--bg-cache", default=None)
    p.add_argument("--roi-json", default=None,
                   help="JSON con el círculo ajustado a mano (ajustar_recipiente.py). "
                        "Si se da, se usa como ROI y como calibración px/mm.")
    p.add_argument("--px-per-mm", type=float, default=None,
                   help="calibración explícita (tiene prioridad)")
    p.add_argument("--diametro-mm", type=float, default=90.0,
                   help="diámetro real del recipiente (ROI automático)")
    p.add_argument("--sin-roi", action="store_true",
                   help="no restringir las detecciones a un ROI")
    p.add_argument("--gate-mm", type=float, default=2.5,
                   help="distancia máxima de vínculo entre cuadros (mm). Debe ser "
                        "apenas mayor que el desplazamiento real: ~1.5×v_max/fps. "
                        "Con 5 mm se cuelan vínculos falsos (sobre todo en el centro); "
                        "2.5 mm los elimina sin perder partículas reales")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir {args.video}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    if args.end_frame is not None:
        total = min(total, args.end_frame + 1)
    print(f"[info] {W}x{H} @ {fps:.2f} fps, frames {args.start_frame}..{total-1}")

    # ---- fondo ----
    print("[info] fondo temporal ...")
    bg = construir_fondo(args.video, n=args.bg_frames, dst_scale=args.scale,
                         max_frames=args.max_frames, cache=args.bg_cache)

    # ---- ROI / calibración ----
    roi = None
    ppm_json = None
    diametro_mm_json = None
    diametro_mm_err_json = None
    if args.roi_json:
        with open(args.roi_json) as fh:
            rj = json.load(fh)
        a = float(rj.get("a", rj.get("diam_px", 0)))
        b = float(rj.get("b", rj.get("diam_px", 0)))
        roi = {"cx": float(rj["cx"]), "cy": float(rj["cy"]), "a": a, "b": b,
               "ang": float(rj.get("ang", 0.0)),
               "diam_px": float(rj.get("diam_px", (a + b) / 2))}
        ppm_json = rj.get("px_per_mm")
        diametro_mm_json = rj.get("diametro_mm")
        diametro_mm_err_json = rj.get("diametro_mm_err")
        print(f"[info] ROI/calibración desde {args.roi_json}: "
              f"centro=({roi['cx']:.0f},{roi['cy']:.0f}) D={roi['diam_px']:.0f} px")
    elif not args.sin_roi:
        try:
            roi = estimar_roi_fluido(bg, escala=args.scale)
            print(f"[info] ROI fluido: centro=({roi['cx']:.0f},{roi['cy']:.0f}) "
                  f"ejes=({roi['a']:.0f},{roi['b']:.0f}) D={roi['diam_px']:.0f} px")
        except Exception as e:
            print(f"[aviso] no se pudo estimar ROI ({e}); se usa todo el cuadro")

    if args.px_per_mm:
        ppm = args.px_per_mm
        print(f"[info] px/mm (dado) = {ppm:.3f}")
    elif ppm_json:
        ppm = float(ppm_json)
        print(f"[info] px/mm (json) = {ppm:.3f}")
    elif roi is not None:
        ppm = roi["diam_px"] / args.diametro_mm
        print(f"[info] px/mm = {ppm:.3f} (D_fluido/{args.diametro_mm} mm)")
    else:
        ppm = None
        print("[aviso] sin calibración; x_mm/y_mm no se generarán")
    gate_px = (args.gate_mm * ppm) if ppm else 80.0

    # ---- filtro de color (auto o explícito) ----
    filtro_nombre = args.filtro
    if args.filtro == "auto":
        print("[info] eligiendo filtro de color automáticamente ...")
        filtro_nombre, score = elegir_filtro_auto(
            args.video, bg, roi, args.scale, args.k, args.min_area,
            args.max_area, n=8, start_frame=args.start_frame,
            max_frames=args.max_frames)
        print("[info] detecciones/cuadro por filtro: " +
              ", ".join(f"{f}={s:.0f}" for f, s in
                        sorted(score.items(), key=lambda x: -x[1])))
    print(f"[info] filtro = {filtro_nombre}")

    # ---- video de debug ----
    debug_w = int(W * args.debug_scale)
    debug_h = int(H * args.debug_scale)
    writer = None
    if args.debug_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.debug_video)), exist_ok=True)
        for fourcc in ("avc1", "H264", "mp4v"):
            writer = cv2.VideoWriter(args.debug_video,
                                     cv2.VideoWriter_fourcc(*fourcc),
                                     fps, (debug_w, debug_h))
            if writer.isOpened():
                if fourcc != "avc1":
                    print(f"[aviso] video de debug en {fourcc} "
                          "(el navegador puede no reproducirlo)")
                break
        if not writer.isOpened():
            sys.exit("No se pudo crear el video de debug.")

    # ---- procesamiento frame a frame ----
    dets = []
    por_frame = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    fr = args.start_frame
    t0 = time.time()
    while fr < total:
        ok, frame = cap.read()
        if not ok:
            break
        work = cv2.resize(frame, None, fx=args.scale, fy=args.scale,
                          interpolation=cv2.INTER_AREA)
        fmap = filtro(work, filtro_nombre)
        bgf = cv2.resize(filtro(bg, filtro_nombre), (work.shape[1], work.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
        diff = fmap - bgf
        th = umbral_robusto(diff, args.k)
        d = detectar(diff, th, args.min_area, args.max_area, escala=args.scale)
        if d:
            arr = np.array(d, float)
            if roi is not None:
                arr = arr[dentro_elipse(arr[:, :2], roi)]
            for (x, y, a, pk) in arr:
                dets.append((fr, fr / fps, x, y, a, pk))
            pts = arr[:, :2]
        else:
            arr = np.zeros((0, 4))
            pts = np.zeros((0, 2))
        por_frame[fr] = pts

        if writer is not None:
            vis = cv2.resize(frame, (debug_w, debug_h), interpolation=cv2.INTER_AREA)
            if roi is not None:
                cv2.ellipse(vis,
                            (int(roi["cx"] * args.debug_scale), int(roi["cy"] * args.debug_scale)),
                            (int(roi["a"] / 2 * args.debug_scale), int(roi["b"] / 2 * args.debug_scale)),
                            roi["ang"], 0, 360, (0, 165, 255), 2)
            # flechas verdes: vínculo entre el cuadro anterior y este
            nflechas = 0
            g = args.flecha_ganancia
            if args.dibujar_flechas and (fr - 1) in por_frame:
                prev = por_frame[fr - 1]
                if len(prev) and len(pts):
                    for i, j in vincular(prev, pts, gate_px):
                        x0 = prev[i, 0] * args.debug_scale
                        y0 = prev[i, 1] * args.debug_scale
                        x1 = x0 + (pts[j, 0] * args.debug_scale - x0) * g
                        y1 = y0 + (pts[j, 1] * args.debug_scale - y0) * g
                        cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                                        (0, 255, 0), 2, tipLength=0.35)
                        nflechas += 1
            for (x, y, a, pk) in arr:
                cv2.circle(vis, (int(x * args.debug_scale), int(y * args.debug_scale)),
                           3, (0, 255, 0), -1)
            etq = f"frame {fr}  n={len(arr)}  flechas={nflechas}"
            if args.dibujar_flechas and g != 1.0:
                etq += f"  (x{g:g})"
            cv2.putText(vis, etq, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            writer.write(vis)
        fr += 1
        if fr % 100 == 0:
            print(f"    frame {fr}/{total}  ({time.time()-t0:.0f}s)")
    cap.release()
    if writer is not None:
        writer.release()

    # ---- guardar detecciones ----
    df = pd.DataFrame(dets, columns=["frame", "t_s", "x_px", "y_px",
                                     "area_px", "pico"])
    if ppm:
        df["x_mm"] = df["x_px"] / ppm
        df["y_mm"] = df["y_px"] / ppm
    df.to_csv(args.output, index=False)
    print(f"[ok] {len(df)} detecciones -> {args.output}")
    print(f"     partículas/cuadro: mediana={df.groupby('frame').size().median():.0f}, "
          f"máx={df.groupby('frame').size().max()}")

    # ---- guardar ROI/calibración para el análisis ----
    roi_json = os.path.splitext(args.output)[0] + "_roi.json"
    if roi is not None:
        diametro_salida = (float(diametro_mm_json)
                           if diametro_mm_json is not None
                           else float(args.diametro_mm))
        datos_roi = {**roi, "px_per_mm": ppm, "fps": fps,
                     "filtro": filtro_nombre,
                     "diametro_mm": diametro_salida,
                     "video": os.path.normpath(args.video)}
        if diametro_mm_err_json is not None:
            datos_roi["diametro_mm_err"] = float(diametro_mm_err_json)
        with open(roi_json, "w") as fh:
            json.dump(datos_roi, fh, indent=2)
        print(f"[ok] ROI/calibración -> {roi_json}")

    # ---- vínculos / velocidades ----
    if args.links:
        filas = []
        frames_ord = sorted(por_frame)
        for f0, f1 in zip(frames_ord[:-1], frames_ord[1:]):
            if f1 != f0 + 1:
                continue
            p0, p1 = por_frame[f0], por_frame[f1]
            for i, j in vincular(p0, p1, gate_px):
                dx = p1[j, 0] - p0[i, 0]
                dy = p1[j, 1] - p0[i, 1]
                dt = 1.0 / fps
                filas.append((f0, p0[i, 0], p0[i, 1], p1[j, 0], p1[j, 1],
                              dx, dy, dx / dt, dy / dt))
        dl = pd.DataFrame(filas, columns=["frame", "x0_px", "y0_px", "x1_px",
                                          "y1_px", "dx_px", "dy_px",
                                          "vx_px_s", "vy_px_s"])
        if ppm:
            cen = (roi["cx"], roi["cy"]) if roi else (W / 2, H / 2)
            xm = (dl["x0_px"] + dl["x1_px"]) / 2 / ppm
            ym = (dl["y0_px"] + dl["y1_px"]) / 2 / ppm
            dl["xmid_mm"] = xm - cen[0] / ppm
            dl["ymid_mm"] = ym - cen[1] / ppm
            dl["r_mm"] = np.hypot(dl["xmid_mm"], dl["ymid_mm"])
            dl["dx_mm"] = dl["dx_px"] / ppm
            dl["dy_mm"] = dl["dy_px"] / ppm
            dl["vx_mm_s"] = dl["vx_px_s"] / ppm
            dl["vy_mm_s"] = dl["vy_px_s"] / ppm
        dl.to_csv(args.links, index=False)
        print(f"[ok] {len(dl)} vínculos -> {args.links}")


if __name__ == "__main__":
    main()
