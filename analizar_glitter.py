#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analizar_glitter.py
====================
Análisis del campo de velocidades a partir de las detecciones y vínculos que
produce `detectar_glitter.py`.

Como hay muchísimas partículas y los vínculos pueden "cambiar de partícula"
frame a frame, NO se usan trayectorias largas: se trabaja con los **vínculos**
(pares de detecciones en cuadros consecutivos), que muestrean el campo de
velocidades (estacionario y axisimétrico).

Pasos
-----
1. **Centro del vórtice**: se busca el centro que minimiza la dispersión de la
   velocidad tangencial dentro de cada anillo radial (coherencia azimutal).
2. **Polares y velocidades**: para cada vínculo se calcula r, θ y v_r, v_θ por
   proyección del vector velocidad sobre los versores radial y tangencial.
3. **Perfil**: mediana de v_θ y v_r por bins radiales (robusta, con MAD).
4. **Modelos** (parametrización del informe `docs/analisis_formulas.pdf`):

   - **Rankine** (ec. 1):
         v_θ = Ω r            (r ≤ R)
         v_θ = Ω R² / r       (r > R)          →  Ω [1/s], R [mm]
   - **Burgers** (ec. 2):
         v_r = -(2ν/R²) r
         v_θ = (Ω R² / r) (1 - e^{-r²/R²})      →  Ω, R, ν [mm²/s]
         ν se estima de la pendiente radial m (v_r = m r):  ν = -m R²/2  (ec. 4)

5. **Errores**: posición (repetición de la detección), velocidad (propagación),
   parámetros (bootstrap sobre el perfil) y centro (sensibilidad).

Salida: `resultados.json`, `perfil.csv`, figuras PNG e `informe.html`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from string import Template

import cv2
import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Claves de los diccionarios de ajuste que no son parámetros escalares.
EXCLUIR_AJUSTE = (
    "modelo", "r2", "curva", "limite_inferior", "limite_superior",
    "chi2", "chi2_red", "dof", "p_value", "n", "n_params",
    "r2_libre", "r2_cero",
    "chi2_libre", "chi2red_libre", "dof_libre", "p_value_libre",
    "n_libre", "n_params_libre",
    "chi2_cero", "chi2red_cero", "dof_cero", "p_value_cero",
    "n_cero", "n_params_cero",
)


# ---------------------------------------------------------------------------
# Estilo de gráficos (el pedido: fuente serif, mathtext cm, etc.)
# ---------------------------------------------------------------------------
def estilo_graficos():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05
    return plt


# ---------------------------------------------------------------------------
# Geometría / velocidades
# ---------------------------------------------------------------------------
def a_polares(x, y, cx, cy):
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    return r, th


def componentes(x, y, vx, vy, cx, cy):
    """Devuelve r, theta, v_r, v_theta para vectores de posición/velocidad."""
    r, th = a_polares(x, y, cx, cy)
    c, s = np.cos(th), np.sin(th)
    vr = vx * c + vy * s
    vt = -vx * s + vy * c
    return r, th, vr, vt


def dispersion_vt(x, y, vx, vy, cx, cy, bins):
    """Dispersión robusta (MAD) de v_theta dentro de cada bin radial.
    Se usa para elegir el centro del vórtice."""
    r, th, vr, vt = componentes(x, y, vx, vy, cx, cy)
    idx = np.digitize(r, bins) - 1
    total, peso = 0.0, 0.0
    for b in range(len(bins) - 1):
        m = idx == b
        n = m.sum()
        if n < 8:
            continue
        v = vt[m]
        mad = 1.4826 * np.median(np.abs(v - np.median(v)))
        total += n * mad
        peso += n
    return total / max(peso, 1.0)


def optimizar_centro(x, y, vx, vy, c0, rango_px, paso_px, bins):
    """Barre una grilla de centros candidatos y se queda con el que minimiza la
    dispersión (MAD) de v_theta dentro de cada anillo.

    Devuelve el mejor centro (mm), su dispersión, y la grilla completa
    (S, xs, ys) para poder graficar la búsqueda.
    """
    n = int(rango_px / paso_px)
    xs = c0[0] + paso_px * np.arange(-n, n + 1)
    ys = c0[1] + paso_px * np.arange(-n, n + 1)
    S = np.full((len(ys), len(xs)), np.nan)
    best = ((float(c0[0]), float(c0[1])), np.inf)
    for j, cy in enumerate(ys):
        for i, cx in enumerate(xs):
            s = dispersion_vt(x, y, vx, vy, float(cx), float(cy), bins)
            S[j, i] = s
            if s < best[1]:
                best = ((float(cx), float(cy)), float(s))
    return best[0], best[1], S, xs, ys


# ---------------------------------------------------------------------------
# Perfil radial y recorte de outliers
# ---------------------------------------------------------------------------
def perfil_radial(r, vr, vt, bins):
    filas = []
    for b in range(len(bins) - 1):
        m = (r >= bins[b]) & (r < bins[b + 1])
        n = int(m.sum())
        if n < 5:
            continue
        vv = vt[m]
        med = float(np.median(vv))
        mad = 1.4826 * float(np.median(np.abs(vv - med)))
        rr = vr[m]
        rmed = float(np.median(rr))
        rmad = 1.4826 * float(np.median(np.abs(rr - rmed)))
        filas.append({"r_mm": 0.5 * (bins[b] + bins[b + 1]),
                      "n": n, "vt_med": med, "vt_std": mad / np.sqrt(n),
                      "vr_med": rmed, "vr_std": rmad / np.sqrt(n),
                      "vt_mad": mad, "vr_mad": rmad})
    return pd.DataFrame(filas)


def clip_outliers(r, vr, vt, bins, k=5.0):
    """Recorta v_r y v_theta a k*MAD respecto de la mediana de su bin radial."""
    keep = np.ones(len(vt), bool)
    idx = np.digitize(r, bins) - 1
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 8:
            continue
        for arr in (vt, vr):
            v = arr[m]
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med))
            if mad > 0:
                keep[m] &= np.abs(v - med) < k * mad
    return keep


# ---------------------------------------------------------------------------
# Bondad de ajuste (se guarda en el JSON, NO se dibuja en los gráficos)
# ---------------------------------------------------------------------------
def r2(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss = np.sum((y - y.mean()) ** 2)
    if ss <= 0:
        return 0.0
    return float(1 - np.sum((y - yhat) ** 2) / ss)


def r2w(y, yhat, w):
    """R² ponderado por los pesos w (1/SE²)."""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    w = np.asarray(w, float)
    ybar = np.average(y, weights=w)
    ss = np.sum(w * (y - ybar) ** 2)
    if ss <= 0:
        return 0.0
    return float(1 - np.sum(w * (y - yhat) ** 2) / ss)


def _metricas_ajuste(y, yhat, w, n_params):
    """χ², χ² reducido y p-valor de la hipótesis nula (el modelo describe
    los datos).  Se usan los mismos pesos w = 1/σ² del ajuste.

    p = P(χ² ≥ χ²_obs | dof).  Un p pequeño rechaza el modelo; un p alto
    indica que los datos son compatibles con él para los σ usados.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    w = np.asarray(w, float)
    m = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
    y, yhat, w = y[m], yhat[m], w[m]
    n = int(y.size)
    dof = max(n - int(n_params), 0)
    chi2 = float(np.sum(w * (y - yhat) ** 2))
    if dof > 0:
        p_value = float(_chi2_dist.sf(chi2, dof))
        chi2_red = float(chi2 / dof)
    else:
        p_value = None
        chi2_red = None
    return {"chi2": chi2, "chi2_red": chi2_red, "dof": int(dof),
            "p_value": p_value, "n": n, "n_params": int(n_params)}


# ---------------------------------------------------------------------------
# Curvas de los modelos (parametrización del PDF)
# ---------------------------------------------------------------------------
def curva_rankine(r, Omega, R):
    r = np.asarray(r, float)
    return np.where(r <= R, Omega * r, Omega * R ** 2 / r)


def curva_burgers(r, Omega, R):
    """v_θ = (Ω R²/r)(1 - e^{-r²/R²})."""
    r = np.asarray(r, float)
    return Omega * R ** 2 / r * (1.0 - np.exp(-(r ** 2) / R ** 2))


# ---------------------------------------------------------------------------
# Ajustes
# ---------------------------------------------------------------------------
def _slope_lineal(x, y, w):
    """Pendiente de la recta y = m x por el origen, ponderada."""
    m = np.sum(w * x * y) / np.sum(w * x * x)
    return float(m)


def _optimizar_1d(objetivo, lo, hi, n=200, refinamientos=2):
    """Grilla gruesa + refinamiento local.  Devuelve (x_opt, en_limite_inf,
    en_limite_sup).  Evita que el óptimo quede cuantizado por la grilla."""
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return lo, True, True
    a, b = lo, hi
    mejor = lo
    for _ in range(refinamientos + 1):
        xs = np.linspace(a, b, n)
        vals = np.array([objetivo(float(x)) for x in xs], float)
        if not np.any(np.isfinite(vals)):
            break
        mejor = float(xs[int(np.nanargmax(vals))])
        paso = (b - a) / max(n - 1, 1)
        a = max(lo, mejor - 2.0 * paso)
        b = min(hi, mejor + 2.0 * paso)
        if b - a <= 1e-10:
            break
    tol = 1e-6 * max(1.0, abs(hi - lo))
    return mejor, mejor <= lo + tol, mejor >= hi - tol


def fit_rankine(r, vt, w):
    """Rankine (ec. 1): v_θ = Ω·min(r, R²/r).  Barre R y resuelve Ω."""
    lo = max(float(r.min()), 0.3)
    hi = float(r.max()) * 1.5 + 1.0

    def obj(R):
        base = np.where(r <= R, r, R ** 2 / r)
        den = np.sum(w * base * base)
        if den <= 0:
            return -np.inf
        Omega = np.sum(w * base * vt) / den
        return r2w(vt, Omega * base, w)

    R, li, ls = _optimizar_1d(obj, lo, hi)
    base = np.where(r <= R, r, R ** 2 / r)
    Omega = np.sum(w * base * vt) / np.sum(w * base * base)
    yhat = Omega * base
    return {"Omega_1_s": float(Omega), "R_mm": float(R),
            "vmax_mm_s": float(Omega * R), "r2": r2w(vt, yhat, w),
            "limite_inferior": bool(li), "limite_superior": bool(ls),
            **_metricas_ajuste(vt, yhat, w, 2)}


def fit_burgers(r, vt, w):
    """Burgers (ec. 2) para v_θ: (Ω R²/r)(1 - e^{-r²/R²}).
    Barre R y resuelve el producto K = Ω R² por mínimos cuadrados lineales."""
    lo = max(float(r.min()), 0.3)
    hi = float(r.max()) * 1.5 + 1.0

    def obj(R):
        base = (1.0 - np.exp(-(r ** 2) / R ** 2)) / r
        den = np.sum(w * base * base)
        if den <= 0:
            return -np.inf
        K = np.sum(w * base * vt) / den
        return r2w(vt, K * base, w)

    R, li, ls = _optimizar_1d(obj, lo, hi)
    base = (1.0 - np.exp(-(r ** 2) / R ** 2)) / r
    K = np.sum(w * base * vt) / np.sum(w * base * base)
    yhat = K * base
    return {"K_mm3_s": float(K), "R_mm": float(R),
            "Omega_1_s": float(K / R ** 2), "r2": r2w(vt, yhat, w),
            "limite_inferior": bool(li), "limite_superior": bool(ls),
            **_metricas_ajuste(vt, yhat, w, 2)}


def fit_radial(r, vr, w):
    """v_r = m r por el origen (Burgers) y recta libre (referencia).

    Se reportan las métricas de los tres modelos radiales:
      * Rankine: v_r = 0 (0 parámetros),
      * radial:  v_r = m r (1 parámetro),
      * libre:   v_r = a + m r (2 parámetros).
    """
    m = _slope_lineal(r, vr, w)
    A = np.column_stack([r, np.ones_like(r)])
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A * sw, vr * np.sqrt(w), rcond=None)
    y_rad = m * r
    y_libre = coef[0] * r + coef[1]
    y_cero = np.zeros_like(vr)
    met_rad = _metricas_ajuste(vr, y_rad, w, 1)
    met_libre = _metricas_ajuste(vr, y_libre, w, 2)
    met_cero = _metricas_ajuste(vr, y_cero, w, 0)
    return {"m_1_s": float(m), "intercepto_mm_s": float(coef[1]),
            "pendiente_libre_1_s": float(coef[0]),
            "r2": r2w(vr, y_rad, w),
            "r2_libre": r2w(vr, y_libre, w),
            "r2_cero": r2w(vr, y_cero, w),
            **met_rad,
            **{f"{k}_libre": v for k, v in met_libre.items()},
            **{f"{k}_cero": v for k, v in met_cero.items()}}


def ajustar_todos(r, vt, vr, w, wr):
    """Ajusta Rankine, Burgers y el perfil radial."""
    rk = fit_rankine(r, vt, w)
    bg = fit_burgers(r, vt, w)
    rad = fit_radial(r, vr, wr)
    m = rad["m_1_s"]
    nu_burgers = -m * bg["R_mm"] ** 2 / 2.0            # ec. 4
    return {
        "Rankine": {**rk, "modelo": "Rankine"},
        "Burgers": {**bg, "nu_mm2_s": float(nu_burgers),
                    "m_1_s": float(m), "modelo": "Burgers"},
        "radial": {**rad, "modelo": "radial"},
    }


def agregar_error_medicion(perfil, sigma_v):
    """Suma en cuadratura el error de medición σ_v/√n al error de cada bin."""
    if perfil is None or len(perfil) == 0:
        return perfil
    n = np.maximum(perfil["n"].values.astype(float), 1.0)
    if sigma_v is None or not np.isfinite(sigma_v) or sigma_v <= 0:
        extra = np.zeros_like(n)
    else:
        extra = sigma_v / np.sqrt(n)
    perfil["vt_std_total"] = np.sqrt(perfil["vt_std"].values ** 2 + extra ** 2)
    perfil["vr_std_total"] = np.sqrt(perfil["vr_std"].values ** 2 + extra ** 2)
    return perfil


def perfil_en_centro(x, y, vx, vy, centro, bins, signo, sigma_v):
    """Perfil radial recalculado para un centro distinto (error de centro)."""
    r, th, vr, vt = componentes(x, y, vx, vy, centro[0], centro[1])
    vt = vt * signo
    keep = clip_outliers(r, vr, vt, bins, k=5.0)
    pf = perfil_radial(r[keep], vr[keep], vt[keep], bins)
    return agregar_error_medicion(pf, sigma_v)


def perfil_con_bins(r, vr, vt, bins, sigma_v):
    """Perfil radial recalculado con otro ancho de bin (error de binning)."""
    keep = clip_outliers(r, vr, vt, bins, k=5.0)
    pf = perfil_radial(r[keep], vr[keep], vt[keep], bins)
    return agregar_error_medicion(pf, sigma_v)


def _error_refits(ajustes_lista, modo="std"):
    """Dispersión de cada parámetro entre varios reajustes.
    modo='std' usa la desviación estándar; modo='semiamplitud' usa (máx-mín)/2."""
    if not ajustes_lista:
        return {}
    out = {}
    modelos = [m for m in ajustes_lista[0] if isinstance(ajustes_lista[0][m], dict)]
    for modelo in modelos:
        out[modelo] = {}
        claves = [k for k, v in ajustes_lista[0][modelo].items()
                  if k not in EXCLUIR_AJUSTE
                  and isinstance(v, (int, float, np.integer, np.floating))]
        for k in claves:
            vals = [float(aj[modelo][k]) for aj in ajustes_lista
                    if modelo in aj and k in aj[modelo]
                    and isinstance(aj[modelo][k], (int, float, np.integer, np.floating))
                    and np.isfinite(aj[modelo][k])]
            if len(vals) < 2:
                out[modelo][k] = None
            elif modo == "semiamplitud":
                out[modelo][k] = float((max(vals) - min(vals)) / 2.0)
            else:
                out[modelo][k] = float(np.std(vals, ddof=1))
    return out


def _combinar_errores(bs, fuentes):
    """σ_total² = σ_boot² + σ_cal² + σ_centro² + σ_bin²."""
    modelos = set(bs)
    for fuente in fuentes:
        if fuente:
            modelos.update(fuente)
    total = {}
    for modelo in modelos:
        params = set(bs.get(modelo, {}))
        for fuente in fuentes:
            if fuente:
                params.update(fuente.get(modelo, {}))
        total[modelo] = {}
        for k in params:
            d = bs.get(modelo, {}).get(k)
            sigma = d.get("std") if isinstance(d, dict) else None
            valores = [] if sigma is None else [float(sigma)]
            for fuente in fuentes:
                s = fuente.get(modelo, {}).get(k) if fuente else None
                if s is not None and np.isfinite(s):
                    valores.append(float(s))
            total[modelo][k] = (float(np.sqrt(np.sum(np.square(valores))))
                                if valores else None)
    return total


# ---------------------------------------------------------------------------
# Bootstrap sobre el perfil radial
# ---------------------------------------------------------------------------
def bootstrap_perfil(perfil, n=300, seed=0):
    """Perturba cada mediana según su error total (estadístico + medición) y
    vuelve a ajustar todos los modelos.  Devuelve media y std por parámetro."""
    rng = np.random.default_rng(seed)
    r = perfil["r_mm"].values
    vt = perfil["vt_med"].values
    vr = perfil["vr_med"].values
    vt_std = perfil.get("vt_std_total", perfil["vt_std"]).values
    vr_std = perfil.get("vr_std_total", perfil["vr_std"]).values
    se = np.maximum(vt_std, 1e-9)
    vre = np.maximum(vr_std, 1e-9)
    w = 1.0 / (se ** 2)
    wr = 1.0 / (vre ** 2)

    acc = {}
    for _ in range(n):
        vtb = vt + rng.normal(0, se)
        vrb = vr + rng.normal(0, vre)
        aj = ajustar_todos(r, vtb, vrb, w, wr)
        for modelo, params in aj.items():
            for k, v in params.items():
                if k in EXCLUIR_AJUSTE:
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    acc.setdefault(modelo, {}).setdefault(k, []).append(float(v))

    out = {}
    for modelo, params in acc.items():
        out[modelo] = {}
        for k, vals in params.items():
            a = np.asarray(vals, float)
            a = a[np.isfinite(a)]
            if len(a) == 0:
                out[modelo][k] = {"media": None, "std": None}
            else:
                out[modelo][k] = {
                    "media": float(a.mean()),
                    "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                }
    return out


# ---------------------------------------------------------------------------
# Error de posición empírico (repetición con otro umbral)
# ---------------------------------------------------------------------------
def sigma_posicion(video, bg_path, roi, ppm, filtro_, escala, k_a=5.0, k_b=7.0,
                   frames=None, max_move_px=12.0, min_area=3.0,
                   max_area=300.0):
    """Repite la detección con dos umbrales sobre cuadros sueltos y mide el
    corrimiento de los centroides ya apareados.  Devuelve sigma en px.

    Si `frames` es None se muestrean 5 cuadros repartidos entre el 10 % y el
    90 % del video, para que funcione también con videos cortos.
    """
    from detectar_glitter import filtro, detectar, umbral_robusto
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bg is None:
        print(f"[aviso] no se pudo leer {bg_path}; se omite sigma_posicion")
        return None
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[aviso] no se pudo abrir {video}; se omite sigma_posicion")
        return None
    if frames is None:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            frames = (30, 60, 90, 120, 150)
        else:
            frames = np.unique(
                np.linspace(0.1, 0.9, 5) * (total - 1)).astype(int)
    disp = []
    for fr in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, f = cap.read()
        if not ok:
            continue
        work = cv2.resize(f, None, fx=escala, fy=escala,
                          interpolation=cv2.INTER_AREA)
        fmap = filtro(work, filtro_)
        bgf = cv2.resize(filtro(bg, filtro_), (work.shape[1], work.shape[0]))
        diff = fmap - bgf
        da = np.array(detectar(diff, umbral_robusto(diff, k_a),
                               min_area, max_area, escala=escala))
        db = np.array(detectar(diff, umbral_robusto(diff, k_b),
                               min_area, max_area, escala=escala))
        if len(da) == 0 or len(db) == 0:
            continue
        pa, pb = da[:, :2], db[:, :2]
        d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        j = d.argmin(1)
        i = d.argmin(0)
        for ii, jj in enumerate(j):
            if i[jj] == ii and d[ii, jj] < max_move_px:
                disp.append(pa[ii] - pb[jj])
    cap.release()
    if not disp:
        return None
    disp = np.array(disp)
    return float(np.sqrt(np.mean(disp[:, 0] ** 2 + disp[:, 1] ** 2)))


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _ylim_robusto(perfil, col_med, col_std, extras=(), factor=3.0):
    med = perfil[col_med].values
    std = perfil[col_std].values
    lo = float(np.nanmin(med - factor * std))
    hi = float(np.nanmax(med + factor * std))
    for e in extras:
        e = np.asarray(e, float)
        if len(e):
            lo = min(lo, float(np.nanmin(e)))
            hi = max(hi, float(np.nanmax(e)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def plot_perfiles(perfil, ajustes, d, bin_mm, rr, curvas_vt, curvas_vr,
                  r_modelo_mm=None):
    """Dos figuras (6×6): v_θ vs r y v_r vs r, con los modelos superpuestos.
    Los puntos medidos llevan la barra de error de su bin (error estándar).

    `r_modelo_mm` es opcional: si se indica, se dibuja una línea vertical
    punteada en ese radio (por ejemplo, hasta dónde se considera válido el
    modelo).  No afecta ningún ajuste.
    """
    plt = estilo_graficos()
    from matplotlib.ticker import MultipleLocator

    rmed = perfil["r_mm"].values
    xerr = 0.5 * bin_mm
    col_vt_std = "vt_std_total" if "vt_std_total" in perfil else "vt_std"
    col_vr_std = "vr_std_total" if "vr_std_total" in perfil else "vr_std"

    # ---------------------------- v_theta ----------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(rmed, perfil["vt_med"], xerr=xerr, yerr=perfil[col_vt_std],
                fmt="o", ms=5, color="k", ecolor="k", elinewidth=1,
                capsize=3, label="Datos")
    estilos = ["-", "--", "-.", ":"]
    colores = ["C3", "C2", "C0", "C4"]
    for (etq, yy), est, col in zip(curvas_vt.items(),
                                   estilos, colores):
        ax.plot(rr, yy, est, color=col, lw=2, label=etq)
    r_lim = r_modelo_mm
    if r_lim is not None:
        ax.axvline(r_lim, color="0.55", ls=":", lw=1.6, zorder=0)
    yl = _ylim_robusto(perfil, "vt_med", col_vt_std, list(curvas_vt.values()))
    if yl:
        ax.set_ylim(*yl)
    ax.set_xlabel("$r$ [mm]")
    ax.set_ylabel("$v_\\theta$ [mm/s]")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    if r_lim is not None:
        ticks = [t for t in np.arange(0.0, float(rr.max()) + 1e-9, 5.0)
                 if abs(t - r_lim) > 1e-9]
        ax.set_xticks(sorted(ticks + [r_lim]))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(os.path.join(d, "vtheta_vs_r.png"))
    plt.close(fig)

    # ------------------------------ v_r ------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(rmed, perfil["vr_med"], xerr=xerr, yerr=perfil[col_vr_std],
                fmt="o", ms=5, color="k", ecolor="k", elinewidth=1,
                capsize=3, label="medido")
    for (etq, yy), est, col in zip(curvas_vr.items(),
                                   estilos, colores):
        ax.plot(rr, yy, est, color=col, lw=2, label=etq)
    yl = _ylim_robusto(perfil, "vr_med", col_vr_std, list(curvas_vr.values()))
    if yl:
        ax.set_ylim(*yl)
    ax.set_xlabel("$r$ [mm]")
    ax.set_ylabel("$v_r$ [mm/s]")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(os.path.join(d, "vr_vs_r.png"))
    plt.close(fig)


def plot_centro(S, xs, ys, center, d):
    """Mapa de la búsqueda del centro."""
    plt = estilo_graficos()
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(S, origin="lower",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto",
                   cmap="viridis")
    ax.plot(center[0], center[1], "x", color="red", ms=14, mew=3,
            label="centro del vórtice")
    ax.plot(0.0, 0.0, "+", color="white", ms=14, mew=2, label="centro del RDI")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("dispersión de $v_\\theta$ [mm/s]")
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.legend(loc="best")
    fig.savefig(os.path.join(d, "centro_vortice.png"))
    plt.close(fig)


def plot_densidad(det, roi, ppm, center, d):
    """Densidad de detecciones (mm respecto del RDI) con los dos centros."""
    plt = estilo_graficos()
    x = (det["x_px"].values - roi["cx"]) / ppm
    y = (det["y_px"].values - roi["cy"]) / ppm
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hexbin(x, y, gridsize=60, cmap="magma")
    ax.plot(0.0, 0.0, "+", color="cyan", ms=14, mew=2, label="centro del RDI")
    ax.plot(center[0], center[1], "x", color="lime", ms=14, mew=3,
            label="centro del vórtice")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.legend(loc="best")
    fig.savefig(os.path.join(d, "densidad_detecciones.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Residuos
# ---------------------------------------------------------------------------
MODELOS_RES_VT = [
    ("Rankine", "rankine"),
    ("Burgers", "burgers"),
]
MODELOS_RES_VR = [
    ("Rankine: v_r = 0", "rankine_cero"),
    ("Radial: v_r = m·r", "radial"),
    ("Recta libre: v_r = a + m·r", "libre"),
]


def construir_residuos(perfil, ajustes):
    """Residuos crudos y normalizados de v_θ y v_r para cada modelo."""
    r = perfil["r_mm"].values.astype(float)
    vt = perfil["vt_med"].values.astype(float)
    vr = perfil["vr_med"].values.astype(float)
    st = perfil.get("vt_std_total", perfil["vt_std"]).values.astype(float)
    sr = perfil.get("vr_std_total", perfil["vr_std"]).values.astype(float)
    st = np.maximum(st, 1e-12)
    sr = np.maximum(sr, 1e-12)

    rk = ajustes["Rankine"]
    bg = ajustes["Burgers"]
    rad = ajustes["radial"]

    pred_vt = {
        "rankine": curva_rankine(r, rk["Omega_1_s"], rk["R_mm"]),
        "burgers": curva_burgers(r, bg["Omega_1_s"], bg["R_mm"]),
    }
    pred_vr = {
        "rankine_cero": np.zeros_like(r),
        "radial": rad["m_1_s"] * r,
        "libre": rad["pendiente_libre_1_s"] * r + rad["intercepto_mm_s"],
    }

    df = pd.DataFrame({
        "r_mm": r, "n": perfil["n"].values,
        "vt_med": vt, "vt_sigma": st,
        "vr_med": vr, "vr_sigma": sr,
    })
    for slug, yhat in pred_vt.items():
        df[f"vt_pred_{slug}"] = yhat
        df[f"vt_res_{slug}"] = vt - yhat
        df[f"vt_z_{slug}"] = (vt - yhat) / st
    for slug, yhat in pred_vr.items():
        df[f"vr_pred_{slug}"] = yhat
        df[f"vr_res_{slug}"] = vr - yhat
        df[f"vr_z_{slug}"] = (vr - yhat) / sr
    return df


def plot_residuos(residuos, d):
    """Dos paneles: residuos normalizados de v_θ y de v_r."""
    plt = estilo_graficos()
    r = residuos["r_mm"].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    estilos = ["o-", "s--", "^-.", "d:"]
    colores_vt = ["C3", "C2", "C0", "C4"]
    colores_vr = ["C4", "C1", "C5", "C3"]

    ax = axes[0]
    ax.axhline(0.0, color="k", lw=1)
    for z in (-2.0, -1.0, 1.0, 2.0):
        ax.axhline(z, color="0.6", lw=0.8,
                   ls="--" if abs(z) == 1.0 else ":")
    for (etq, slug), est, col in zip(MODELOS_RES_VT, estilos, colores_vt):
        ax.plot(r, residuos[f"vt_z_{slug}"], est, ms=4, lw=1.1, color=col,
                label=etq)
    ax.set_ylabel("$(v_\\theta^{dato}-v_\\theta^{modelo})/\\sigma$")
    ax.set_title("Residuos normalizados de $v_\\theta$")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncol=3, fontsize=11)

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=1)
    for z in (-2.0, -1.0, 1.0, 2.0):
        ax.axhline(z, color="0.6", lw=0.8,
                   ls="--" if abs(z) == 1.0 else ":")
    for (etq, slug), est, col in zip(MODELOS_RES_VR, estilos, colores_vr):
        ax.plot(r, residuos[f"vr_z_{slug}"], est, ms=4, lw=1.1, color=col,
                label=etq)
    ax.set_xlabel("$r$ [mm]")
    ax.set_ylabel("$(v_r^{dato}-v_r^{modelo})/\\sigma$")
    ax.set_title("Residuos normalizados de $v_r$")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncol=3, fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(d, "residuos.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Informe HTML
# ---------------------------------------------------------------------------
NOMBRES_PAR = {
    "Omega_1_s": "Ω [1/s]", "R_mm": "R [mm]", "vmax_mm_s": "v_max [mm/s]",
    "K_mm3_s": "K = ΩR² [mm³/s]", "nu_mm2_s": "ν [mm²/s]",
    "m_1_s": "m [1/s]",
    "intercepto_mm_s": "intercepto [mm/s]",
    "pendiente_libre_1_s": "pendiente libre [1/s]",
}
NOMBRES_MODELO = {
    "Rankine": "Rankine (ec. 1)",
    "Burgers": "Burgers (ec. 2)",
    "radial": "Radial v_r = m r",
}
# Descripción en palabras de cada parámetro, por modelo.
DESCRIPCION_PAR = {
    ("Rankine", "Omega_1_s"):
        "velocidad angular del núcleo rígido (rotación de cuerpo rígido)",
    ("Rankine", "R_mm"):
        "radio del núcleo: posición del máximo de v_θ",
    ("Rankine", "vmax_mm_s"):
        "velocidad tangencial máxima, v_max = ΩR",
    ("Burgers", "K_mm3_s"):
        "amplitud del perfil tangencial; K = ΩR² es lo que se ajusta linealmente",
    ("Burgers", "R_mm"):
        "radio característico del decaimiento exponencial de v_θ",
    ("Burgers", "Omega_1_s"):
        "velocidad angular inferida como Ω = K/R²",
    ("Burgers", "nu_mm2_s"):
        "viscosidad cinemática estimada con ν = -mR²/2",
    ("Burgers", "m_1_s"):
        "pendiente radial medida: v_r = m r",
    ("radial", "m_1_s"):
        "pendiente de la velocidad radial: v_r = m r",
    ("radial", "intercepto_mm_s"):
        "ordenada al origen de la recta libre (diagnóstico)",
    ("radial", "pendiente_libre_1_s"):
        "pendiente de la recta libre (diagnóstico)",
}


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return "%.4g" % v
    return str(v)


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "—"
    if p < 1e-12:
        return "&lt; 1e-12"
    if p < 1e-3:
        return "%.2e" % p
    if p > 0.999:
        return "&gt; 0.999"
    return "%.3f" % p


def _veredicto(p):
    if p is None:
        return "—"
    if p < 0.05:
        return '<span class="bad">rechaza H₀</span>'
    return '<span class="ok">compatible</span>'


def _met(aj, suf=""):
    """Extrae las métricas de un ajuste, con sufijo opcional."""
    return {k: aj.get(k + suf) for k in
            ("chi2", "chi2_red", "dof", "p_value", "n", "n_params")}


def _tabla_estadistica(items):
    """items: lista de (etiqueta, métricas, R²)."""
    filas = []
    for etq, met, r2v in items:
        p = met.get("p_value")
        filas.append(
            "<tr><td>%s</td><td class=\"num\">%s</td>"
            "<td class=\"num\">%s</td><td class=\"num\">%s</td>"
            "<td class=\"num\">%s</td><td class=\"num\">%s</td>"
            "<td class=\"num\">%s</td><td class=\"num\">%s</td>"
            "<td>%s</td></tr>"
            % (etq, _fmt(met.get("n")), _fmt(met.get("n_params")),
               _fmt(met.get("chi2")), _fmt(met.get("dof")),
               _fmt(met.get("chi2_red")), _fmt_p(p), _fmt(r2v),
               _veredicto(p)))
    return (
        "<table><thead><tr><th>Modelo</th>"
        "<th class=\"num\">N</th><th class=\"num\">k</th>"
        "<th class=\"num\">χ²</th><th class=\"num\">dof</th>"
        "<th class=\"num\">χ²/dof</th><th class=\"num\">p-valor</th>"
        "<th class=\"num\">R² ponderado</th><th>Veredicto</th>"
        "</tr></thead><tbody>%s</tbody></table>" % "".join(filas))


def _tabla_residuos(residuos, modelos, var):
    """Tabla de residuos normalizados por bin radial.

    var='vt' o 'vr'; modelos es [(etiqueta, slug), ...].
    """
    cabecera = "".join("<th class=\"num\">%s</th>" % etq
                       for etq, _ in modelos)
    filas = []
    for _, fila in residuos.iterrows():
        celdas = []
        for _, slug in modelos:
            z = float(fila["%s_z_%s" % (var, slug)])
            estilo = (" style=\"color:#b3261e;font-weight:600\""
                      if abs(z) > 2.0 else "")
            celdas.append("<td class=\"num\"%s>%s</td>"
                          % (estilo, _fmt(z)))
        filas.append(
            "<tr><td class=\"num\">%s</td><td class=\"num\">%d</td>%s</tr>"
            % (_fmt(float(fila["r_mm"])), int(fila["n"]), "".join(celdas)))
    return (
        "<table><thead><tr><th class=\"num\">r [mm]</th>"
        "<th class=\"num\">n</th>%s</tr></thead><tbody>%s</tbody></table>"
        % (cabecera, "".join(filas)))


def _data_uri(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _video_data_uri(path, crf=28):
    if not path or not os.path.exists(path):
        return None
    exe = _ffmpeg_exe()
    entrada, tmp = path, None
    if exe:
        tmp = os.path.join(os.path.dirname(os.path.abspath(path)), "_informe_tmp.mp4")
        cmd = [exe, "-y", "-loglevel", "error", "-i", path,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", tmp]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            entrada = tmp
        except Exception as e:
            print(f"[aviso] no se pudo transcodificar el video ({e})")
            tmp = None
    try:
        with open(entrada, "rb") as fh:
            datos = base64.b64encode(fh.read()).decode("ascii")
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return "data:video/mp4;base64," + datos


PLANTILLA_INFORME = Template(r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe — $titulo</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f5f6f8; color: #1c1e21; }
  main { max-width: 1100px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 1.55rem; margin: 0 0 4px; }
  h2 { font-size: 1.05rem; margin: 30px 0 8px; padding-bottom: 4px;
       border-bottom: 1px solid #dcdde0; }
  .sub { color: #606770; font-size: .9rem; margin-bottom: 6px; }
  table { border-collapse: collapse; width: 100%; background: #fff;
          border-radius: 6px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,.06); }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eceef1;
           font-size: .92rem; }
  th { background: #f0f2f5; font-weight: 600; }
  tr:last-child td { border-bottom: none; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  figure { margin: 14px 0; background: #fff; border-radius: 6px; padding: 8px;
           box-shadow: 0 1px 2px rgba(0,0,0,.06); }
  figure img, figure video { width: 100%; display: block; border-radius: 4px; }
  figcaption { color: #606770; font-size: .85rem; padding: 6px 4px 0; }
  .nota { color: #606770; font-size: .85rem; }
  .ok { color: #1b7f3b; font-weight: 600; }
  .bad { color: #b3261e; font-weight: 600; }
  .warn { color: #9a6700; font-weight: 600; }
  .caja { background: #fff; border-left: 4px solid #1f77b4; border-radius: 4px;
          padding: 10px 14px; margin: 12px 0; box-shadow: 0 1px 2px rgba(0,0,0,.05);
          font-size: .9rem; line-height: 1.55; }
  h3 { font-size: .98rem; margin: 20px 0 6px; }
  code { background:#f0f2f5; padding:1px 4px; border-radius:4px; }
</style>
</head>
<body>
<main>
  <h1>Análisis del vórtice (PTV sin identidad)</h1>
  <div class="sub">$titulo &nbsp;·&nbsp; generado el $fecha &nbsp;·&nbsp;
    <a href="../../METODOLOGIA.html">Metodología y ecuaciones</a></div>

  <h2>Resumen</h2>
  <table><tbody>$resumen_filas</tbody></table>

  <h2>Parámetros estimados</h2>
  <table>
    <thead><tr><th>Modelo</th><th>Parámetro</th><th>Significado</th>
      <th class="num">Valor ± error</th></tr></thead>
    <tbody>$ajustes_filas</tbody>
  </table>

  <h2>Estadística de los ajustes</h2>
  <div class="caja">
    <b>Hipótesis nula (H₀):</b> el modelo describe los datos y los σ usados son
    correctos. Se calcula <b>χ² = Σ [ (y<sub>i</sub> − ŷ<sub>i</sub>) / σ<sub>i</sub> ]²</b>,
    con <b>dof = N − k</b> y <b>p = P(χ² ≥ χ²<sub>obs</sub> | dof)</b>.
    Un <b>p &lt; 0,05 rechaza H₀</b>; un <b>χ²/dof ≈ 1</b> indica que los
    errores están bien estimados. Un χ²/dof mucho mayor que 1 puede significar
    que el modelo no describe la forma del perfil o que los σ son demasiado
    pequeños (errores sistemáticos no incluidos).
  </div>

  <h3>Velocidad tangencial v<sub>θ</sub></h3>
  $tabla_stats_vt

  <h3>Velocidad radial v<sub>r</sub></h3>
  $tabla_stats_vr

  <h2>Residuos</h2>
  <figure><img src="$img_residuos" alt="residuos normalizados">
    <figcaption>Residuos normalizados (dato − modelo)/σ para
      v<sub>θ</sub> (arriba) y v<sub>r</sub> (abajo). Las líneas punteadas
      marcan ±1 y ±2 σ.</figcaption></figure>
  <p class="nota">Valores en rojo: |residuo normalizado| &gt; 2.
    El archivo <a href="residuos.csv">residuos.csv</a> contiene además los
    residuos crudos y las predicciones de cada modelo.</p>
  <h3>Residuos de v<sub>θ</sub> por bin radial</h3>
  $tabla_res_vt
  <h3>Residuos de v<sub>r</sub> por bin radial</h3>
  $tabla_res_vr

  <h2>Cómo se estiman los parámetros</h2>
  <ul class="nota">$cadena_estimacion</ul>

  <h2>Notas sobre las fórmulas (docs/analisis_formulas.pdf)</h2>
  <ul class="nota">$notas</ul>

  $seccion_video

  <h2>Gráficos</h2>
  <figure><img src="$img_vtheta" alt="v_theta vs r">
    <figcaption>Velocidad tangencial v<sub>θ</sub> vs radio con los modelos
      superpuestos.</figcaption></figure>
  <figure><img src="$img_vr" alt="v_r vs r">
    <figcaption>Velocidad radial v<sub>r</sub> vs radio con los modelos superpuestos.</figcaption></figure>
  <figure><img src="$img_centro" alt="centro del vórtice">
    <figcaption>Cálculo del centro: dispersión de v<sub>θ</sub> para cada centro
      candidato; el mínimo (cruz roja) es el centro del vórtice.</figcaption></figure>
  <figure><img src="$img_densidad" alt="densidad de detecciones">
    <figcaption>Densidad de detecciones y centros (ROI y vórtice).</figcaption></figure>
</main>
</body>
</html>
""")


def _params_no_acotados(modelo, aj):
    li = bool(aj.get("limite_inferior", False))
    ls = bool(aj.get("limite_superior", False))
    if not (li or ls):
        return set()
    if modelo == "Rankine":
        return {"R_mm"}
    if modelo == "Burgers":
        return {"K_mm3_s", "R_mm", "nu_mm2_s"}
    return set()


def _celda_limite(k, v, aj):
    li = bool(aj.get("limite_inferior", False))
    ls = bool(aj.get("limite_superior", False))
    if k == "R_mm":
        signo = "≥" if ls else "≤"
        return "%s %s (no acotado)" % (signo, _fmt(v))
    return "— (no acotado)"


def escribir_informe(d, resultados, ajustes, errores, notas, cadena,
                     video_path=None, residuos=None):
    r = resultados
    sentido = r.get("sentido_giro")
    if not sentido:
        sentido = ("antihorario" if r.get("sentido_giro_signo", 1) < 0
                   else "horario")
    filas_resumen = "".join(
        '<tr><td>%s</td><td class="num">%s</td></tr>' % (k, v) for k, v in [
            ("Video", r.get("video") or "—"),
            ("Calibración", "%s px/mm" % _fmt(r.get("px_per_mm"))),
            ("Cuadros por segundo", _fmt(r.get("fps"))),
            ("Detecciones", _fmt(r.get("n_detecciones"))),
            ("Vínculos (usados / total)", "%s / %s" % (
                _fmt(r.get("n_vinculos_usados")), _fmt(r.get("n_vinculos")))),
            ("Radio máximo [mm]", _fmt(r.get("rmax_mm"))),
            ("Centro del vórtice [mm] (rel. ROI)", "(%s, %s)" % (
                _fmt(r["centro_mm_respecto_roi"][0]),
                _fmt(r["centro_mm_respecto_roi"][1]))),
            ("Centro del vórtice [px]", "(%s, %s)" % (
                _fmt(r["centro_px"][0]), _fmt(r["centro_px"][1]))),
            ("Sentido de giro", sentido),
            ("σ de posición [px]", _fmt(r.get("sigma_pos_px"))),
            ("σ de velocidad [mm/s]", _fmt(r.get("sigma_v_mm_s"))),
        ])

    filas_ajustes = []
    for m, aj in ajustes.items():
        if not isinstance(aj, dict):
            continue
        etq = NOMBRES_MODELO.get(m, m)
        items = [(k, v) for k, v in aj.items()
                 if k not in EXCLUIR_AJUSTE
                 and not k.startswith("_")]
        no_acotados = _params_no_acotados(m, aj)
        for i, (k, v) in enumerate(items):
            if k in no_acotados:
                celda = _celda_limite(k, v, aj)
            else:
                sigma = errores.get(m, {}).get(k)
                celda = ("%s ± %s" % (_fmt(v), _fmt(sigma))
                         if sigma is not None else _fmt(v))
            desc = DESCRIPCION_PAR.get((m, k), DESCRIPCION_PAR.get(k, ""))
            filas_ajustes.append(
                '<tr><td>%s</td><td>%s</td><td>%s</td>'
                '<td class="num">%s</td></tr>'
                % (etq if i == 0 else "", NOMBRES_PAR.get(k, k), desc, celda))
    filas_ajustes = "".join(filas_ajustes)

    # ---- tablas de estadística de ajuste ----
    stats_vt = []
    for modelo in ("Rankine", "Burgers"):
        aj = ajustes.get(modelo)
        if isinstance(aj, dict):
            stats_vt.append((NOMBRES_MODELO.get(modelo, modelo),
                             _met(aj), aj.get("r2")))
    tabla_stats_vt = _tabla_estadistica(stats_vt)

    rad = ajustes.get("radial", {})
    stats_vr = [
        ("Rankine: v_r = 0", _met(rad, "_cero"), rad.get("r2_cero")),
        ("Radial: v_r = m·r", _met(rad), rad.get("r2")),
        ("Recta libre: v_r = a + m·r", _met(rad, "_libre"),
         rad.get("r2_libre")),
    ]
    tabla_stats_vr = _tabla_estadistica(stats_vr)

    if residuos is not None and len(residuos):
        tabla_res_vt = _tabla_residuos(residuos, MODELOS_RES_VT, "vt")
        tabla_res_vr = _tabla_residuos(residuos, MODELOS_RES_VR, "vr")
    else:
        tabla_res_vt = tabla_res_vr = ""

    notas_html = "".join("<li>%s</li>" % n for n in notas)
    cadena_html = "".join("<li>%s</li>" % n for n in cadena)

    uri_video = _video_data_uri(video_path) if video_path else None
    if uri_video:
        seccion_video = (
            '<h2>Video de la medición</h2>\n'
            '  <figure><video controls playsinline preload="metadata" '
            'src="%s"></video>\n'
            '    <figcaption>Detecciones superpuestas al video de la medición.'
            '</figcaption></figure>' % uri_video)
    else:
        seccion_video = ""

    html = PLANTILLA_INFORME.substitute(
        titulo=r.get("video") or os.path.basename(os.path.abspath(d)),
        fecha=datetime.now().strftime("%Y-%m-%d %H:%M"),
        resumen_filas=filas_resumen,
        ajustes_filas=filas_ajustes,
        tabla_stats_vt=tabla_stats_vt,
        tabla_stats_vr=tabla_stats_vr,
        tabla_res_vt=tabla_res_vt,
        tabla_res_vr=tabla_res_vr,
        cadena_estimacion=cadena_html,
        notas=notas_html,
        seccion_video=seccion_video,
        img_residuos=_data_uri(os.path.join(d, "residuos.png")),
        img_vtheta=_data_uri(os.path.join(d, "vtheta_vs_r.png")),
        img_vr=_data_uri(os.path.join(d, "vr_vs_r.png")),
        img_centro=_data_uri(os.path.join(d, "centro_vortice.png")),
        img_densidad=_data_uri(os.path.join(d, "densidad_detecciones.png")),
    )
    salida = os.path.join(d, "informe.html")
    with open(salida, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[ok] informe -> {salida}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Análisis del vórtice (PTV sin identidad).")
    p.add_argument("--dir", required=True,
                   help="carpeta con detecciones.csv, vinculos.csv y roi json")
    p.add_argument("--diametro-mm", type=float, default=90.0)
    p.add_argument("--diametro-mm-err", type=float, default=0.5,
                   help="incertidumbre del diámetro usado para calibrar [mm]")
    p.add_argument("--bin-mm", type=float, default=1.0)
    p.add_argument("--center-search-mm", type=float, default=6.0)
    p.add_argument("--center-step-mm", type=float, default=0.5)
    p.add_argument("--center-err-mm", type=float, default=None,
                   help="error de centro para el presupuesto (default: paso/2)")
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--sin-center-opt", action="store_true")
    p.add_argument("--titulo", default=None)
    p.add_argument("--filtro", default="auto")
    p.add_argument("--escala", type=float, default=0.5)
    p.add_argument("--video", default=None,
                   help="video de entrada para calcular sigma_posicion; "
                        "tiene prioridad sobre la ruta guardada en el JSON")
    p.add_argument("--r-modelo-mm", type=float, default=None,
                   help="radio opcional [mm] para marcar en el gráfico como "
                        "límite de validez del modelo")
    p.add_argument("--video-informe", default=None)
    args = p.parse_args()

    d = args.dir
    det = pd.read_csv(os.path.join(d, "detecciones.csv"))
    vin = pd.read_csv(os.path.join(d, "vinculos.csv"))
    with open(os.path.join(d, "detecciones_roi.json")) as fh:
        roi = json.load(fh)
    ppm = roi["px_per_mm"]
    fps = roi["fps"]
    dt = 1.0 / fps
    print(f"[info] {len(det)} detecciones, {len(vin)} vínculos, {ppm:.2f} px/mm, {fps:.2f} fps")

    if "xmid_mm" in vin.columns:
        xm = vin["xmid_mm"].values
        ym = vin["ymid_mm"].values
    else:
        xm = vin["x_mm"].values
        ym = vin["y_mm"].values
    vx = vin["vx_mm_s"].values
    vy = vin["vy_mm_s"].values
    c0 = (0.0, 0.0)

    rmax = float(np.percentile(np.hypot(xm, ym), 99.0))
    bins = np.arange(0.0, rmax + args.bin_mm, args.bin_mm)

    if args.sin_center_opt:
        center, disp, S, xs, ys = c0, None, None, None, None
    else:
        center, disp, S, xs, ys = optimizar_centro(
            xm, ym, vx, vy, c0, args.center_search_mm, args.center_step_mm, bins)
        print(f"[info] centro óptimo (mm respecto del ROI) = "
              f"({center[0]:.2f}, {center[1]:.2f}) mm; dispersión={disp:.1f} mm/s")

    r, th, vr, vt = componentes(xm, ym, vx, vy, center[0], center[1])

    # El eje y de la imagen apunta hacia abajo: con la proyección estándar,
    # v_θ crudo negativo corresponde a un giro antihorario visto desde la cámara.
    # Se guarda el sentido como texto y se invierte el signo solo para ajustar.
    sentido = "antihorario" if np.nanmedian(vt) < 0 else "horario"
    signo = -1.0 if sentido == "antihorario" else 1.0
    vt = vt * signo

    center_px = (roi["cx"] + center[0] * ppm, roi["cy"] + center[1] * ppm)

    # Se conservan los arreglos completos para los reajustes sistemáticos.
    r_all, vr_all, vt_all = r.copy(), vr.copy(), vt.copy()
    keep = clip_outliers(r, vr, vt, bins, k=5.0)
    print(f"[info] vínculos tras recorte robusto: {keep.sum()}/{len(keep)}")
    r, vr, vt = r[keep], vr[keep], vt[keep]

    # ---- error de posición y velocidad (antes de fijar los pesos) ----
    # En modo láser no hay blobs: se usa el sigma estimado por el detector
    # (ruido de flujo) y no se intenta repetir la detección de partículas.
    modo = roi.get("modo", "glitter")
    video = args.video or roi.get("video", "")
    if video and not os.path.isabs(video):
        for base in (os.getcwd(), os.path.abspath(d)):
            cand = os.path.join(base, video)
            if os.path.exists(cand):
                video = cand
                break
    sp = None
    if modo == "laser_flow":
        sigma_v = roi.get("sigma_v_mm_s")
    else:
        bg_path = os.path.join(d, "bg.png")
        if video and os.path.exists(video) and os.path.exists(bg_path):
            filtro_efectivo = roi.get("filtro") or args.filtro
            sp = sigma_posicion(video, bg_path, roi, ppm,
                                filtro_efectivo, args.escala)
        elif video and os.path.exists(video):
            print("[aviso] falta bg.png; se omite sigma_posicion")
        sigma_v = (np.sqrt(2) * (sp / ppm) / dt) if sp else None

    # ---- perfil radial ----
    perfil = perfil_radial(r, vr, vt, bins)
    perfil = agregar_error_medicion(perfil, sigma_v)
    perfil.to_csv(os.path.join(d, "perfil.csv"), index=False)

    rp = perfil["r_mm"].values
    vtp = perfil["vt_med"].values
    vrp = perfil["vr_med"].values
    sep = np.maximum(perfil["vt_std_total"].values, 1e-9)
    verp = np.maximum(perfil["vr_std_total"].values, 1e-9)
    w = 1.0 / (sep ** 2)
    wr = 1.0 / (verp ** 2)

    # ---- ajustes ----
    ajustes = ajustar_todos(rp, vtp, vrp, w, wr)
    rk, bg = ajustes["Rankine"], ajustes["Burgers"]
    nu_burgers = ajustes["Burgers"]["nu_mm2_s"]

    print(f"[info] modelos: Ω_Rankine={rk['Omega_1_s']:.3g} R={rk['R_mm']:.2f} | "
          f"Ω_Burgers={bg['Omega_1_s']:.3g} R={bg['R_mm']:.2f} | "
          f"ν_Burgers={nu_burgers:.3g}")

    # ---- bootstrap (estadístico + medición) ----
    print("[info] bootstrap ...")
    bs = bootstrap_perfil(perfil, n=args.bootstrap)

    # ---- errores sistemáticos: calibración, centro y binning ----
    D = float(roi.get("diametro_mm", args.diametro_mm) or args.diametro_mm)
    sD_roi = roi.get("diametro_mm_err")
    sigma_D = float(args.diametro_mm_err if sD_roi is None else sD_roi)
    eps_cal = sigma_D / max(D, 1e-9)
    aj_cal = []
    for s in (-1.0, 1.0):
        fac = 1.0 + s * eps_cal
        aj_cal.append(ajustar_todos(rp / fac, vtp / fac, vrp / fac, w, wr))
    err_cal = _error_refits(aj_cal, modo="semiamplitud")

    sigma_centro_mm = 0.0 if args.sin_center_opt else (
        args.center_err_mm if args.center_err_mm is not None
        else args.center_step_mm / 2.0)
    aj_centro = []
    if sigma_centro_mm > 0:
        for dx, dy in ((sigma_centro_mm, 0.0), (-sigma_centro_mm, 0.0),
                       (0.0, sigma_centro_mm), (0.0, -sigma_centro_mm)):
            pf = perfil_en_centro(xm, ym, vx, vy,
                                  (center[0] + dx, center[1] + dy),
                                  bins, signo, sigma_v)
            if pf is None or len(pf) < 3:
                continue
            aj_centro.append(ajustar_todos(
                pf["r_mm"].values, pf["vt_med"].values, pf["vr_med"].values,
                1.0 / np.maximum(pf["vt_std_total"].values, 1e-9) ** 2,
                1.0 / np.maximum(pf["vr_std_total"].values, 1e-9) ** 2))
    err_centro = _error_refits(aj_centro, modo="std")

    aj_bin = []
    for frac in (1.0 - 0.25, 1.0 + 0.25):
        b = args.bin_mm * frac
        bins_b = np.arange(0.0, rmax + b, b)
        pf = perfil_con_bins(r_all, vr_all, vt_all, bins_b, sigma_v)
        if pf is None or len(pf) < 3:
            continue
        aj_bin.append(ajustar_todos(
            pf["r_mm"].values, pf["vt_med"].values, pf["vr_med"].values,
            1.0 / np.maximum(pf["vt_std_total"].values, 1e-9) ** 2,
            1.0 / np.maximum(pf["vr_std_total"].values, 1e-9) ** 2))
    err_bin = _error_refits(aj_bin, modo="semiamplitud")

    errores_total = _combinar_errores(bs, [err_cal, err_centro, err_bin])
    errores_fuentes = {"calibracion": err_cal, "centro": err_centro,
                       "binning": err_bin}

    # ---- notas sobre las fórmulas ----
    notas = [
        "Rankine (ec. 1): Ω y R son separables (el codo marca R, la amplitud "
        "Ω). v_max = ΩR. Predice v_r = 0.",
        "Burgers (ec. 2): para cada R se ajusta linealmente K = ΩR². R sale de "
        "la forma del decaimiento y luego Ω = K/R². La viscosidad sale de la "
        "pendiente radial m (v_r = m r) mediante la ec. 4: ν = −m R²/2.",
        "Consistencia de la ec. 2: la solución EXACTA de Burgers con v_r = −(2ν/R²)r "
        "exige el exponente −2r²/R² (δ = R/√2), no −r²/R². La ec. 2 tal como está "
        "queda a un factor √2 en el largo de decaimiento; se ajusta igual el largo "
        "libre y se reporta como R del ajuste.",
        "Si la pendiente radial m es compatible con 0 (dentro de su error), la ν "
        "deducida de v_r NO es confiable (los modelos la obtienen de datos casi "
        "nulos).",
        "Los valores marcados como 'no acotado' tocan el borde del rango de "
        "búsqueda: el parámetro no queda determinado por los datos y no se "
        "reporta un error simétrico.",
    ]

    cadena = [
        "El perfil radial se obtiene de los vínculos (pares de detecciones en "
        "cuadros consecutivos), no de trayectorias largas. En cada bin se usa "
        "la mediana de v_θ y v_r.",
        "Rankine: el codo del perfil fija R y la amplitud fija Ω; después "
        "v_max = ΩR. Es el único modelo donde Ω y R se separan directamente.",
        "Burgers: la curva solo depende de K = ΩR² y de R. El solver recorre R, "
        "resuelve K por mínimos cuadrados lineales (es la amplitud que se ajusta "
        "directamente) y después calcula Ω = K/R². La pendiente radial m da "
        "ν = −mR²/2.",
        "Si el decaimiento exponencial no se ve dentro del radio medido, R "
        "queda no acotado y solo se determina la combinación K. El "
        "informe lo marca explícitamente.",
        "Errores: σ_total se obtiene en cuadratura a partir del bootstrap "
        "(estadístico + medición), la calibración (σ_D = 0.5 mm), la sensibilidad "
        "al centro y el cambio de ancho de bin. Las cantidades derivadas se "
        "recalculan en cada muestra y en cada reajuste.",
    ]

    resultados = {
        "video": args.titulo or os.path.basename(d),
        "px_per_mm": ppm, "fps": fps, "diametro_mm": roi.get("diametro_mm"),
        "diametro_mm_err": sigma_D,
        "n_detecciones": int(len(det)), "n_vinculos": int(len(vin)),
        "n_vinculos_usados": int(keep.sum()),
        "centro_mm_respecto_roi": [float(center[0]), float(center[1])],
        "centro_px": [float(center_px[0]), float(center_px[1])],
        "sentido_giro": sentido,
        "sentido_giro_signo": float(signo),
        "ajustes": ajustes,
        "bootstrap": bs,
        "errores_total": errores_total,
        "errores_fuentes": errores_fuentes,
        "sigma_pos_px": sp,
        "sigma_v_mm_s": sigma_v,
        "dispersion_centro_mm_s": (None if disp is None else float(disp)),
        "rmax_mm": rmax,
        "notas": notas,
        "cadena_estimacion": cadena,
    }
    with open(os.path.join(d, "resultados.json"), "w") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)

    # ---- curvas para graficar ----
    rr = np.linspace(max(rp.min(), 0.5), rp.max(), 400)
    curvas_vt = {
        "Rankine": curva_rankine(rr, rk["Omega_1_s"], rk["R_mm"]),
        "Burgers": curva_burgers(rr, bg["Omega_1_s"], bg["R_mm"]),
    }

    vr_burgers = -(2.0 * nu_burgers / bg["R_mm"] ** 2) * rr
    curvas_vr = {
        "Rankine": np.zeros_like(rr),
        "Burgers": vr_burgers,
    }

    plot_perfiles(perfil, ajustes, d, args.bin_mm, rr, curvas_vt, curvas_vr,
                  r_modelo_mm=args.r_modelo_mm)

    residuos = construir_residuos(perfil, ajustes)
    residuos.to_csv(os.path.join(d, "residuos.csv"), index=False)
    plot_residuos(residuos, d)

    if S is not None:
        plot_centro(S, xs, ys, center, d)
    plot_densidad(det, roi, ppm, center, d)
    video_informe = args.video_informe or os.path.join(d, "deteccion.mp4")
    escribir_informe(d, resultados, ajustes, errores_total, notas, cadena,
                     video_informe, residuos=residuos)

    print("[ok] resultados:")
    for k, v in ajustes.items():
        print(f"     {k}: {json.dumps({kk: vv for kk, vv in v.items()}, ensure_ascii=False)}")
    if sp:
        print(f"     sigma_pos = {sp:.2f} px  ->  sigma_v ≈ {sigma_v:.2f} mm/s")


if __name__ == "__main__":
    main()
