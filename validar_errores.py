#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validar_errores.py
==================
Chequeos de consistencia de los parámetros y errores guardados en
`resultados/<video>/resultados.json`.

No vuelve a ajustar nada: sólo verifica que las cantidades derivadas cumplan
las relaciones del modelo y que no haya errores nulos artificiales.

Uso:
    python3 validar_errores.py
    python3 validar_errores.py resultados/ensayo
    python3 validar_errores.py resultados/ensayo/resultados.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

TOL = 1e-6


def _rel(a, b):
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    escala = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / escala


def _chequear(nombre, obtenido, esperado):
    r = _rel(obtenido, esperado)
    if r is None:
        return nombre, "sin datos", None
    estado = "ok" if r <= TOL else "FALLA"
    return nombre, estado, r


def validar_archivo(path):
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    aj = r.get("ajustes", {})
    fallas = []

    def add(nombre, estado, rel):
        if estado == "FALLA":
            fallas.append((nombre, rel))

    rk = aj.get("Rankine", {})
    bg = aj.get("Burgers", {})
    rad = aj.get("radial", {})

    if rk:
        add(*_chequear("Rankine: vmax = ΩR",
                       rk.get("vmax_mm_s"),
                       (rk.get("Omega_1_s") or 0.0) * (rk.get("R_mm") or 0.0)))
    if bg:
        add(*_chequear("Burgers: Ω = K/R²",
                       bg.get("Omega_1_s"),
                       (bg.get("K_mm3_s") or 0.0) / (bg.get("R_mm") or np.nan) ** 2))
        add(*_chequear("Burgers: ν = -mR²/2",
                       bg.get("nu_mm2_s"),
                       -(bg.get("m_1_s") or 0.0) * (bg.get("R_mm") or 0.0) ** 2 / 2.0))

    ceros = []
    errores = r.get("errores_total")
    if errores:
        for modelo, params in errores.items():
            for k, s in params.items():
                if s is not None and np.isfinite(s) and abs(s) < 1e-12:
                    ceros.append("%s.%s" % (modelo, k))
    else:
        # Resultados viejos: bootstrap con claves internas.
        bs = r.get("bootstrap", {})
        for k, d in bs.items():
            if isinstance(d, dict) and d.get("std") is not None:
                if abs(d["std"]) < 1e-12:
                    ceros.append("bootstrap.%s" % k)

    return {
        "archivo": path,
        "fallas": fallas,
        "ceros": ceros,
        "tiene_errores_total": bool(errores),
        "ajustes": aj,
    }


def main():
    rutas = sys.argv[1:]
    if not rutas:
        rutas = sorted(glob.glob(os.path.join("resultados", "*", "resultados.json")))
    if not rutas:
        print("No encontré resultados.json")
        return 1

    fallas_totales = 0
    for path in rutas:
        if os.path.isdir(path):
            path = os.path.join(path, "resultados.json")
        if not os.path.exists(path):
            print("== %s ==  [aviso] no existe" % path)
            continue
        res = validar_archivo(path)
        nombre = os.path.basename(os.path.dirname(path))
        print("== %s ==" % nombre)
        if not res["tiene_errores_total"]:
            print("   [aviso] resultado viejo: no tiene errores_total")
        if res["fallas"]:
            fallas_totales += len(res["fallas"])
            for f, rel in res["fallas"]:
                print("   [FALLA] %s  (dif. rel. %.3g)" % (f, rel))
        else:
            print("   fórmulas: ok")
        if res["ceros"]:
            print("   [aviso] errores nulos: %s" % ", ".join(res["ceros"]))
    return 1 if fallas_totales else 0


if __name__ == "__main__":
    raise SystemExit(main())
