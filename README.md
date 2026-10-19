# PTV sin identidad

Pipeline para medir el **campo de velocidades de un vórtice** a partir de un
video con **muchas partículas trazadoras** (por ejemplo, glitter amarillo).
No se sigue la identidad de cada partícula: se detectan todas en cada cuadro y
se emparejan con el cuadro siguiente mediante **vecino mutuo**. Esos pares
("vínculos") muestrean el campo de velocidades.

La idea conceptual es la de **PIV** (*Particle Image Velocimetry*), pero la
implementación es **PTV sin identidad**: detección de partículas + emparejamiento
puntual, sin trayectorias largas y sin IDs persistentes.

> Este repositorio es una versión genérica del pipeline. No incluye los scripts
> de procesamiento masivo (`procesar_todos.py`), calibración masiva
> (`calibrar_todos.sh`) ni la validación por concentraciones
> (`verificar_viscosidad.py`), porque dependían de la organización particular de
> los datos del laboratorio.

---

## Índice

1. [Qué hace y qué no hace](#1-qué-hace-y-qué-no-hace)
2. [Mapa del repositorio](#2-mapa-del-repositorio)
3. [Instalación](#3-instalación)
4. [Flujo rápido para un video](#4-flujo-rápido-para-un-video)
5. [Calibración (`ajustar_recipiente.py`)](#5-calibración-ajustar_recipientepy)
6. [Detección y vínculos (`detectar_glitter.py`)](#6-detección-y-vínculos-detectar_glitterpy)
7. [Análisis del campo (`analizar_glitter.py`)](#7-análisis-del-campo-analizar_glitterpy)
8. [Formatos de datos](#8-formatos-de-datos)
9. [Cómo se calcula](#9-cómo-se-calcula)
10. [Parámetros y reglas prácticas](#10-parámetros-y-reglas-prácticas)
11. [Variante láser (`detectar_laser.py`)](#11-variante-láser-detectar_laserpy)
12. [Ejemplo incluido](#12-ejemplo-incluido)
13. [Constantes hardcodeadas y cómo cambiarlas](#13-constantes-hardcodeadas-y-cómo-cambiarlas)
14. [Diagnóstico de problemas frecuentes](#14-diagnóstico-de-problemas-frecuentes)
15. [Límites y supuestos](#15-límites-y-supuestos)
16. [Referencias](#16-referencias)
17. [Licencia](#17-licencia)

---

## 1. Qué hace y qué no hace

En cada cuadro se buscan **manchas** (blobs) que cumplen un filtro de color, un
umbral robusto y un rango de área. Cada mancha es una **detección**. Después,
para cada par de cuadros consecutivos \(f\) y \(f+1\), se conserva el par de
detecciones que son **mutuamente más cercanas** y que están dentro de una
compuerta. Ese par es un **vínculo**, y su desplazamiento dividido por
\(\Delta t = 1/fps\) da una velocidad.

| Pregunta | Respuesta en este código |
|---|---|
| ¿Cómo sabe cuál es la partícula? | No lo sabe: no hay ID. Detecta manchas en cada cuadro por separado. |
| ¿Cómo detecta cada una? | Filtro de color, resta de fondo, umbral \(k\cdot\)MAD, morfología y componentes conexas. |
| ¿Cuál es el criterio? | Área del blob entre `min_area` y `max_area` (3–300 px² a escala de proceso por defecto). |
| ¿Cuál es "la misma" en el cuadro siguiente? | El vecino mutuo más cercano dentro de `--gate-mm`. |
| ¿Qué se mide? | El campo de velocidades del fluido, no la historia de cada partícula. |

**Por qué funciona sin seguir a nadie:** el vórtice es estacionario, suave y
axisimétrico. Aunque un vínculo empareje por error dos partículas físicas
distintas, el desplazamiento medido en esa zona sigue siendo representativo de
la velocidad del fluido allí. Importa *dónde* y *cuánto*, no *quién*.

---

## 2. Mapa del repositorio

```text
PTV_sin_identidad/
├── ajustar_recipiente.py     # calibración interactiva (ROI + px/mm)
├── detectar_glitter.py       # detección de blobs + vínculos + video de debug
├── analizar_glitter.py       # perfil radial, modelos, errores e informe
├── detectar_laser.py         # variante láser con flujo óptico denso
├── validar_errores.py        # chequeos de consistencia de resultados.json
├── requirements.txt
├── METODOLOGIA.html          # ecuaciones y propagación de errores
├── EXPLICACION_GLITTER.html  # explicación didáctica del detector
├── docs/
│   └── analisis_formulas.pdf # modelos y bibliografía de referencia
└── ejemplos/
    └── 30_2_5/               # ejemplo de resultados (ver §12)
```

Cada corrida del pipeline crea una carpeta de resultados, por ejemplo
`resultados/<nombre>/`, con:

```text
resultados/<nombre>/
├── recipiente.json        # salida de la calibración
├── bg.png                 # fondo temporal (mediana) a escala de proceso
├── detecciones.csv        # detecciones por cuadro
├── vinculos.csv           # pares f -> f+1 con velocidades
├── detecciones_roi.json   # ROI + px/mm + fps + filtro + ruta del video
├── deteccion.mp4          # video de debug (círculos + flechas)
├── perfil.csv             # perfil radial por bins
├── resultados.json        # ajustes, bootstrap, errores y metadatos
├── residuos.csv           # residuos crudos y normalizados
├── vtheta_vs_r.png
├── vr_vs_r.png
├── centro_vortice.png
├── densidad_detecciones.png
├── residuos.png
└── informe.html           # informe autocontenido
```

---

## 3. Instalación

```bash
python3 -m pip install --user -r requirements.txt
```

`requirements.txt` incluye `numpy`, `pandas`, `scipy`, `opencv-python`,
`matplotlib`, `imageio-ffmpeg` y `PyQt5`.

- `imageio-ffmpeg` se usa para transcodificar el video de debug a H.264 y que
  el navegador lo reproduzca dentro de `informe.html`.
- `PyQt5` sólo hace falta para la ventana interactiva de calibración. En un
  equipo sin pantalla se puede usar `opencv-python-headless` y omitir PyQt5;
  los pasos de detección y análisis corren sin interfaz gráfica.

---

## 4. Flujo rápido para un video

Ejecutar desde la raíz del repositorio. Se usa `videos/<video>.mp4` como
ejemplo; reemplazalo por tu ruta real.

```bash
# 1) Calibrar: ajustar un círculo sobre la zona de diámetro conocido
python3 ajustar_recipiente.py \
    --video videos/ensayo.mp4 --bg \
    --diametro-mm 90 \
    --salida resultados/ensayo/recipiente.json

# 2) Detectar partículas + vínculos + video de debug
python3 detectar_glitter.py \
    --video videos/ensayo.mp4 \
    --roi-json resultados/ensayo/recipiente.json \
    --output resultados/ensayo/detecciones.csv \
    --links  resultados/ensayo/vinculos.csv \
    --bg-cache resultados/ensayo/bg.png \
    --debug-video resultados/ensayo/deteccion.mp4

# 3) Analizar: perfil radial, modelos, errores, figuras e informe
python3 analizar_glitter.py \
    --dir resultados/ensayo \
    --titulo "ensayo" \
    --video videos/ensayo.mp4
```

Notas:

- `--diametro-mm` es el diámetro real de la zona que se ajusta en el paso 1.
  El default es **90.0 mm**; cambialo según tu montaje.
- El paso 2 escribe automáticamente `resultados/ensayo/detecciones_roi.json`
  (el mismo nombre que `--output` pero terminado en `_roi.json`).
- En el paso 3, `--video` es opcional. Sirve para recalcular el error de
  posición cuando la ruta guardada en el JSON ya no existe.
- Si querés procesar varios videos, repetí los tres comandos por video. Este
  repositorio no incluye un orquestador masivo a propósito.

---

## 5. Calibración (`ajustar_recipiente.py`)

La herramienta abre un cuadro del video (o el fondo temporal, que se ve más
limpio) y permite ajustar a mano un círculo sobre una zona de **diámetro real
conocido**. De ahí se obtiene:

```text
px/mm = diámetro_px / diámetro_real_mm
```

Controles:

| Acción | Efecto |
|---|---|
| Arrastrar cerca del borde | Cambiar el radio |
| Arrastrar adentro | Mover el centro |
| Rueda del mouse | Radio (paso grueso) |
| Flechas | Mover el centro 1 px (Shift: 10 px) |
| `+` / `-` | Radio ±1 px (Shift: ±10 px) |
| `g` | Centrar el círculo en el cuadro |
| `s` | Guardar JSON + PNG de verificación |
| `q` / `Esc` | Salir |

Salida JSON (`recipiente.json`):

```json
{
  "cx": 654.85,
  "cy": 1009.85,
  "a": 766.14,
  "b": 766.14,
  "ang": 0.0,
  "diam_px": 766.14,
  "px_per_mm": 8.618,
  "diametro_mm": 88.9,
  "diametro_mm_err": 0.5,
  "fuente": "manual",
  "video": "videos/30_2_5.mp4"
}
```

`a` y `b` son las longitudes completas de los ejes de la elipse; `diam_px` es
el diámetro equivalente. `diametro_mm_err` es la incertidumbre declarada del
diámetro real y se usa en el presupuesto de errores del análisis.

Si no se pasa `--roi-json` a la detección, `detectar_glitter.py` intenta
estimar automáticamente el borde del fluido (rayos desde el centro + ajuste de
elipse) y usa `--diametro-mm` para calibrar. Ese modo automático es sólo un
respaldo: para resultados reproducibles conviene calibrar a mano.

---

## 6. Detección y vínculos (`detectar_glitter.py`)

Pipeline por cuadro:

1. **Fondo temporal.** Se muestrean ~40 cuadros repartidos en el video y se
   calcula la mediana por píxel y por canal. El trazador se mueve y desaparece
   de la mediana; la mesa, el vidrio y los reflejos fijos quedan.
   Implementado en `construir_fondo`.
2. **Filtro de color.** Se calcula un valor por píxel que mide "cuán parecido"
   es al trazador. El default es el filtro amarillo:
   \[
   \text{amarillo} = \frac{G+R}{2} - B
   \]
   También hay `green`, `red`, `blue`, `magenta`, `greenx2`, `white`, `maxc`,
   `sat` y `gray`. Con `--filtro auto` se prueban varios sobre cuadros de
   muestra y se elige el que da más detecciones medianas dentro del ROI.
3. **Resta de fondo.** `diff = filtro(cuadro) − filtro(fondo)`.
4. **Umbral robusto.** Se estima el ruido de `diff` con la MAD:
   \[
   \sigma_{\text{robusto}} = 1.4826 \cdot
   \operatorname{mediana}\left(|diff - \operatorname{mediana}(diff)|\right)
   \]
   y se usa \(\text{umbral} = \max(3.0,\; k\cdot\sigma_{\text{robusto}})\),
   con \(k=6\) por defecto.
5. **Máscara y morfología.** `diff > umbral`, apertura \(2\times2\), cierre
   \(3\times3\) y componentes conexas con conectividad 8.
6. **Blobs.** Cada componente con área entre `--min-area` y `--max-area`
   (3–300 px² a escala de proceso) es una detección.
7. **Centroide subpíxel.** La posición es el centroide pesado por intensidad,
   no el centro de la caja:
   \[
   c_x = \frac{\sum_i I_i x_i}{\sum_i I_i}, \qquad
   c_y = \frac{\sum_i I_i y_i}{\sum_i I_i}
   \]
8. **ROI.** Se descartan las detecciones fuera de la elipse del recipiente
   (`dentro_elipse`, margen 0.98).
9. **Vínculos.** Para cada par de cuadros consecutivos se calcula la matriz de
   distancias. Se conserva el par sólo si la elección es **recíproca** y la
   distancia es \(\le gate\_px = \text{gate-mm}\cdot\text{px/mm}\).
10. **Velocidad.** \(v_x = \Delta x/\Delta t\), \(v_y = \Delta y/\Delta t\),
    con \(\Delta t = 1/fps\).

**Video de debug.** Con `--debug-video` se escribe un video con las detecciones
como círculos verdes y las flechas de los vínculos. Como el desplazamiento
entre cuadros es de pocos píxeles, las flechas se exageran con
`--flecha-ganancia` (default 5; `1` = real). Usar `--sin-flechas` para no
dibujarlas.

Parámetros principales:

| Argumento | Default | Descripción |
|---|---|---|
| `--video` | (req.) | Video de entrada |
| `--output` | (req.) | CSV de detecciones |
| `--links` | — | CSV de vínculos (velocidades) |
| `--roi-json` | — | JSON del círculo ajustado a mano (ROI + px/mm) |
| `--px-per-mm` | — | Calibración explícita (prioridad sobre el JSON) |
| `--diametro-mm` | 90.0 | Diámetro real (sólo si no hay `--roi-json`) |
| `--sin-roi` | False | No restringir a un ROI |
| `--filtro` | `yellow` | Filtro de color (`auto` prueba varios) |
| `--k` | 6 | Umbral = \(k\cdot\)MAD del ruido |
| `--min-area` / `--max-area` | 3 / 300 | Área del blob en px² a escala de proceso |
| `--scale` | 0.5 | Escala de procesamiento respecto del video |
| `--bg-frames` | 40 | Cuadros para el fondo temporal |
| `--bg-cache` | — | Guardar/reusar el fondo (`bg.png`) |
| `--gate-mm` | 2.5 | Distancia máxima de vínculo entre cuadros |
| `--debug-video` / `--debug-scale` | — / 0.4 | Video de debug y su escala |
| `--flecha-ganancia` | 5 | Exageración del largo de las flechas |
| `--sin-flechas` | False | No dibujar flechas de vínculos |
| `--fps`, `--start-frame`, `--end-frame`, `--max-frames` | — | Ajustes finos |

> `--bg-cache` reutiliza el archivo si ya existe, **sin verificar** que
> corresponda al mismo video. Si cambiás de video, borrá el `bg.png` o usá otra
> ruta.

---

## 7. Análisis del campo (`analizar_glitter.py`)

Lee `detecciones.csv`, `vinculos.csv` y `detecciones_roi.json` de `--dir`.

1. **Centro del vórtice.** Se prueba una grilla de centros candidatos
   (default \(\pm6\) mm, paso 0.5 mm) y se elige el que minimiza la dispersión
   robusta (MAD) de \(v_\theta\) dentro de cada anillo radial. La idea es la
   **coherencia azimutal**: si el centro es correcto, todos los puntos de un
   mismo radio tienen aproximadamente la misma \(v_\theta\). Se puede
   desactivar con `--sin-center-opt`.
2. **Polares.** Para cada vínculo, respecto del centro \((c_x,c_y)\):
   \[
   r=\sqrt{(x-c_x)^2+(y-c_y)^2},\qquad
   \theta=\operatorname{atan2}(y-c_y,\;x-c_x)
   \]
   \[
   v_r=v_x\cos\theta+v_y\sin\theta,\qquad
   v_\theta=-v_x\sin\theta+v_y\cos\theta
   \]
3. **Perfil radial.** Bins de `--bin-mm` (default 1 mm). En cada bin con al
   menos 5 puntos se toma la mediana de \(v_\theta\) y de \(v_r\), su MAD y el
   error estándar \(\text{MAD}/\sqrt{n}\).
4. **Recorte de outliers.** Se descartan los valores que se alejan más de
   \(5\cdot\)MAD de la mediana de su bin, tanto para \(v_r\) como para
   \(v_\theta\).
5. **Modelos.** Se ajustan Rankine, Burgers y el perfil radial (ver §9).
6. **Pesos.** Todos los ajustes usan \(w = 1/\sigma_{\text{bin}}^2\).
7. **Bootstrap.** Se perturba cada mediana del perfil con
   \(\mathcal{N}(0,\sigma_{\text{bin}})\), se reajustan todos los modelos y se
   recalculan las cantidades derivadas (300 remuestreos, semilla 0).
8. **Errores sistemáticos.** Se reajusta con:
   - calibración escalada por \(1\pm\sigma_D/D\), con \(\sigma_D=0.5\) mm;
   - centro desplazado \(\pm\) paso/2 en cuatro direcciones;
   - ancho de bin \(25\%\) mayor y menor.
9. **Error de posición.** Se repite la detección con dos umbrales (\(k=5\) y
   \(k=7\)) sobre 5 cuadros repartidos entre el 10% y el 90% del video; se
   emparejan los centroides y se mide su corrimiento. Luego:
   \[
   \sigma_v = \sqrt{2}\,\frac{\sigma_{\text{pos}}}{\text{px/mm}}\,\frac{1}{\Delta t}
   \]
   Si falta `bg.png` o el video, se avisa y se omite este término (no se
   inventa un error).
10. **Informe.** Se escribe `informe.html` autocontenido con figuras y video
    embebidos en base64. El video se transcodifica a H.264 con `imageio-ffmpeg`
    para que el navegador lo reproduzca.

Parámetros principales:

| Argumento | Default | Descripción |
|---|---|---|
| `--dir` | (req.) | Carpeta con los CSV y el JSON del ROI |
| `--diametro-mm` | 90.0 | Fallback si el ROI no trae diámetro |
| `--diametro-mm-err` | 0.5 | Incertidumbre del diámetro de calibración [mm] |
| `--bin-mm` | 1.0 | Ancho de los bins radiales |
| `--center-search-mm` / `--center-step-mm` | 6.0 / 0.5 | Búsqueda del centro |
| `--center-err-mm` | paso/2 | Error de centro para el presupuesto |
| `--bootstrap` | 300 | Remuestreos |
| `--sin-center-opt` | False | Usar el centro del ROI |
| `--titulo` | nombre de la carpeta | Etiqueta del análisis |
| `--filtro` / `--escala` | `auto` / 0.5 | Para el error de posición |
| `--video` | — | Video para recalcular `sigma_posicion` |
| `--r-modelo-mm` | — | Radio opcional a marcar en el gráfico |
| `--video-informe` | `deteccion.mp4` | Video a incrustar en el informe |

Salidas: `perfil.csv`, `resultados.json`, `residuos.csv`, los PNG y
`informe.html`.

### Validación de resultados

`validar_errores.py` no vuelve a ajustar nada: revisa que las cantidades
derivadas de `resultados.json` cumplan las relaciones del modelo y que no haya
errores nulos artificiales. Acepta una carpeta o un `resultados.json`:

```bash
python3 validar_errores.py resultados/ensayo
python3 validar_errores.py resultados/ensayo/resultados.json
```

> El informe enlaza `../../METODOLOGIA.html`. Eso funciona si la carpeta de
> resultados está dos niveles por debajo de la raíz (por ejemplo,
> `resultados/<nombre>/` o `ejemplos/<nombre>/`). Si usás otra estructura,
> editá el enlace en `PLANTILLA_INFORME` dentro de `analizar_glitter.py`.

---

## 8. Formatos de datos

### `detecciones.csv`

| Columna | Significado |
|---|---|
| `frame` | Índice de cuadro |
| `t_s` | Tiempo (`frame/fps`) |
| `x_px`, `y_px` | Centroide en píxeles del frame completo (subpíxel) |
| `area_px` | Área del blob a escala **original** |
| `pico` | Pico de intensidad del blob |
| `x_mm`, `y_mm` | `x_px/ppm`, `y_px/ppm` (no centradas; sólo si hay calibración) |

> `--min-area` y `--max-area` se aplican a escala de **proceso**. Por eso un
> blob puede tener `area_px > 300` en el CSV y aun así haber sido aceptado.

### `vinculos.csv` (modo glitter)

| Columna | Significado |
|---|---|
| `frame` | Cuadro \(f\) (el par es \(f\to f+1\)) |
| `x0_px`, `y0_px`, `x1_px`, `y1_px` | Posiciones en \(f\) y \(f+1\) |
| `dx_px`, `dy_px` | Desplazamiento |
| `vx_px_s`, `vy_px_s` | Velocidad en px/s |
| `xmid_mm`, `ymid_mm` | Punto medio relativo al centro del ROI |
| `r_mm` | Radio respecto del centro del ROI |
| `dx_mm`, `dy_mm` | Desplazamiento en mm |
| `vx_mm_s`, `vy_mm_s` | Velocidad en mm/s |

Las columnas `*_mm` requieren calibración. El análisis usa `xmid_mm` y
`ymid_mm` y luego optimiza un corrimiento del centro.

### `vinculos.csv` (modo láser)

El detector láser escribe `frame, xmid_mm, ymid_mm, vx_mm_s, vy_mm_s`.
`analizar_glitter.py` acepta ambos formatos.

### `perfil.csv`

`r_mm, n, vt_med, vt_std, vr_med, vr_std, vt_mad, vr_mad,
vt_std_total, vr_std_total`

`vt_std`/`vr_std` son errores estándar \(\text{MAD}/\sqrt{n}\).
`*_std_total` agrega en cuadratura el error de medición \(\sigma_v/\sqrt{n}\).

### `resultados.json`

Claves principales:

```text
video, px_per_mm, fps, diametro_mm, diametro_mm_err,
n_detecciones, n_vinculos, n_vinculos_usados,
centro_mm_respecto_roi, centro_px,
sentido_giro, sentido_giro_signo,
ajustes, bootstrap, errores_total, errores_fuentes,
sigma_pos_px, sigma_v_mm_s, dispersion_centro_mm_s,
rmax_mm, notas, cadena_estimacion
```

`ajustes` guarda los parámetros de cada modelo (`Omega_1_s`, `R_mm`,
`K_mm3_s`, `nu_mm2_s`, `m_1_s`, etc.). `bootstrap` da `{media, std}` por
modelo y parámetro. `errores_fuentes` separa calibración, centro y binning.

### `recipiente.json` / `detecciones_roi.json`

```json
{
  "cx": 654.85, "cy": 1009.85,
  "a": 766.14, "b": 766.14, "ang": 0.0,
  "diam_px": 766.14,
  "px_per_mm": 8.618,
  "fps": 58.499,
  "filtro": "magenta",
  "diametro_mm": 88.9,
  "video": "videos/30_2_5.mp4"
}
```

`recipiente.json` agrega `fuente` y `diametro_mm_err`.
`detecciones_roi.json` agrega `fps`, `filtro` y, en modo láser, `modo` y
`sigma_v_mm_s`.

---

## 9. Cómo se calcula

### 9.1. Componentes polares

Para cada vínculo, con centro \((c_x,c_y)\):

\[
r=\sqrt{(x-c_x)^2+(y-c_y)^2},\qquad
\theta=\operatorname{atan2}(y-c_y,\;x-c_x)
\]

\[
v_r=v_x\cos\theta+v_y\sin\theta,\qquad
v_\theta=-v_x\sin\theta+v_y\cos\theta
\]

### 9.2. Modelo de Rankine (ec. 1)

\[
v_\theta(r)=
\begin{cases}
\Omega r, & r\le R,\\[4pt]
\dfrac{\Omega R^2}{r}, & r>R,
\end{cases}
\qquad v_r=0.
\]

\(\Omega\) es la velocidad angular del núcleo rígido, \(R\) el radio del núcleo
y \(v_{\max}=\Omega R\). El código barre \(R\) y resuelve \(\Omega\) por
mínimos cuadrados lineales.

### 9.3. Modelo de Burgers (ec. 2)

\[
v_r=-\frac{2\nu}{R^2}r,\qquad
v_\theta=\frac{\Omega R^2}{r}\left(1-e^{-r^2/R^2}\right).
\]

Se define \(K=\Omega R^2\). Para cada \(R\) candidato se ajusta \(K\)
linealmente (la curva es lineal en \(K\)); después:

\[
\Omega=\frac{K}{R^2},\qquad
\nu=-\frac{mR^2}{2},
\]

donde \(m\) es la pendiente del ajuste radial \(v_r=m r\) (ec. 4).

### 9.4. Perfil radial

\[
v_r(r)\approx m r
\]

El ajuste radial es un diagnóstico: si \(m\) es compatible con 0 dentro de su
error, la viscosidad deducida de \(v_r\) no es confiable. El informe lo marca.

### 9.5. Errores

Por bin:

\[
\sigma_{\text{stat}}=\frac{\text{MAD}}{\sqrt{n}},\qquad
\sigma_{\text{med}}=\frac{\sigma_v}{\sqrt{n}},
\qquad
\sigma_{\text{bin}}^2=\sigma_{\text{stat}}^2+\sigma_{\text{med}}^2.
\]

El error total de cada parámetro combina en cuadratura:

\[
\sigma_{\text{total}}^2=
\sigma_{\text{bootstrap}}^2+
\sigma_{\text{calibración}}^2+
\sigma_{\text{centro}}^2+
\sigma_{\text{binning}}^2.
\]

Las cantidades derivadas (\(\Omega=K/R^2\), \(\nu=-mR^2/2\)) se recalculan en
cada muestra del bootstrap y en cada reajuste sistemático.

### 9.6. Sentido de giro

El eje \(y\) de la imagen apunta hacia abajo. Con la proyección usada, un
\(v_\theta\) crudo negativo corresponde a giro **antihorario** visto desde la
cámara. Se guarda el sentido como texto y se invierte el signo sólo para los
ajustes.

### 9.7. Bondad de ajuste

El informe calcula:

\[
\chi^2=\sum_i\left(\frac{y_i-\hat y_i}{\sigma_i}\right)^2,
\qquad \text{dof}=N-k,
\qquad p=P(\chi^2\ge\chi^2_{\text{obs}}\mid\text{dof}).
\]

Un \(p<0.05\) rechaza la hipótesis nula; un \(\chi^2/\text{dof}\approx1\)
indica que los errores usados son razonables.

---

## 10. Parámetros y reglas prácticas

| Parámetro | Efecto al subirlo | Efecto al bajarlo |
|---|---|---|
| `--k` | Más estricto: detecta menos y más brillante | Detecta más, puede meter ruido |
| `--min-area` | Ignora manchas chicas | Puede contar ruido como partícula |
| `--max-area` | Admite grupos más grandes (pegadas) | Descarta grupos grandes / reflejos |
| `--gate-mm` | Permite saltos grandes: **mete vínculos falsos** | Más seguro, pero puede perder partículas rápidas |
| `--scale` | Mayor resolución (más lento) | Más rápido, pierde partículas chicas |
| `--bin-mm` | Perfil más suave, menos bins | Más detalle, más ruido por bin |
| `--bootstrap` | Errores más estables (más lento) | Más rápido, más ruido Monte Carlo |

Regla para la compuerta:

\[
gate\_mm \approx 1.5\cdot\frac{v_{\max,\text{esperado}}}{fps}.
\]

Si el desplazamiento real entre cuadros es de ~1 mm y la velocidad máxima
física es ~50 mm/s a ~30 fps, `--gate-mm 2.5` es un buen punto de partida.
Una compuerta demasiado grande produce flechas largas y cruzadas,
especialmente en el centro del vórtice, donde hay más densidad y brillo.

---

## 11. Variante láser (`detectar_laser.py`)

En lugar de partículas puntuales, esta variante segmenta una región iluminada
por un láser rojo y estima el movimiento interno con **flujo óptico denso**
(Farnebäck):

1. **Máscara del láser** por exceso de rojo \(R-\max(G,B)\), umbral robusto
   dentro del ROI, morfología y eliminación de componentes chicos.
2. **Flujo óptico denso** sobre la imagen de exceso de rojo con paso alto.
3. **Muestreo en grilla**: cada celda aporta una posición y una velocidad
   media. Escribe `detecciones.csv` y `vinculos.csv` en un formato compatible
   con `analizar_glitter.py`.

Ejemplo:

```bash
python3 detectar_laser.py \
    --video videos/ensayo.mp4 \
    --roi-json resultados/ensayo/recipiente.json \
    --output resultados/ensayo_laser/detecciones.csv \
    --links  resultados/ensayo_laser/vinculos.csv \
    --debug-video resultados/ensayo_laser/deteccion_laser.mp4 \
    --summary-png resultados/ensayo_laser/resumen_laser.png

python3 analizar_glitter.py --dir resultados/ensayo_laser
```

En modo láser, `analizar_glitter.py` no intenta recalcular el error de
posición con blobs: usa el `sigma_v_mm_s` que estima el detector.

Parámetros principales:

| Argumento | Default | Descripción |
|---|---|---|
| `--video` / `--roi-json` | (req.) | Video y calibración |
| `--output` / `--links` | (req.) / auto | CSVs de salida |
| `--grid-mm` | 1.5 | Tamaño de celda de muestreo |
| `--flow-stride` | 4 | Cuadros entre los que se calcula el flujo |
| `--suavizado-mm` | 3.0 | Suavizado espacial del campo (0 lo desactiva) |
| `--laser-pct` | 85.0 | Percentil de exceso de rojo para la máscara |
| `--laser-k` | 0.0 | Piso opcional mediana + k·MAD |
| `--min-area-mm2` | 0.5 | Área mínima de componente iluminada |
| `--min-comp-ratio` | 0.15 | Fracción mínima respecto de la componente mayor |
| `--highpass-sigma` | 5.0 | Paso alto para quedarse con el moteado |
| `--roi-margin` | 0.95 | Margen del ROI |
| `--win-size` / `--levels` | 41 / 4 | Ventana y niveles de Farnebäck |
| `--frame-stride` | 1 | Guardar vínculos cada N cuadros |
| `--debug-paso` / `--debug-cada` | 3 / 1 | Densidad y frecuencia de flechas en debug |

---

## 12. Ejemplo incluido

`ejemplos/30_2_5/` contiene una corrida real del método de glitter:

```text
ejemplos/30_2_5/
├── informe.html               # informe completo, autocontenido
├── perfil.csv
├── resultados.json
├── residuos.csv
├── vtheta_vs_r.png
├── vr_vs_r.png
├── centro_vortice.png
├── densidad_detecciones.png
├── residuos.png
├── recipiente.json
├── detecciones_roi.json
├── muestra_detecciones.csv    # primeras 100 filas
└── muestra_vinculos.csv       # primeras 100 filas
```

El ejemplo **no** incluye `detecciones.csv`, `vinculos.csv`, `bg.png` ni
`deteccion.mp4` para no subir cientos de MB. El `informe.html` ya trae las
figuras y el video embebidos.

Abrir directamente:

```text
ejemplos/30_2_5/informe.html
```

El ejemplo sirve para inspeccionar el formato de salida y el informe. No se
puede volver a analizar tal cual porque se omitieron `detecciones.csv` y
`vinculos.csv` (pesan decenas de MB). Para reproducir una corrida completa,
generá esos dos CSV con `detectar_glitter.py` a partir del video y después
corré `analizar_glitter.py` sobre esa carpeta. Los archivos
`muestra_detecciones.csv` y `muestra_vinculos.csv` muestran el esquema de
columnas con las primeras 100 filas.

Las rutas `video` de los JSON del ejemplo fueron saneadas a
`videos/30_2_5.mp4`. Son ilustrativas: si querés recalcular el error de
posición, pasá el video real con `--video`.

---

## 13. Constantes hardcodeadas y cómo cambiarlas

Estos valores **no** se exponen por línea de comandos. Si necesitás otro
comportamiento, editá la función indicada. La mayoría son elecciones
metodológicas razonables, no parámetros del experimento.

### 13.1. Detección (`detectar_glitter.py`)

| Constante | Valor | Dónde | Cómo cambiarla |
|---|---|---|---|
| Factor MAD → \(\sigma\) | `1.4826` | `sigma_mad` | Editar el factor (sólo si cambiás la definición de MAD) |
| Piso de umbral | `3.0` | `umbral_robusto` | Editar el argumento `minimo` |
| \(k\) por defecto | `6` | `--k` | Usar `--k` |
| Área mínima/máxima | `3` / `300` px² | `--min-area`, `--max-area` | Usar los flags |
| Escala de proceso | `0.5` | `--scale` | Usar `--scale` |
| Cuadros del fondo | `40` | `--bg-frames` | Usar `--bg-frames` |
| Mínimo de cuadros para el fondo | `3` | `construir_fondo` | Editar `max(3, n)` |
| Compuerta de vínculo | `2.5` mm | `--gate-mm` | Usar `--gate-mm` |
| Margen del ROI | `0.98` | `dentro_elipse` | Editar `margen` |
| Candidatos de filtro automático | `yellow, magenta, green, red, greenx2, sat, maxc, white` | `FILTROS_AUTO` | Editar la lista |
| Cuadros para elegir filtro | `8` | `elegir_filtro_auto(..., n=8)` | Editar `n` |
| ROI automático: rayos | `0.25 W` a `0.52 W`, 720 ángulos | `estimar_roi_fluido` | Editar `rr` y `np.linspace` |
| ROI automático: gradiente | `> 2.0` | `estimar_roi_fluido` | Editar `d > 2.0` |
| ROI automático: mínimo de puntos | `30` | `estimar_roi_fluido` | Editar `len(pts) < 30` |
| ROI automático: iteraciones | `3` | `estimar_roi_fluido` | Editar `range(3)` |
| FPS de respaldo | `30` | `main` | Editar el `fallback` |
| Códecs de debug | `avc1, H264, mp4v` | `main` | Editar la tupla |
| Ganancia de flechas | `5` | `--flecha-ganancia` | Usar el flag |
| Escala del video de debug | `0.4` | `--debug-scale` | Usar el flag |

### 13.2. Análisis (`analizar_glitter.py`)

| Constante | Valor | Dónde | Cómo cambiarla |
|---|---|---|---|
| Radio máximo | percentil `99` de \(r\) | `main` | Editar `np.percentile(..., 99.0)` |
| Búsqueda de centro | \(\pm6\) mm, paso `0.5` mm | `--center-search-mm`, `--center-step-mm` | Usar los flags |
| Error de centro por defecto | paso/2 (`0.25` mm) | `main` | Usar `--center-err-mm` |
| Mínimo de puntos por anillo (dispersión) | `8` | `dispersion_vt` | Editar `n < 8` |
| Mínimo de puntos por bin (perfil) | `5` | `perfil_radial` | Editar `n < 5` |
| Mínimo de puntos por bin (recorte) | `8` | `clip_outliers` | Editar `m.sum() < 8` |
| Recorte de outliers | `5·MAD` | `clip_outliers(..., k=5.0)` | Editar `k` |
| Rango de búsqueda de \(R\) | `[max(r_min, 0.3), 1.5·r_max+1]` | `fit_rankine`, `fit_burgers` | Editar `lo`/`hi` |
| Grilla 1D | `200` puntos, 2 refinamientos | `_optimizar_1d` | Editar `n` y `refinamientos` |
| Bootstrap | `300`, semilla `0` | `--bootstrap`, `bootstrap_perfil` | Usar `--bootstrap`; editar `seed` |
| Incertidumbre de calibración | `0.5` mm | `--diametro-mm-err` | Usar el flag |
| Error sistemático de binning | \(\pm25\%\) | `main` | Editar `0.25` |
| `sigma_posicion`: umbrales | \(k=5\), \(k=7\) | `sigma_posicion` | Editar `k_a`/`k_b` |
| `sigma_posicion`: corrimiento máximo | `12` px | `sigma_posicion` | Editar `max_move_px` |
| `sigma_posicion`: cuadros | 5 entre 10% y 90% | `sigma_posicion` | Editar `np.linspace(0.1, 0.9, 5)` |
| Área para `sigma_posicion` | `3` / `300` px² | llamada en `main` | Editar los argumentos `min_area`/`max_area` |
| Significancia del informe | `p < 0.05` | `_veredicto` | Editar el umbral |
| `ylim` robusto de gráficos | factor `3`, padding `8%` | `_ylim_robusto` | Editar `factor` y `pad` |
| Enlace a la metodología | `../../METODOLOGIA.html` | `PLANTILLA_INFORME` | Editar el `href` |

> El viejo diccionario `RADIO_MODELO_MM` (que fijaba un radio de modelo por
> nombre de ensayo) fue eliminado en esta versión genérica. Usá
> `--r-modelo-mm` si querés marcar un radio en el gráfico.

### 13.3. Láser (`detectar_laser.py`)

| Constante | Valor | Dónde | Cómo cambiarla |
|---|---|---|---|
| Percentil de máscara | `85` | `--laser-pct` | Usar el flag |
| Piso \(k\) | `0` | `--laser-k` | Usar el flag |
| Paso alto | `5` px | `--highpass-sigma` | Usar el flag |
| Grilla | `1.5` mm | `--grid-mm` | Usar el flag |
| Flujo cada | `4` cuadros | `--flow-stride` | Usar el flag |
| Suavizado espacial | `3` mm | `--suavizado-mm` | Usar el flag |
| Ventana Farnebäck | `41` | `--win-size` | Usar el flag |
| Niveles Farnebäck | `4` | `--levels` | Usar el flag |
| Margen del ROI | `0.95` | `--roi-margin` | Usar el flag |
| Área mínima | `0.5` mm² | `--min-area-mm2` | Usar el flag |
| Ratio mínimo de componente | `0.15` | `--min-comp-ratio` | Usar el flag |
| Morfología | apertura `3×3`, cierre `9×9` | `mascara_laser` | Editar los kernels |
| Normalización del flujo | percentiles `1` y `99.5` | `preparar_flujo` | Editar los percentiles |
| Exterior del ROI | `127` | `preparar_flujo` | Editar `out[~roi_m] = 127` |
| Cobertura mínima de celda | `0.4` | `muestrear_flujo` | Editar `min_frac` |
| Peso mínimo de suavizado | `0.2` | `suavizar_campo` | Editar `wb[ij] > 0.2` |
| `sigma_pos` fijo | `0.3` px | `main` (cálculo de `sigma_v`) | Editar `sigma_pos_px` |
| FPS de respaldo | `60` | `main` | Editar el `fallback` |
| Diámetro de respaldo | `90.0` mm | `main` | Editar `rj.get("diametro_mm", 90.0)` |

### 13.4. Calibración (`ajustar_recipiente.py`)

| Constante | Valor | Dónde | Cómo cambiarla |
|---|---|---|---|
| Diámetro por defecto | `90.0` mm | `--diametro-mm` | Usar el flag |
| Incertidumbre por defecto | `0.5` mm | `--diametro-mm-err` | Usar el flag |
| Cuadros para el fondo | `30` | `--n-bg` | Usar el flag |
| Escala de visualización | `0.4` | `--scale` | Usar el flag |
| Radio inicial | `0.28·min(W,H)` | `main` | Editar `estado["r"]` |
| Sensibilidad de arrastre | `12` px | `on_press` | Editar `abs(d - ...) < 12` |
| Factor de rueda | `1.03` | `on_scroll` | Editar `1.03` |
| Radio mínimo | `5` px | `on_motion`, `on_scroll`, `on_key` | Editar `max(5.0, ...)` |
| Pasos de teclado | `1` px / `10` px con Shift | `on_key` | Editar `paso` |

### 13.5. Otros valores fijos

- `validar_errores.py` usa una tolerancia relativa `1e-6` para comparar
  cantidades derivadas.
- `analizar_glitter.py` sólo grafica si hay al menos 3 bins; si hay menos,
  omite algunos reajustes sistemáticos.
- El informe usa figuras de `6×6` pulgadas y ticks cada 5 mm.
- El informe marca como "no acotado" un parámetro que toca el borde del rango
  de búsqueda de \(R\).

---

## 14. Diagnóstico de problemas frecuentes

- **No detecta partículas** → bajar `--k` (por ejemplo, 4), subir
  `--max-area` o probar `--filtro green`/`--filtro auto`.
- **Flechas raras o muy largas** → bajar `--gate-mm`; bajar
  `--flecha-ganancia` para inspeccionar. Ver §10.
- **El canal de vínculos queda vacío** → revisar que `--output` y `--links`
  usen la misma escala y que haya calibración (`--roi-json` o `--px-per-mm`).
- **La ventana de calibración no abre** → instalar PyQt5:
  `python3 -m pip install --user PyQt5`.
- **El video de debug no se reproduce en el navegador** → se escribió en
  `mp4v`; el informe intenta transcodificar a H.264 con `imageio-ffmpeg`.
  Verificar que `imageio-ffmpeg` esté instalado.
- **`--bg-cache` reutiliza un fondo viejo** → borrar el `bg.png` o usar otra
  ruta. El script no verifica que el fondo corresponda al video actual.
- **`sigma_pos` se omite** → faltan `bg.png` o el video, o la ruta guardada en
  el JSON ya no existe. Pasar `--video` al análisis.
- **El informe tiene un enlace roto a `METODOLOGIA.html`** → la carpeta de
  resultados no está dos niveles debajo de la raíz. Editar el `href` en
  `PLANTILLA_INFORME`.
- **Resultados viejos con rutas absolutas** → volver a correr la calibración o
  pasar `--video` al análisis.
- **Video corto y `sigma_pos` falla** → en esta versión se muestrean 5 cuadros
  entre el 10% y el 90%; si el video tiene muy pocos cuadros, se usa la lista
  histórica `(30, 60, 90, 120, 150)`. Para videos muy cortos, usar
  `--sin-center-opt` no ayuda: conviene aumentar la duración del video.

---

## 15. Límites y supuestos

El método asume:

- flujo **estacionario** (el campo no cambia durante el video);
- campo **suave** y **axisimétrico** (la velocidad depende principalmente de
  \(r\));
- trazadores que actúan como **pasivos** y siguen al fluido;
- **calibración px/mm conocida**;
- fondo **estático** o que cambia poco.

Dónde puede fallar:

- centro del vórtice: mayor densidad, brillo y partículas solapadas;
- compuerta demasiado grande: vínculos falsos;
- movimiento fuera del plano;
- reflejos y cambios de iluminación;
- calibración y centro como errores sistemáticos.

**No es PIV clásico.** Es un método liviano de detección de partículas +
emparejamiento puntual. La analogía con PIV es conceptual: sembrar trazadores
y medir desplazamientos locales.

---

## 16. Referencias

Las referencias explícitas de los modelos están en
[`docs/analisis_formulas.pdf`](docs/analisis_formulas.pdf):

1. D. J. Acheson. *Elementary Fluid Dynamics*. Oxford University Press, 1990.
2. J. M. Burgers. "A mathematical model illustrating the theory of turbulence".
   *Advances in Applied Mechanics*, Elsevier, 1948.

Referencias metodológicas sugeridas:

- R. J. Adrian. "Particle-imaging techniques for experimental fluid mechanics".
  *Annual Review of Fluid Mechanics*, 23 (1991), 261–304.
- M. Raffel, C. E. Willert, F. Scarano, C. J. Kähler, S. T. Wereley y
  J. Kompenhans. *Particle Image Velocimetry: A Practical Guide*. 3.ª ed.,
  Springer, 2018.
- J. C. Crocker y D. G. Grier. "Methods of digital video microscopy for
  colloidal studies". *Journal of Colloid and Interface Science*, 179(1)
  (1996), 298–310.
- P. J. Rousseeuw y C. Croux. "Alternatives to the median absolute deviation".
  *Journal of the American Statistical Association*, 88(424) (1993),
  1273–1283.
- B. Efron. "Bootstrap methods: Another look at the jackknife".
  *The Annals of Statistics*, 7(1) (1979), 1–26.
- G. Farnebäck. "Two-frame motion estimation based on polynomial expansion".
  *Image Analysis* (SCIA 2003), LNCS 2749, Springer, 2003, pp. 363–370.

Software:

- G. Bradski. "The OpenCV Library". *Dr. Dobb's Journal of Software Tools*,
  25(11) (2000), 120–125.
- S. van der Walt, S. C. Colbert y G. Varoquaux. "The NumPy array: A structure
  for efficient numerical computation". *Computing in Science & Engineering*,
  13(2) (2011), 22–30.
- W. McKinney. "Data structures for statistical computing in Python".
  *Proceedings of the 9th Python in Science Conference* (2010), 56–61.
- J. D. Hunter. "Matplotlib: A 2D graphics environment". *Computing in Science
  & Engineering*, 9(3) (2007), 90–95.

---

## 17. Licencia

GNU General Public License v3.0. Ver [`LICENSE`](LICENSE).
