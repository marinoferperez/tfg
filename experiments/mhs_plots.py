# BOXPLOTS (fitness y tiempo) -> rendimiento final
# CURVA DE CONVERGENCIA -> dinámica del fitness (mejor/promedio/peor run)
# CURVA DE DIVERSIDAD VS EVALS -> dinámica estructural
# CURVAS CONJUNTAS DE FITNESS VS DIVERSIDAD -> relación fitness vs diversidad
# HISTOGRAMA FITNESS -> estabilidad existente entre las distintas runs
# PREDICTOR CON DIVERSIDAD -> capacidad de mejora de fitness futura


import argparse
import csv
import os
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ADAPTACION_PRIORIDAD = ("age_cec2017", "de_cec2017", "age_qap", "de_qap")
ALGORITMO_PRIORIDAD = ("age", "de")
COLORES_ALGORITMO = {
    "age": "#4e79a7",
    "de": "#f28e2b",
}
DIVERSIDAD_POR_PROBLEMA = {
    "cec2017": (
        ("div_dist_euclidea", "Diversidad (distancia euclidea media al centroide)"),
    ),
    "qap": (
        ("div_media_hamming", "Diversidad (Hamming media normalizada)"),
    ),
}
METRICAS_NORMALIZADAS = {"div_media_hamming"}

def _crear_formateador_cientifico():
    formateador = ScalarFormatter(useMathText=True)
    formateador.set_powerlimits((-3, 3))
    formateador.set_useOffset(True)
    return formateador

def aplicar_formato_ejes_compacto(ax, ejes="both", max_ticks=6):
    if ejes in ("x", "both"):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
        ax.xaxis.set_major_formatter(_crear_formateador_cientifico())
    if ejes in ("y", "both"):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
        ax.yaxis.set_major_formatter(_crear_formateador_cientifico())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=str, default="results/experimentos_mhs/runs.csv")
    parser.add_argument("--outdir", type=str, default="results/experimentos_mhs/plots")
    parser.add_argument(
        "--algoritmo",
        type=str,
        default="ambos",
        choices=["age", "de", "ambos"],
        help="Filtra por algoritmo para graficar",
    )
    parser.add_argument(
        "--problema",
        type=str,
        default="ambos",
        choices=["cec2017", "qap", "ambos"],
        help="Filtra por problema para graficar",
    )
    parser.add_argument(
        "--delta-futuro",
        type=int,
        default=50,
        help="Delta en generaciones para medir mejora futura: fitness(t)-fitness(t+delta)",
    )
    return parser.parse_args()

def ordenar_por_prioridad(valores, prioridad):
    presentes = set(valores)
    ordenados = [v for v in prioridad if v in presentes]
    extras = sorted(presentes.difference(set(ordenados)))
    return ordenados + extras

def cargar_runs(path, algoritmo, problema):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de resultados: {path}")

    filas = []
    with path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for fila in reader:
            fila_alg = fila["algoritmo"].strip().lower()
            fila_prob = fila["problema"].strip().lower()

            if algoritmo != "ambos" and fila_alg != algoritmo:
                continue
            if problema != "ambos" and fila_prob != problema:
                continue

            filas.append(
                {
                    "problema": fila_prob,
                    "algoritmo": fila_alg,
                    "adaptacion": fila["adaptacion"].strip().lower(),
                    "semilla": int(fila["semilla"]) if str(fila.get("semilla", "")).strip() != "" else None,
                    "fitness": float(fila["fitness"]),
                    "tiempo_s": float(fila["tiempo_s"]),
                    "ruta_metricas": fila.get("ruta_metricas", "").strip(),
                }
            )
    return filas


def cargar_curva_desde_logbook(ruta_metricas):
    ruta_logbook = Path(ruta_metricas) / "logbook.csv"
    if not ruta_metricas or not ruta_logbook.exists():
        return None

    evaluaciones = []
    mejor_hasta_ahora = []
    with ruta_logbook.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for fila in reader:
            if "evaluaciones" not in fila:
                continue

            evals = int(float(fila["evaluaciones"]))
            if str(fila.get("mejor_hasta_ahora", "")).strip() != "":
                mejor = float(fila["mejor_hasta_ahora"])
            elif str(fila.get("min/mejor_hasta_ahora", "")).strip() != "":
                mejor = float(fila["min/mejor_hasta_ahora"])
            elif str(fila.get("min", "")).strip() != "":
                mejor = float(fila["min"])
            else:
                continue

            evaluaciones.append(evals)
            mejor_hasta_ahora.append(mejor)

    if len(evaluaciones) == 0:
        return None

    pares = sorted(zip(evaluaciones, mejor_hasta_ahora), key=lambda p: p[0])
    x_limpio = []
    y_limpio = []
    for x, y in pares:
        if len(x_limpio) > 0 and x == x_limpio[-1]:
            y_limpio[-1] = y
        else:
            x_limpio.append(x)
            y_limpio.append(y)

    x = np.asarray(x_limpio, dtype=int)
    y = np.minimum.accumulate(np.asarray(y_limpio, dtype=float))
    return x, y

def cargar_curva_promedio_desde_logbook(ruta_metricas):
    ruta_logbook = Path(ruta_metricas) / "logbook.csv"
    if not ruta_metricas or not ruta_logbook.exists():
        return None

    evaluaciones = []
    fitness_promedio = []
    with ruta_logbook.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for fila in reader:
            if "evaluaciones" not in fila:
                continue

            txt_eval = str(fila.get("evaluaciones", "")).strip()
            txt_promedio = str(fila.get("promedio", "")).strip()
            if txt_eval == "" or txt_promedio == "":
                continue

            evals = int(float(txt_eval))
            promedio = float(txt_promedio)
            if not np.isfinite(promedio):
                continue

            evaluaciones.append(evals)
            fitness_promedio.append(promedio)

    if len(evaluaciones) == 0:
        return None

    pares = sorted(zip(evaluaciones, fitness_promedio), key=lambda p: p[0])
    x_limpio = []
    y_limpio = []
    for x, y in pares:
        if len(x_limpio) > 0 and x == x_limpio[-1]:
            y_limpio[-1] = y
        else:
            x_limpio.append(x)
            y_limpio.append(y)

    x = np.asarray(x_limpio, dtype=int)
    y = np.asarray(y_limpio, dtype=float)
    return x, y

def cargar_curva_metrica_desde_logbook(ruta_metricas, metrica):
    if not ruta_metricas:
        return None

    rutas_npz = sorted(Path(ruta_metricas).glob("dataset_*.npz"))
    if len(rutas_npz) == 0:
        return None

    with np.load(rutas_npz[0], allow_pickle=True) as data:
        if metrica not in data or "generacion" not in data or "eval_id" not in data:
            return None
        arr_gen = np.asarray(data["generacion"], dtype=int).reshape(-1)
        arr_eval = np.asarray(data["eval_id"], dtype=int).reshape(-1)
        arr_val = np.asarray(data[metrica], dtype=float).reshape(-1)

    if not (arr_gen.size == arr_eval.size == arr_val.size):
        return None

    por_generacion = {}
    for gen, evals, valor in zip(arr_gen, arr_eval, arr_val):
        if not np.isfinite(valor):
            continue
        previo = por_generacion.get(int(gen))
        if previo is None or int(evals) >= previo[0]:
            por_generacion[int(gen)] = (int(evals), float(valor))

    if len(por_generacion) == 0:
        return None

    pares = sorted((ev, val) for ev, val in por_generacion.values())
    x = np.asarray([ev for ev, _ in pares], dtype=int)
    y = np.asarray([val for _, val in pares], dtype=float)
    return x, y

def detectar_punto_colapso_por_pendiente(x, y):
    if len(x) < 8:
        return None

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    mascara = (dx > 0.0) & np.isfinite(dx) & np.isfinite(dy)
    if np.sum(mascara) < 6:
        return None

    x_pendiente = x_arr[1:][mascara]
    pendiente = dy[mascara] / dx[mascara]
    mascara_finita = np.isfinite(pendiente)
    if np.sum(mascara_finita) < 6:
        return None

    x_pendiente = x_pendiente[mascara_finita]
    pendiente = pendiente[mascara_finita]

    ventana = max(3, len(pendiente) // 12)
    if ventana % 2 == 0:
        ventana += 1
    ventana = min(ventana, 11)
    if ventana > len(pendiente):
        ventana = len(pendiente) if len(pendiente) % 2 == 1 else len(pendiente) - 1
    if ventana >= 3:
        kernel = np.ones(ventana, dtype=float) / float(ventana)
        pendiente_suav = np.convolve(pendiente, kernel, mode="same")
    else:
        pendiente_suav = pendiente

    pendiente_min = float(np.min(pendiente_suav))
    x_span = max(float(x_arr[-1] - x_arr[0]), 1.0)
    y_span = float(np.max(y_arr) - np.min(y_arr))
    pendiente_ref = max(y_span / x_span, 1e-12)

    # Exigimos una fase inicial de caída "fuerte"; si no aparece, no hay transición clara.
    if pendiente_min >= -2.5 * pendiente_ref:
        return None

    idx_min = int(np.argmin(pendiente_suav))
    umbral_meseta = max(0.20 * abs(pendiente_min), 0.75 * pendiente_ref)
    puntos_meseta = max(3, len(pendiente_suav) // 20)

    for idx in range(idx_min + 1, len(pendiente_suav) - puntos_meseta + 1):
        tramo = pendiente_suav[idx : idx + puntos_meseta]
        if np.all(np.abs(tramo) <= umbral_meseta):
            x_colapso = int(np.rint(x_pendiente[idx]))
            if x_colapso <= x_arr[0] or x_colapso >= x_arr[-1]:
                return None
            return x_colapso

    return None

def alinear_curvas_por_evaluaciones(curvas):
    malla = np.unique(np.concatenate([x for x, _ in curvas]))
    matriz = np.empty((len(curvas), len(malla)), dtype=float)

    for i, (x, y) in enumerate(curvas):
        idx = np.searchsorted(x, malla, side="right") - 1
        idx[idx < 0] = 0
        matriz[i, :] = y[idx]

    return malla, matriz

def alinear_serie_a_malla(x, y, malla):
    idx = np.searchsorted(x, malla, side="right") - 1
    idx[idx < 0] = 0
    return y[idx]

def combinar_series_en_malla(x_1, y_1, x_2, y_2):
    malla = np.unique(np.concatenate([x_1, x_2]))
    y_1_malla = alinear_serie_a_malla(x_1, y_1, malla)
    y_2_malla = alinear_serie_a_malla(x_2, y_2, malla)
    return malla, y_1_malla, y_2_malla

def cargar_diversidad_y_mejora_futura_desde_logbook(ruta_metricas, metrica, delta_pasos):
    if delta_pasos < 1:
        raise ValueError("delta_pasos debe ser >= 1")

    rutas_npz = sorted(Path(ruta_metricas).glob("dataset_*.npz")) if ruta_metricas else []
    ruta_logbook = Path(ruta_metricas) / "logbook.csv"
    if not ruta_metricas or len(rutas_npz) == 0 or not ruta_logbook.exists():
        return None

    with np.load(rutas_npz[0], allow_pickle=True) as data:
        if metrica not in data or "generacion" not in data or "eval_id" not in data:
            return None
        arr_gen = np.asarray(data["generacion"], dtype=int).reshape(-1)
        arr_eval = np.asarray(data["eval_id"], dtype=int).reshape(-1)
        arr_div = np.asarray(data[metrica], dtype=float).reshape(-1)

    if not (arr_gen.size == arr_eval.size == arr_div.size):
        return None

    div_por_generacion = {}
    for gen, evals, div in zip(arr_gen, arr_eval, arr_div):
        if not np.isfinite(div):
            continue
        previo = div_por_generacion.get(int(gen))
        if previo is None or int(evals) >= previo[0]:
            div_por_generacion[int(gen)] = (int(evals), float(div))

    if len(div_por_generacion) == 0:
        return None

    registros = []
    with ruta_logbook.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for fila in reader:
            txt_gen = str(fila.get("generacion", "")).strip()
            txt_eval = str(fila.get("evaluaciones", "")).strip()
            if txt_gen == "" or txt_eval == "":
                continue

            gen = int(float(txt_gen))
            if gen not in div_por_generacion:
                continue

            txt_fit = str(fila.get("mejor_hasta_ahora", "")).strip()
            if txt_fit == "":
                txt_fit = str(fila.get("min/mejor_hasta_ahora", "")).strip()
            if txt_fit == "":
                txt_fit = str(fila.get("min", "")).strip()
            if txt_fit == "":
                continue

            evals = int(float(txt_eval))
            fitness = float(txt_fit)
            _, div = div_por_generacion[gen]
            if not np.isfinite(fitness):
                continue
            registros.append((evals, fitness, div))

    if len(registros) == 0:
        return None

    registros.sort(key=lambda r: r[0])
    evals = []
    fitness = []
    diversidad = []
    for ev, fit, div in registros:
        if len(evals) > 0 and ev == evals[-1]:
            fitness[-1] = fit
            diversidad[-1] = div
        else:
            evals.append(ev)
            fitness.append(fit)
            diversidad.append(div)

    if len(fitness) <= delta_pasos:
        return None

    y_fit = np.minimum.accumulate(np.asarray(fitness, dtype=float))
    x_div = np.asarray(diversidad, dtype=float)

    x = x_div[:-delta_pasos]
    y = y_fit[:-delta_pasos] - y_fit[delta_pasos:]
    y = np.maximum(y, 0.0)
    mejora = y > 0.0

    return x, y, mejora

def calcular_resumen_bins_predictor(x, y, mejora, n_bins=12):
    if len(x) == 0:
        return None

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not (np.isfinite(x_min) and np.isfinite(x_max)):
        return None

    if np.isclose(x_min, x_max):
        centro = np.asarray([x_min], dtype=float)
        mejora_media = np.asarray([float(np.mean(y))], dtype=float)
        prob = np.asarray([float(np.mean(mejora.astype(float)))], dtype=float)
        conteo = np.asarray([len(x)], dtype=int)
        return centro, mejora_media, prob, conteo

    bordes = np.linspace(x_min, x_max, num=n_bins + 1)
    centros = []
    mejora_media = []
    prob = []
    conteos = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mascara = (x >= bordes[i]) & (x < bordes[i + 1])
        else:
            mascara = (x >= bordes[i]) & (x <= bordes[i + 1])
        if not np.any(mascara):
            continue

        centros.append((bordes[i] + bordes[i + 1]) / 2.0)
        mejora_media.append(float(np.mean(y[mascara])))
        prob.append(float(np.mean(mejora[mascara].astype(float))))
        conteos.append(int(np.sum(mascara)))

    if len(centros) == 0:
        return None

    return (
        np.asarray(centros, dtype=float),
        np.asarray(mejora_media, dtype=float),
        np.asarray(prob, dtype=float),
        np.asarray(conteos, dtype=int),
    )

def preparar_predictor_diversidad_mejora_futura(filas, problema, delta_futuro):
    metricas = DIVERSIDAD_POR_PROBLEMA.get(problema, ())
    if len(metricas) == 0:
        return None

    filas_por_algoritmo = {}
    for fila in filas:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    orden_algoritmos = ordenar_algoritmos(list(filas_por_algoritmo.keys()))
    if len(orden_algoritmos) == 0:
        return None

    hay_grafica = False
    paneles_por_algoritmo = {}
    limites_x_por_algoritmo = {}
    limites_y_por_algoritmo = {}

    for algoritmo in orden_algoritmos:
        paneles = []
        series_x_fila = []
        series_y_fila = []
        for metrica, etiqueta_div in metricas:
            xs = []
            ys = []
            mejoras = []
            for fila in filas_por_algoritmo[algoritmo]:
                datos = cargar_diversidad_y_mejora_futura_desde_logbook(
                    fila.get("ruta_metricas", ""), metrica, delta_futuro
                )
                if datos is None:
                    continue
                x_run, y_run, mejora_run = datos
                xs.append(x_run)
                ys.append(y_run)
                mejoras.append(mejora_run)

            if len(xs) == 0:
                paneles.append(None)
                continue

            x = np.concatenate(xs)
            y = np.concatenate(ys)
            mejora = np.concatenate(mejoras)

            color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")
            resumen = calcular_resumen_bins_predictor(x, y, mejora, n_bins=12)
            panel = {
                "metrica": metrica,
                "etiqueta_div": etiqueta_div,
                "nombre_algoritmo": algoritmo.upper(),
                "color": color,
                "x": x,
                "y": y,
                "resumen": resumen,
            }
            paneles.append(panel)
            series_x_fila.append(x)
            series_y_fila.append(y)
            if resumen is not None:
                _, mejora_media, _, _ = resumen
                series_y_fila.append(mejora_media)
            hay_grafica = True

        paneles_por_algoritmo[algoritmo] = paneles
        if all(panel is None or panel["metrica"] in METRICAS_NORMALIZADAS for panel in paneles):
            limites_x_por_algoritmo[algoritmo] = (-0.05, 1.05)
        else:
            limites_x_por_algoritmo[algoritmo] = calcular_limites_compartidos(
                series_x_fila, padding_ratio=0.05, padding_min=1e-6
            )
        limites_y_por_algoritmo[algoritmo] = calcular_limites_compartidos(
            series_y_fila, padding_ratio=0.08, padding_min=1e-6
        )

    if not hay_grafica:
        return None

    return {
        "metricas": metricas,
        "orden_algoritmos": orden_algoritmos,
        "paneles_por_algoritmo": paneles_por_algoritmo,
        "limites_x_por_algoritmo": limites_x_por_algoritmo,
        "limites_y_por_algoritmo": limites_y_por_algoritmo,
    }

def calcular_histograma_instancias(x, n_bins=12, limites=None):
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    if x_arr.size == 0:
        return None

    if limites is None:
        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
    else:
        x_min, x_max = limites

    if not (np.isfinite(x_min) and np.isfinite(x_max)):
        return None

    if np.isclose(x_min, x_max):
        return (
            np.asarray([x_min], dtype=float),
            np.asarray([int(x_arr.size)], dtype=int),
            np.asarray([1.0], dtype=float),
        )

    bordes = np.linspace(x_min, x_max, num=n_bins + 1)
    conteos, _ = np.histogram(x_arr, bins=bordes)
    centros = (bordes[:-1] + bordes[1:]) / 2.0
    anchos = np.diff(bordes)
    return centros, conteos.astype(int), anchos

def ordenar_etiquetas(etiquetas):
    return ordenar_por_prioridad(etiquetas, ADAPTACION_PRIORIDAD)

def generar_boxplot(datos_por_etiqueta, etiquetas, titulo, ylabel, outpath):
    series = [datos_por_etiqueta[e] for e in etiquetas]
    labels = [e.upper() for e in etiquetas]

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        bp = ax.boxplot(series, tick_labels=labels, patch_artist=True, showmeans=True)
    except TypeError:
        bp = ax.boxplot(series, labels=labels, patch_artist=True, showmeans=True)

    colores = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948"]
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colores[i % len(colores)])
        box.set_alpha(0.65)

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    aplicar_formato_ejes_compacto(ax, ejes="y")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

def construir_series_por_adaptacion(filas):
    fitness_por_adaptacion = {}
    tiempo_por_adaptacion = {}
    for fila in filas:
        etiqueta = fila["adaptacion"]
        fitness_por_adaptacion.setdefault(etiqueta, []).append(fila["fitness"])
        tiempo_por_adaptacion.setdefault(etiqueta, []).append(fila["tiempo_s"])

    etiquetas = ordenar_etiquetas(list(fitness_por_adaptacion.keys()))
    return fitness_por_adaptacion, tiempo_por_adaptacion, etiquetas

def ordenar_algoritmos(algoritmos):
    return ordenar_por_prioridad(algoritmos, ALGORITMO_PRIORIDAD)

def extraer_series_finitas(series):
    series_finitas = []
    for serie in series:
        if serie is None:
            continue
        arr = np.asarray(serie, dtype=float).ravel()
        if arr.size == 0:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        series_finitas.append(arr)
    return series_finitas

def calcular_limites_compartidos(series, padding_ratio=0.05, padding_min=0.0, limites_forzados=None):
    series_finitas = extraer_series_finitas(series)
    if len(series_finitas) == 0:
        return None

    minimo = float(min(np.min(serie) for serie in series_finitas))
    maximo = float(max(np.max(serie) for serie in series_finitas))

    if np.isclose(minimo, maximo):
        referencia = max(abs(minimo), 1.0)
        padding = max(referencia * max(padding_ratio, 0.02), padding_min)
    else:
        amplitud = maximo - minimo
        referencia = max(abs(minimo), abs(maximo), 1.0)
        padding = max(amplitud * padding_ratio, referencia * padding_ratio * 0.1, padding_min)

    minimo -= padding
    maximo += padding

    if limites_forzados is not None:
        limite_inf, limite_sup = limites_forzados
        if limite_inf is not None:
            minimo = float(limite_inf)
        if limite_sup is not None:
            maximo = float(limite_sup)

    return minimo, maximo

def aplicar_limites_compartidos(ax, eje, limites):
    if limites is None:
        return
    if eje == "x":
        ax.set_xlim(*limites)
    elif eje == "y":
        ax.set_ylim(*limites)
    else:
        raise ValueError(f"Eje no soportado: {eje}")

def generar_histograma_fitness_final(filas, titulo, outpath):
    fitness_por_algoritmo = {}
    for fila in filas:
        fitness_por_algoritmo.setdefault(fila["algoritmo"], []).append(float(fila["fitness"]))

    algoritmos = ordenar_algoritmos(list(fitness_por_algoritmo.keys()))
    if len(algoritmos) == 0:
        return False

    datos_por_algoritmo = {
        algoritmo: np.asarray(fitness_por_algoritmo[algoritmo], dtype=float) for algoritmo in algoritmos
    }
    datos_validos = [datos for datos in datos_por_algoritmo.values() if len(datos) > 0]
    if len(datos_validos) == 0:
        return False

    n_bins = max(12, int(np.sqrt(sum(len(datos) for datos in datos_validos)) * 1.5))
    limites_x = calcular_limites_compartidos(datos_validos, padding_ratio=0.12, padding_min=1.0)
    bin_edges = np.linspace(limites_x[0], limites_x[1], num=n_bins + 1)

    max_conteo = 0
    for datos in datos_validos:
        conteos, _ = np.histogram(datos, bins=bin_edges)
        if conteos.size > 0:
            max_conteo = max(max_conteo, int(np.max(conteos)))
    limite_y = (0.0, max_conteo + max(max_conteo * 0.08, 1.0))

    fig, axes = plt.subplots(
        len(algoritmos),
        1,
        figsize=(10, 4.8 * len(algoritmos)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes[:, 0]

    for i, algoritmo in enumerate(algoritmos):
        ax = axes[i]
        datos = datos_por_algoritmo[algoritmo]
        if len(datos) == 0:
            continue

        color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")

        media = float(np.mean(datos))
        mediana = float(np.median(datos))

        ax.hist(datos, bins=bin_edges, color=color, edgecolor="white", alpha=0.78)

        ax.axvline(media, color="#222222", linestyle="--", linewidth=1.2, label="media")
        ax.axvline(mediana, color="#222222", linestyle=":", linewidth=1.2, label="mediana")

        ax.set_title(f"{algoritmo.upper()} (n={len(datos)})")
        if i == len(algoritmos) - 1:
            ax.set_xlabel("Fitness final")
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)
        aplicar_formato_ejes_compacto(ax, ejes="x")
        aplicar_limites_compartidos(ax, "x", limites_x)
        aplicar_limites_compartidos(ax, "y", limite_y)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(titulo)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    return True

def generar_curva_convergencia(filas, titulo, outpath):
    filas_por_algoritmo = {}
    for fila in filas:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    orden_algoritmos = ordenar_algoritmos(list(filas_por_algoritmo.keys()))
    series_por_algoritmo = []

    for algoritmo in orden_algoritmos:
        curvas = []
        semillas = []
        for fila in filas_por_algoritmo[algoritmo]:
            curva = cargar_curva_desde_logbook(fila.get("ruta_metricas", ""))
            if curva is None:
                continue
            curvas.append(curva)
            semillas.append(fila.get("semilla"))

        if len(curvas) == 0:
            continue

        x, y_runs = alinear_curvas_por_evaluaciones(curvas)
        media = np.mean(y_runs, axis=0)
        desv = np.std(y_runs, axis=0)

        fitness_final = np.asarray([curva[1][-1] for curva in curvas], dtype=float)
        idx_mejor = int(np.argmin(fitness_final))
        idx_peor = int(np.argmax(fitness_final))
        x_mejor, y_mejor = curvas[idx_mejor]
        x_peor, y_peor = curvas[idx_peor]
        seed_mejor = semillas[idx_mejor]
        seed_peor = semillas[idx_peor]

        color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")
        nombre = algoritmo.upper()

        series_por_algoritmo.append(
            {
                "algoritmo": algoritmo,
                "nombre": nombre,
                "color": color,
                "n_curvas": len(curvas),
                "x": x,
                "media": media,
                "desv": desv,
                "x_mejor": x_mejor,
                "y_mejor": y_mejor,
                "x_peor": x_peor,
                "y_peor": y_peor,
                "seed_mejor": seed_mejor,
                "seed_peor": seed_peor,
            }
        )

    if len(series_por_algoritmo) == 0:
        return False

    fig, axes = plt.subplots(
        len(series_por_algoritmo),
        1,
        figsize=(11, 4.8 * len(series_por_algoritmo)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes[:, 0]

    limites_x = calcular_limites_compartidos(
        [serie["x"] for serie in series_por_algoritmo], padding_ratio=0.02, padding_min=1.0
    )
    limites_y = calcular_limites_compartidos(
        [serie["media"] - serie["desv"] for serie in series_por_algoritmo]
        + [serie["media"] + serie["desv"] for serie in series_por_algoritmo]
        + [serie["y_mejor"] for serie in series_por_algoritmo]
        + [serie["y_peor"] for serie in series_por_algoritmo],
        padding_ratio=0.05,
        padding_min=1e-6,
    )

    for i, (ax, serie) in enumerate(zip(axes, series_por_algoritmo)):
        color = serie["color"]

        ax.step(
            serie["x"],
            serie["media"],
            where="post",
            color=color,
            linewidth=2.2,
            label=f"Media (n={serie['n_curvas']})",
        )
        ax.fill_between(
            serie["x"],
            serie["media"] - serie["desv"],
            serie["media"] + serie["desv"],
            color=color,
            alpha=0.20,
            label="±1σ",
        )
        ax.step(
            serie["x_mejor"],
            serie["y_mejor"],
            where="post",
            color=color,
            linestyle="--",
            linewidth=1.3,
            alpha=0.95,
            label=f"Mejor run (seed={serie['seed_mejor']})",
        )
        ax.step(
            serie["x_peor"],
            serie["y_peor"],
            where="post",
            color=color,
            linestyle=":",
            linewidth=1.6,
            alpha=0.95,
            label=f"Peor run (seed={serie['seed_peor']})",
        )

        ax.set_title(serie["nombre"])
        if i == len(series_por_algoritmo) - 1:
            ax.set_xlabel("Evaluaciones")
        ax.set_ylabel("Mejor fitness hasta ahora")
        ax.grid(alpha=0.25)
        aplicar_formato_ejes_compacto(ax, ejes="both")
        aplicar_limites_compartidos(ax, "x", limites_x)
        aplicar_limites_compartidos(ax, "y", limites_y)
        ax.legend(loc="best", fontsize=8, ncols=2)

    fig.suptitle(titulo)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.92, wspace=0.28, hspace=0.34)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_curva_diversidad(filas, problema, titulo, outpath):
    metricas = DIVERSIDAD_POR_PROBLEMA.get(problema, ())
    if len(metricas) == 0:
        return False

    filas_por_algoritmo = {}
    for fila in filas:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    orden_algoritmos = ordenar_algoritmos(list(filas_por_algoritmo.keys()))
    if len(orden_algoritmos) == 0:
        return False

    hay_grafica = False
    datos_por_algoritmo = {}
    limites_x_por_algoritmo = {}
    limites_y_por_algoritmo = {}

    for algoritmo in orden_algoritmos:
        paneles = []
        series_x_fila = []
        series_y_fila = []
        for metrica, etiqueta_y in metricas:
            curvas = []
            semillas = []
            fitness_final = []

            for fila in filas_por_algoritmo[algoritmo]:
                curva = cargar_curva_metrica_desde_logbook(fila.get("ruta_metricas", ""), metrica)
                if curva is None:
                    continue
                curvas.append(curva)
                semillas.append(fila.get("semilla"))
                fitness_final.append(float(fila.get("fitness", np.nan)))

            if len(curvas) == 0:
                paneles.append(None)
                continue

            x, y_runs = alinear_curvas_por_evaluaciones(curvas)
            media = np.mean(y_runs, axis=0)
            desv = np.std(y_runs, axis=0)

            fitness_arr = np.asarray(fitness_final, dtype=float)
            if np.any(np.isfinite(fitness_arr)):
                idx_mejor = int(np.nanargmin(fitness_arr))
                idx_peor = int(np.nanargmax(fitness_arr))
            else:
                idx_mejor = 0
                idx_peor = len(curvas) - 1

            x_mejor, y_mejor = curvas[idx_mejor]
            x_peor, y_peor = curvas[idx_peor]
            seed_mejor = semillas[idx_mejor]
            seed_peor = semillas[idx_peor]

            color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")
            nombre = algoritmo.upper()
            x_colapso = detectar_punto_colapso_por_pendiente(x, media)

            paneles.append(
                {
                    "metrica": metrica,
                    "etiqueta_y": etiqueta_y,
                    "color": color,
                    "nombre": nombre,
                    "n_curvas": len(curvas),
                    "x": x,
                    "media": media,
                    "desv": desv,
                    "x_mejor": x_mejor,
                    "y_mejor": y_mejor,
                    "x_peor": x_peor,
                    "y_peor": y_peor,
                    "seed_mejor": seed_mejor,
                    "seed_peor": seed_peor,
                    "x_colapso": x_colapso,
                }
            )
            series_x_fila.append(x)
            series_y_fila.extend([media - desv, media + desv, y_mejor, y_peor])
            hay_grafica = True

        datos_por_algoritmo[algoritmo] = paneles
        limites_x_por_algoritmo[algoritmo] = calcular_limites_compartidos(
            series_x_fila, padding_ratio=0.02, padding_min=1.0
        )
        if all(panel is None or panel["metrica"] in METRICAS_NORMALIZADAS for panel in paneles):
            limites_y_por_algoritmo[algoritmo] = (-0.05, 1.05)
        else:
            limites_y_por_algoritmo[algoritmo] = calcular_limites_compartidos(
                series_y_fila, padding_ratio=0.05, padding_min=1e-6
            )

    if not hay_grafica:
        return False

    fig, axes = plt.subplots(
        len(orden_algoritmos),
        len(metricas),
        figsize=(6.8 * len(metricas), 4.8 * len(orden_algoritmos)),
        squeeze=False,
    )

    for i, algoritmo in enumerate(orden_algoritmos):
        paneles = datos_por_algoritmo[algoritmo]
        for j, panel in enumerate(paneles):
            ax = axes[i, j]
            if panel is None:
                ax.set_visible(False)
                continue

            color = panel["color"]
            n_transiciones = 0

            ax.step(
                panel["x"],
                panel["media"],
                where="post",
                color=color,
                linewidth=2.2,
                label=f"Media (n={panel['n_curvas']})",
            )
            ax.fill_between(
                panel["x"],
                panel["media"] - panel["desv"],
                panel["media"] + panel["desv"],
                color=color,
                alpha=0.20,
                label="±1σ",
            )
            ax.step(
                panel["x_mejor"],
                panel["y_mejor"],
                where="post",
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.95,
                label=f"Mejor run (seed={panel['seed_mejor']})",
            )
            ax.step(
                panel["x_peor"],
                panel["y_peor"],
                where="post",
                color=color,
                linestyle=":",
                linewidth=1.6,
                alpha=0.95,
                label=f"Peor run (seed={panel['seed_peor']})",
            )

            if panel["x_colapso"] is not None:
                ax.axvspan(panel["x"][0], panel["x_colapso"], color=color, alpha=0.05)
                ax.axvspan(panel["x_colapso"], panel["x"][-1], color=color, alpha=0.018)
                ax.axvline(
                    panel["x_colapso"],
                    color=color,
                    linestyle="-.",
                    linewidth=1.4,
                    alpha=0.9,
                    label="Transición exploración–explotación",
                )
                n_transiciones = 1

            ax.set_title(f"{panel['nombre']} | {panel['etiqueta_y']}")
            if i == len(orden_algoritmos) - 1:
                ax.set_xlabel("Evaluaciones")
            ax.set_ylabel("Diversidad")
            ax.grid(alpha=0.25)
            aplicar_formato_ejes_compacto(ax, ejes="both")
            aplicar_limites_compartidos(ax, "x", limites_x_por_algoritmo[algoritmo])
            aplicar_limites_compartidos(ax, "y", limites_y_por_algoritmo[algoritmo])

            if n_transiciones > 0:
                ax.text(
                    0.01,
                    0.98,
                    "Fases por pendiente: izquierda=exploración, derecha=explotación",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#444444",
                    bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 2.0},
                )

            ax.legend(loc="best", fontsize=8, ncols=2)

    fig.suptitle(titulo)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.92, wspace=0.28, hspace=0.34)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_curva_conjunta_fitness_diversidad(filas, problema, titulo, outpath):
    metricas = DIVERSIDAD_POR_PROBLEMA.get(problema, ())
    if len(metricas) == 0:
        return False

    filas_por_algoritmo = {}
    for fila in filas:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    orden_algoritmos = ordenar_algoritmos(list(filas_por_algoritmo.keys()))
    if len(orden_algoritmos) == 0:
        return False

    hay_grafica = False
    paneles_por_algoritmo = {}
    limites_x_por_algoritmo = {}
    limites_fit_y_por_algoritmo = {}
    limites_div_y_por_algoritmo = {}

    color_fitness = "#1f77b4"
    color_fitness_promedio = "#2ca02c"
    color_diversidad = "#ff7f0e"

    for algoritmo in orden_algoritmos:
        paneles = []
        series_x_fila = []
        series_fit_fila = []
        series_div_fila = []
        for metrica, etiqueta_div in metricas:
            curvas_fitness = []
            curvas_fitness_promedio = []
            curvas_diversidad = []
            for fila in filas_por_algoritmo[algoritmo]:
                curva_fitness = cargar_curva_desde_logbook(fila.get("ruta_metricas", ""))
                curva_fitness_promedio = cargar_curva_promedio_desde_logbook(fila.get("ruta_metricas", ""))
                curva_diversidad = cargar_curva_metrica_desde_logbook(fila.get("ruta_metricas", ""), metrica)
                if curva_fitness is None or curva_fitness_promedio is None or curva_diversidad is None:
                    continue
                curvas_fitness.append(curva_fitness)
                curvas_fitness_promedio.append(curva_fitness_promedio)
                curvas_diversidad.append(curva_diversidad)

            if len(curvas_fitness) == 0:
                paneles.append(None)
                continue

            x_fit, y_fit_runs = alinear_curvas_por_evaluaciones(curvas_fitness)
            x_fit_promedio, y_fit_promedio_runs = alinear_curvas_por_evaluaciones(curvas_fitness_promedio)
            x_div, y_div_runs = alinear_curvas_por_evaluaciones(curvas_diversidad)
            media_fit = np.mean(y_fit_runs, axis=0)
            media_fit_promedio = np.mean(y_fit_promedio_runs, axis=0)
            media_div = np.mean(y_div_runs, axis=0)

            x = np.unique(np.concatenate([x_fit, x_fit_promedio, x_div]))
            y_fit = alinear_serie_a_malla(x_fit, media_fit, x)
            y_fit_promedio = alinear_serie_a_malla(x_fit_promedio, media_fit_promedio, x)
            y_div = alinear_serie_a_malla(x_div, media_div, x)
            paneles.append(
                {
                    "metrica": metrica,
                    "etiqueta_div": etiqueta_div,
                    "nombre_algoritmo": algoritmo.upper(),
                    "x": x,
                    "y_fit": y_fit,
                    "y_fit_promedio": y_fit_promedio,
                    "y_div": y_div,
                }
            )
            series_x_fila.append(x)
            series_fit_fila.extend([y_fit, y_fit_promedio])
            series_div_fila.append(y_div)
            hay_grafica = True

        paneles_por_algoritmo[algoritmo] = paneles
        limites_x_por_algoritmo[algoritmo] = calcular_limites_compartidos(
            series_x_fila, padding_ratio=0.02, padding_min=1.0
        )
        limites_fit_y_por_algoritmo[algoritmo] = calcular_limites_compartidos(
            series_fit_fila, padding_ratio=0.05, padding_min=1e-6
        )
        if all(panel is None or panel["metrica"] in METRICAS_NORMALIZADAS for panel in paneles):
            limites_div_y_por_algoritmo[algoritmo] = (-0.05, 1.05)
        else:
            limites_div_y_por_algoritmo[algoritmo] = calcular_limites_compartidos(
                series_div_fila, padding_ratio=0.05, padding_min=1e-6
            )

    if not hay_grafica:
        return False

    fig, axes = plt.subplots(
        len(orden_algoritmos),
        len(metricas),
        figsize=(6.8 * len(metricas), 4.8 * len(orden_algoritmos)),
        squeeze=False,
    )

    for i, algoritmo in enumerate(orden_algoritmos):
        for j, panel in enumerate(paneles_por_algoritmo[algoritmo]):
            ax = axes[i, j]
            if panel is None:
                ax.set_visible(False)
                continue

            linea_fit = ax.step(
                panel["x"],
                panel["y_fit"],
                where="post",
                color=color_fitness,
                linewidth=2.1,
                label="Mejor fitness",
            )[0]
            linea_fit_promedio = ax.step(
                panel["x"],
                panel["y_fit_promedio"],
                where="post",
                color=color_fitness_promedio,
                linewidth=1.9,
                linestyle="--",
                label="Fitness promedio",
            )[0]

            ax2 = ax.twinx()
            linea_div = ax2.step(
                panel["x"],
                panel["y_div"],
                where="post",
                color=color_diversidad,
                linewidth=2.1,
                label="Diversidad",
            )[0]

            ax.set_title(f"{panel['nombre_algoritmo']} | {panel['etiqueta_div']}")
            if i == len(orden_algoritmos) - 1:
                ax.set_xlabel("Evaluaciones")
            ax.set_ylabel("Fitness", color=color_fitness)
            ax.tick_params(axis="y", labelcolor=color_fitness)
            aplicar_formato_ejes_compacto(ax, ejes="both")
            ax.grid(alpha=0.25)
            aplicar_limites_compartidos(ax, "x", limites_x_por_algoritmo[algoritmo])
            aplicar_limites_compartidos(ax, "y", limites_fit_y_por_algoritmo[algoritmo])

            ax2.set_ylabel("Diversidad", color=color_diversidad)
            ax2.tick_params(axis="y", labelcolor=color_diversidad)
            aplicar_limites_compartidos(ax2, "y", limites_div_y_por_algoritmo[algoritmo])

            ax.legend(
                [linea_fit, linea_fit_promedio, linea_div],
                ["Mejor fitness", "Fitness promedio", "Diversidad"],
                loc="best",
                fontsize=8,
            )

    fig.suptitle(titulo)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_predictor_diversidad_mejora_futura(filas, problema, titulo, outpath, delta_futuro):
    datos_predictor = preparar_predictor_diversidad_mejora_futura(filas, problema, delta_futuro)
    if datos_predictor is None:
        return False

    metricas = datos_predictor["metricas"]
    orden_algoritmos = datos_predictor["orden_algoritmos"]
    paneles_por_algoritmo = datos_predictor["paneles_por_algoritmo"]
    limites_x_por_algoritmo = datos_predictor["limites_x_por_algoritmo"]
    limites_y_por_algoritmo = datos_predictor["limites_y_por_algoritmo"]

    fig, axes = plt.subplots(
        len(orden_algoritmos),
        len(metricas),
        figsize=(7.0 * len(metricas), 4.8 * len(orden_algoritmos)),
        squeeze=False,
    )

    for i, algoritmo in enumerate(orden_algoritmos):
        for j, panel in enumerate(paneles_por_algoritmo[algoritmo]):
            ax = axes[i, j]
            if panel is None:
                ax.set_visible(False)
                continue

            scatter = ax.scatter(
                panel["x"],
                panel["y"],
                s=12,
                alpha=0.23,
                color=panel["color"],
                edgecolors="none",
                label="Muestras (t)",
            )

            handles = [scatter]
            labels = [scatter.get_label()]
            if panel["resumen"] is not None:
                centros, mejora_media, _, _ = panel["resumen"]
                linea_mejora = ax.plot(
                    centros,
                    mejora_media,
                    color="#111111",
                    linewidth=2.6,
                    label="Mejora media futura",
                )[0]
                handles.append(linea_mejora)
                labels.append(linea_mejora.get_label())

            ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
            ax.set_title(f"{panel['nombre_algoritmo']} | {panel['etiqueta_div']} (delta={delta_futuro})")
            if i == len(orden_algoritmos) - 1:
                ax.set_xlabel("Diversidad en t")
            ax.set_ylabel(f"Mejora fitness en t+{delta_futuro}")
            ax.grid(alpha=0.25)
            aplicar_formato_ejes_compacto(ax, ejes="both")
            aplicar_limites_compartidos(ax, "x", limites_x_por_algoritmo[algoritmo])
            aplicar_limites_compartidos(ax, "y", limites_y_por_algoritmo[algoritmo])

            ax.legend(handles, labels, loc="best", fontsize=8)

    fig.suptitle(titulo)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_predictor_diversidad_mejora_futura_histograma(filas, problema, titulo, outpath, delta_futuro):
    datos_predictor = preparar_predictor_diversidad_mejora_futura(filas, problema, delta_futuro)
    if datos_predictor is None:
        return False

    metricas = datos_predictor["metricas"]
    orden_algoritmos = datos_predictor["orden_algoritmos"]
    paneles_por_algoritmo = datos_predictor["paneles_por_algoritmo"]
    limites_x_por_algoritmo = datos_predictor["limites_x_por_algoritmo"]
    limites_y_por_algoritmo = datos_predictor["limites_y_por_algoritmo"]

    n_bins_hist = 12
    limite_freq_por_algoritmo = {}
    for algoritmo in orden_algoritmos:
        max_conteo = 0
        limites_x = limites_x_por_algoritmo[algoritmo]
        for panel in paneles_por_algoritmo[algoritmo]:
            if panel is None:
                continue
            histograma = calcular_histograma_instancias(panel["x"], n_bins=n_bins_hist, limites=limites_x)
            if histograma is None:
                continue
            _, conteos, _ = histograma
            if conteos.size > 0:
                max_conteo = max(max_conteo, int(np.max(conteos)))
        limite_freq_por_algoritmo[algoritmo] = (0.0, max_conteo + max(max_conteo * 0.10, 1.0))

    fig, axes = plt.subplots(
        len(orden_algoritmos),
        len(metricas),
        figsize=(7.2 * len(metricas), 4.9 * len(orden_algoritmos)),
        squeeze=False,
    )

    for i, algoritmo in enumerate(orden_algoritmos):
        limites_x = limites_x_por_algoritmo[algoritmo]
        limites_y = limites_y_por_algoritmo[algoritmo]

        for j, panel in enumerate(paneles_por_algoritmo[algoritmo]):
            ax = axes[i, j]
            if panel is None:
                ax.set_visible(False)
                continue

            ax_freq = ax.twinx()
            histograma = calcular_histograma_instancias(panel["x"], n_bins=n_bins_hist, limites=limites_x)
            handles = []
            labels = []

            if histograma is not None:
                centros_hist, conteos_hist, anchos_hist = histograma
                barras = ax_freq.bar(
                    centros_hist,
                    conteos_hist,
                    width=anchos_hist * 0.88,
                    color=panel["color"],
                    alpha=0.35,
                    edgecolor="white",
                    linewidth=0.8,
                    label="Frecuencia de instancias",
                    zorder=1,
                )
                if len(barras) > 0:
                    handles.append(barras[0])
                    labels.append("Frecuencia de instancias")

            ax_freq.set_ylabel("Frecuencia", color=panel["color"])
            ax_freq.tick_params(axis="y", labelcolor=panel["color"])
            aplicar_limites_compartidos(ax_freq, "y", limite_freq_por_algoritmo[algoritmo])

            if panel["resumen"] is not None:
                centros, mejora_media, _, _ = panel["resumen"]
                linea_mejora = ax.plot(
                    centros,
                    mejora_media,
                    color="#111111",
                    linewidth=2.8,
                    zorder=4,
                    label="Mejora media futura",
                )[0]
                handles.append(linea_mejora)
                labels.append(linea_mejora.get_label())

            ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":", zorder=3)
            ax.set_title(f"{panel['nombre_algoritmo']} | {panel['etiqueta_div']} (delta={delta_futuro})")
            if i == len(orden_algoritmos) - 1:
                ax.set_xlabel("Diversidad en t")
            ax.set_ylabel(f"Mejora fitness en t+{delta_futuro}")
            ax.grid(alpha=0.18)
            aplicar_formato_ejes_compacto(ax, ejes="both")
            aplicar_limites_compartidos(ax, "x", limites_x)
            aplicar_limites_compartidos(ax, "y", limites_y)

            if len(handles) > 0:
                ax.legend(handles, labels, loc="best", fontsize=8)

    fig.suptitle(titulo)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.92, wspace=0.28, hspace=0.34)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_graficos_para_bloque(filas, outdir, algoritmo, problema, sufijo_archivo, delta_futuro):
    fitness_por_adaptacion, tiempo_por_adaptacion, etiquetas = construir_series_por_adaptacion(filas)
    sufijo_titulo = f" (algoritmo={algoritmo}, problema={problema})"

    ruta_fitness = outdir / f"boxplot_fitness{sufijo_archivo}.png"
    ruta_tiempo = outdir / f"boxplot_tiempo{sufijo_archivo}.png"
    ruta_convergencia = outdir / f"curva_convergencia{sufijo_archivo}.png"
    ruta_diversidad = outdir / f"curva_diversidad{sufijo_archivo}.png"
    ruta_conjunta_fit_div = outdir / f"curva_fitness_diversidad{sufijo_archivo}.png"
    ruta_predictor_div = outdir / f"predictor_diversidad_mejora_futura{sufijo_archivo}.png"
    ruta_predictor_div_hist = outdir / f"predictor_diversidad_mejora_futura_histograma{sufijo_archivo}.png"
    ruta_histograma = outdir / f"histograma_fitness_final{sufijo_archivo}.png"

    generar_boxplot(
        datos_por_etiqueta=fitness_por_adaptacion,
        etiquetas=etiquetas,
        titulo="Boxplot de Fitness por Algoritmo/Adaptacion" + sufijo_titulo,
        ylabel="Fitness",
        outpath=ruta_fitness,
    )
    generar_boxplot(
        datos_por_etiqueta=tiempo_por_adaptacion,
        etiquetas=etiquetas,
        titulo="Boxplot de Tiempo por Algoritmo/Adaptacion" + sufijo_titulo,
        ylabel="Tiempo (s)",
        outpath=ruta_tiempo,
    )

    rutas = [ruta_fitness, ruta_tiempo]
    if generar_curva_convergencia(
        filas=filas,
        titulo="Curvas de Convergencia por Algoritmo" + sufijo_titulo,
        outpath=ruta_convergencia,
    ):
        rutas.append(ruta_convergencia)

    if generar_curva_diversidad(
        filas=filas,
        problema=problema,
        titulo="Curvas de Diversidad vs Evaluaciones" + sufijo_titulo,
        outpath=ruta_diversidad,
    ):
        rutas.append(ruta_diversidad)

    if generar_curva_conjunta_fitness_diversidad(
        filas=filas,
        problema=problema,
        titulo="Curvas Conjuntas de Fitness y Diversidad" + sufijo_titulo,
        outpath=ruta_conjunta_fit_div,
    ):
        rutas.append(ruta_conjunta_fit_div)

    if generar_predictor_diversidad_mejora_futura(
        filas=filas,
        problema=problema,
        titulo="Diversidad como Predictor de Mejora Futura" + sufijo_titulo,
        outpath=ruta_predictor_div,
        delta_futuro=delta_futuro,
    ):
        rutas.append(ruta_predictor_div)

    if generar_predictor_diversidad_mejora_futura_histograma(
        filas=filas,
        problema=problema,
        titulo="Diversidad como Predictor de Mejora Futura (Histograma 2D)" + sufijo_titulo,
        outpath=ruta_predictor_div_hist,
        delta_futuro=delta_futuro,
    ):
        rutas.append(ruta_predictor_div_hist)

    if generar_histograma_fitness_final(
        filas=filas,
        titulo="Histograma de Fitness Final por Algoritmo" + sufijo_titulo,
        outpath=ruta_histograma,
    ):
        rutas.append(ruta_histograma)

    return rutas

def main():
    args = parse_args()
    outdir = Path(args.outdir)

    filas = cargar_runs(args.results_csv, args.algoritmo, args.problema)
    if len(filas) == 0:
        raise RuntimeError("No hay filas tras aplicar filtros de algoritmo/problema")
    outdir.mkdir(parents=True, exist_ok=True)

    ficheros_generados = []

    if args.problema == "ambos":
        for problema in ("cec2017", "qap"):
            filas_p = [f for f in filas if f["problema"] == problema]
            if len(filas_p) == 0:
                continue
            ficheros_generados.extend(
                generar_graficos_para_bloque(
                    filas=filas_p,
                    outdir=outdir,
                    algoritmo=args.algoritmo,
                    problema=problema,
                    sufijo_archivo=f"_{problema}",
                    delta_futuro=max(1, args.delta_futuro),
                )
            )
    else:
        ficheros_generados.extend(
            generar_graficos_para_bloque(
                filas=filas,
                outdir=outdir,
                algoritmo=args.algoritmo,
                problema=args.problema,
                sufijo_archivo="",
                delta_futuro=max(1, args.delta_futuro),
            )
        )

    print(f"Graficos generados en: {outdir}")
    for ruta in ficheros_generados:
        print(f"- {ruta}")

if __name__ == "__main__":
    main()
