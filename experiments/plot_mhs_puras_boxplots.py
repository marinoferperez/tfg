#!/usr/bin/env python3
# genera boxplots y curvas de convergencia para AGE/DE en CEC2017 y QAP.

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

# asegura una ruta de cache escribible en entornos restringidos
cache_root = Path(tempfile.gettempdir()) / "plot_mhs_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera boxplots a partir de runs.csv del benchmark de MHs puras"
    )
    parser.add_argument("--results-csv", type=str, default="results/experimentos_mhs_puras/runs.csv")
    parser.add_argument("--outdir", type=str, default="results/experimentos_mhs_puras/plots")
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
    return parser.parse_args()


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


def alinear_curvas_por_evaluaciones(curvas):
    malla = np.unique(np.concatenate([x for x, _ in curvas]))
    matriz = np.empty((len(curvas), len(malla)), dtype=float)

    for i, (x, y) in enumerate(curvas):
        idx = np.searchsorted(x, malla, side="right") - 1
        idx[idx < 0] = 0
        matriz[i, :] = y[idx]

    return malla, matriz


def ordenar_etiquetas(etiquetas):
    prioridad = [
        "age_cec2017",
        "de_cec2017",
        "age_qap",
        "de_qap",
    ]
    presentes = set(etiquetas)
    ordenadas = [e for e in prioridad if e in presentes]
    extras = sorted(presentes.difference(set(ordenadas)))
    return ordenadas + extras


def generar_boxplot(datos_por_etiqueta, etiquetas, titulo, ylabel, outpath):
    if len(etiquetas) == 0:
        raise RuntimeError("No hay etiquetas para graficar")

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
    prioridad = ["age", "de"]
    presentes = set(algoritmos)
    ordenados = [a for a in prioridad if a in presentes]
    extras = sorted(presentes.difference(set(ordenados)))
    return ordenados + extras


def generar_histograma_fitness_final(filas, titulo, outpath):
    fitness_por_algoritmo = {}
    for fila in filas:
        fitness_por_algoritmo.setdefault(fila["algoritmo"], []).append(float(fila["fitness"]))

    algoritmos = ordenar_algoritmos(list(fitness_por_algoritmo.keys()))
    if len(algoritmos) == 0:
        return False

    colores = {
        "age": "#4e79a7",
        "de": "#f28e2b",
    }

    ncols = len(algoritmos)
    fig, axes = plt.subplots(1, ncols, figsize=(8.5 * ncols, 6), squeeze=False)
    axes = axes[0]

    for i, algoritmo in enumerate(algoritmos):
        ax = axes[i]
        datos = np.asarray(fitness_por_algoritmo[algoritmo], dtype=float)
        if len(datos) == 0:
            continue

        color = colores.get(algoritmo, "#59a14f")

        # Para evitar histogramas "colapsados", se fuerza un rango con padding
        # y un numero minimo de bins aunque la dispersion sea pequena.
        data_min = float(np.min(datos))
        data_max = float(np.max(datos))
        media = float(np.mean(datos))
        mediana = float(np.median(datos))

        n_bins = max(12, int(np.sqrt(len(datos)) * 2))
        if data_max == data_min:
            padding = max(abs(data_min) * 0.002, 1.0)
        else:
            spread = data_max - data_min
            padding = max(spread * 0.25, abs(media) * 0.0005, 1.0)

        left = data_min - padding
        right = data_max + padding
        bin_edges = np.linspace(left, right, num=n_bins + 1)

        ax.hist(datos, bins=bin_edges, color=color, edgecolor="white", alpha=0.78)

        ax.axvline(media, color="#222222", linestyle="--", linewidth=1.2, label="media")
        ax.axvline(mediana, color="#222222", linestyle=":", linewidth=1.2, label="mediana")

        ax.set_title(f"{algoritmo.upper()} (n={len(datos)})")
        ax.set_xlabel("Fitness final")
        if i == 0:
            ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
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

    orden_algoritmos = [a for a in ("age", "de") if a in filas_por_algoritmo]
    orden_algoritmos += sorted(set(filas_por_algoritmo.keys()) - set(orden_algoritmos))
    colores = {
        "age": "#4e79a7",
        "de": "#f28e2b",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    se_grafo_algo = False

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

        color = colores.get(algoritmo, "#59a14f")
        nombre = algoritmo.upper()

        ax.step(x, media, where="post", color=color, linewidth=2.2, label=f"{nombre} media (n={len(curvas)})")
        ax.fill_between(x, media - desv, media + desv, color=color, alpha=0.20, label=f"{nombre} ±1σ")
        ax.step(
            x_mejor,
            y_mejor,
            where="post",
            color=color,
            linestyle="--",
            linewidth=1.3,
            alpha=0.95,
            label=f"{nombre} mejor run (seed={seed_mejor})",
        )
        ax.step(
            x_peor,
            y_peor,
            where="post",
            color=color,
            linestyle=":",
            linewidth=1.6,
            alpha=0.95,
            label=f"{nombre} peor run (seed={seed_peor})",
        )
        se_grafo_algo = True

    if not se_grafo_algo:
        plt.close(fig)
        return False

    ax.set_title(titulo)
    ax.set_xlabel("Evaluaciones")
    ax.set_ylabel("Mejor fitness hasta ahora")
    ax.grid(alpha=0.25)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    filas = cargar_runs(args.results_csv, args.algoritmo, args.problema)
    if len(filas) == 0:
        raise RuntimeError("No hay filas tras aplicar filtros de algoritmo/problema")

    ficheros_generados = []

    # si se pide ambos problemas, se generan plots separados por problema
    if args.problema == "ambos":
        for problema in ("cec2017", "qap"):
            filas_p = [f for f in filas if f["problema"] == problema]
            if len(filas_p) == 0:
                continue

            fitness_por_adaptacion, tiempo_por_adaptacion, etiquetas = construir_series_por_adaptacion(filas_p)
            sufijo = f" (algoritmo={args.algoritmo}, problema={problema})"

            ruta_fitness = outdir / f"boxplot_fitness_{problema}.png"
            ruta_tiempo = outdir / f"boxplot_tiempo_{problema}.png"
            ruta_convergencia = outdir / f"curva_convergencia_{problema}.png"
            ruta_histograma = outdir / f"histograma_fitness_final_{problema}.png"

            generar_boxplot(
                datos_por_etiqueta=fitness_por_adaptacion,
                etiquetas=etiquetas,
                titulo="Boxplot de Fitness por Algoritmo/Adaptacion" + sufijo,
                ylabel="Fitness",
                outpath=ruta_fitness,
            )
            generar_boxplot(
                datos_por_etiqueta=tiempo_por_adaptacion,
                etiquetas=etiquetas,
                titulo="Boxplot de Tiempo por Algoritmo/Adaptacion" + sufijo,
                ylabel="Tiempo (s)",
                outpath=ruta_tiempo,
            )
            ficheros_generados.extend([ruta_fitness, ruta_tiempo])

            if generar_curva_convergencia(
                filas=filas_p,
                titulo="Curvas de Convergencia por Algoritmo" + sufijo,
                outpath=ruta_convergencia,
            ):
                ficheros_generados.append(ruta_convergencia)
            else:
                print(f"[aviso] Sin datos suficientes para convergencia en problema={problema}")

            if generar_histograma_fitness_final(
                filas=filas_p,
                titulo="Histograma de Fitness Final por Algoritmo" + sufijo,
                outpath=ruta_histograma,
            ):
                ficheros_generados.append(ruta_histograma)
            else:
                print(f"[aviso] Sin datos suficientes para histograma en problema={problema}")
    else:
        fitness_por_adaptacion, tiempo_por_adaptacion, etiquetas = construir_series_por_adaptacion(filas)
        sufijo = f" (algoritmo={args.algoritmo}, problema={args.problema})"

        ruta_fitness = outdir / "boxplot_fitness.png"
        ruta_tiempo = outdir / "boxplot_tiempo.png"
        ruta_convergencia = outdir / "curva_convergencia.png"
        ruta_histograma = outdir / "histograma_fitness_final.png"

        generar_boxplot(
            datos_por_etiqueta=fitness_por_adaptacion,
            etiquetas=etiquetas,
            titulo="Boxplot de Fitness por Algoritmo/Adaptacion" + sufijo,
            ylabel="Fitness",
            outpath=ruta_fitness,
        )
        generar_boxplot(
            datos_por_etiqueta=tiempo_por_adaptacion,
            etiquetas=etiquetas,
            titulo="Boxplot de Tiempo por Algoritmo/Adaptacion" + sufijo,
            ylabel="Tiempo (s)",
            outpath=ruta_tiempo,
        )
        ficheros_generados.extend([ruta_fitness, ruta_tiempo])

        if generar_curva_convergencia(
            filas=filas,
            titulo="Curvas de Convergencia por Algoritmo" + sufijo,
            outpath=ruta_convergencia,
        ):
            ficheros_generados.append(ruta_convergencia)
        else:
            print(f"[aviso] Sin datos suficientes para convergencia en problema={args.problema}")

        if generar_histograma_fitness_final(
            filas=filas,
            titulo="Histograma de Fitness Final por Algoritmo" + sufijo,
            outpath=ruta_histograma,
        ):
            ficheros_generados.append(ruta_histograma)
        else:
            print(f"[aviso] Sin datos suficientes para histograma en problema={args.problema}")

    print(f"Graficos generados en: {outdir}")
    for ruta in ficheros_generados:
        print(f"- {ruta}")


if __name__ == "__main__":
    main()
