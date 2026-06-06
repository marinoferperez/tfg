from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.utils import concatenar_runs
from preprocesado_de_datos.utils.path_utils import asegurar_directorio, escribir_csv_dicts, escribir_json
from preprocesado_de_datos.utils.fitness_utils import (
    asignar_bins_fitness,
    balanceo_a_la_baja,
    contabilizar_muestras_por_bin,
    construir_fases_relativas_por_seed,
)

TIPOS_BINS = ("uniformes", "cuantiles")
COLORES = {
    "original": "#4e79a7",
    "uniformes": "#f28e2b",
    "cuantiles": "#59a14f",
}
FASES_LABELS = ("0-20", "20-40", "40-60", "60-80", "80-100")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compara visual y numericamente bins uniformes vs cuantiles para decidir "
            "que discretizacion de fitness conviene antes del preprocesado."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Lista de rutas a dataset_*.npz de un mismo bloque (funcion+algoritmo o instancia+algoritmo).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directorio de salida. Si no se indica, se infiere una ruta benchmarking/offline/dataset_preprocesado/.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Numero de bins configurados para comparar. Default: 10.",
    )
    parser.add_argument(
        "--max-por-bin",
        type=int,
        default=3000,
        help="Tope maximo de muestras por bin a simular. Default: 3000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla del submuestreo aleatorio. Default: 42.",
    )
    parser.add_argument(
        "--estratificar-por-fase",
        action="store_true",
        help="Aplica tambien estratificacion temporal por fase al comparar ambos tipos de bins.",
    )
    return parser.parse_args()


def inferir_slug_experimento(rutas_npz):
    rutas = [Path(r) for r in rutas_npz]
    if len(rutas) == 0:
        return "experimento_desconocido"

    primera = rutas[0]
    partes = primera.parts

    nombre_experimento = None
    for parte in partes:
        if "experimentos" in str(parte):
            nombre_experimento = parte
            break
    if nombre_experimento is None:
        for idx, parte in enumerate(partes):
            if parte == "metricas_runs" and idx >= 1:
                nombre_experimento = partes[idx - 1]
                break
    if nombre_experimento is None:
        nombre_experimento = primera.parent.name

    run_dir = primera.parent.name
    run_tokens = run_dir.split("_s", 1)[0]
    match_tam = re.search(r"(?:^|_)tam_(\d+)(?:_|$)", nombre_experimento)
    if match_tam is not None:
        return f"experimentos_tam_{match_tam.group(1)}_{run_tokens}"
    return f"experimentos_{run_tokens}"


def inferir_contexto_inputs(rutas_npz):
    primera = Path(rutas_npz[0]).resolve()
    algoritmo = None
    bloque = None
    metricas_runs_dir = None

    for parent in primera.parents:
        if parent.name.lower() in {"age", "de"} and algoritmo is None:
            algoritmo = parent.name.lower()
        if re.fullmatch(r"f\d+", parent.name):
            bloque = parent.name
        if parent.name == "metricas_runs":
            metricas_runs_dir = parent
            break

    if bloque is None and metricas_runs_dir is not None:
        bloque = metricas_runs_dir.parent.name

    return {
        "primera": primera,
        "algoritmo": algoritmo or "comparacion",
        "bloque": bloque or "bloque_desconocido",
        "metricas_runs_dir": metricas_runs_dir,
    }


def inferir_outdir(rutas_npz, outdir):
    if outdir is not None:
        return asegurar_directorio(outdir)

    contexto = inferir_contexto_inputs(rutas_npz)
    metricas_runs_dir = contexto["metricas_runs_dir"]
    bloque = contexto["bloque"]

    if metricas_runs_dir is not None:
        if re.fullmatch(r"f\d+", bloque):
            experimento_dir = metricas_runs_dir.parent.parent
            if experimento_dir.name == "metaheuristica_resultados":
                experimento_dir = experimento_dir.parent
        else:
            experimento_dir = metricas_runs_dir.parent
            if experimento_dir.name == "metaheuristica_resultados":
                experimento_dir = experimento_dir.parent
        return asegurar_directorio(
            experimento_dir
            / "benchmarking"
            / "offline"
            / "dataset_preprocesado"
            / "seleccion_muestras"
            / bloque
            / "bins_uniformes_vs_cuantiles"
        )

    slug = inferir_slug_experimento(rutas_npz)
    return asegurar_directorio(Path("preprocesado_de_datos/comparacion_tipo_bins") / slug)


def estilo_ejes(ax):
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def formatear_tipo_bins(tipo_bins):
    return "Uniformes" if tipo_bins == "uniformes" else "Cuantiles"


def conteo_categorias(arr):
    valores, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(valores, counts)}


def vector_retencion_por_categoria(dict_antes, dict_despues, categorias):
    antes = np.asarray([int(dict_antes.get(cat, 0)) for cat in categorias], dtype=float)
    despues = np.asarray([int(dict_despues.get(cat, 0)) for cat in categorias], dtype=float)
    ret = np.divide(
        despues,
        np.maximum(antes, 1.0),
        out=np.zeros_like(despues, dtype=float),
        where=antes > 0,
    ) * 100.0
    return antes, despues, ret


def coeficiente_variacion(valores):
    valores = np.asarray(valores, dtype=float)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        return 0.0
    media = float(np.mean(valores))
    if np.isclose(media, 0.0):
        return 0.0
    return float(np.std(valores) / media)


def ancho_robusto(valor_min, valor_max, fallback):
    ancho = float(valor_max - valor_min)
    if np.isfinite(ancho) and ancho > 1e-12:
        return ancho
    return max(float(fallback), 1e-12)


def transformar_fitness_para_comparacion(fitness):
    fitness = np.asarray(fitness, dtype=float)
    minimo = float(np.min(fitness))
    shift = 0.0
    if minimo <= -1.0:
        shift = -minimo + 1.0
    return np.log1p(fitness + shift)


def preparar_dataset(rutas_npz, estratificar_por_fase):
    dataset = concatenar_runs(rutas_npz)
    fases = None
    if estratificar_por_fase:
        if "eval_id" not in dataset or "seed" not in dataset:
            raise ValueError("Para estratificar por fase se requieren las claves 'eval_id' y 'seed'.")
        fases = construir_fases_relativas_por_seed(dataset["eval_id"], dataset["seed"])
    elif "eval_id" in dataset and "seed" in dataset:
        fases = construir_fases_relativas_por_seed(dataset["eval_id"], dataset["seed"])
    return dataset, fases


def evaluar_tipo_bins(dataset, fases, *, tipo_bins, n_bins, max_por_bin, random_state, estratificar_por_fase):
    fitness = np.asarray(dataset["fitness"], dtype=float)
    seeds = np.asarray(dataset["seed"], dtype=np.int32)

    bin_ids_pre, edges = asignar_bins_fitness(fitness, n_bins, tipo_bins=tipo_bins)
    conteo_pre = contabilizar_muestras_por_bin(bin_ids_pre, n_bins)

    indices_sel, bin_ids_post, edges_post = balanceo_a_la_baja(
        fitness,
        seeds,
        n_bins=n_bins,
        max_por_bin=max_por_bin,
        tipo_bins=tipo_bins,
        random_state=random_state,
        fase_arr=fases,
        estratificar_por_fase=estratificar_por_fase,
    )

    indices_sel = np.asarray(indices_sel, dtype=np.int64)
    fitness_bal = fitness[indices_sel]
    conteo_post = contabilizar_muestras_por_bin(bin_ids_post[indices_sel], n_bins)

    fitness_cmp = transformar_fitness_para_comparacion(fitness)
    fitness_bal_cmp = transformar_fitness_para_comparacion(fitness_bal)
    quantiles_ref = np.percentile(fitness_cmp, [1, 5, 25, 50, 75, 95, 99])
    quantiles_bal = np.percentile(fitness_bal_cmp, [1, 5, 25, 50, 75, 95, 99])
    rango_global = max(float(np.max(fitness_cmp) - np.min(fitness_cmp)), 1e-12)
    span_ref = ancho_robusto(quantiles_ref[1], quantiles_ref[-2], fallback=rango_global)
    low_ref, high_ref = float(quantiles_ref[1]), float(quantiles_ref[-2])
    low_bal, high_bal = float(quantiles_bal[1]), float(quantiles_bal[-2])
    overlap = max(0.0, min(high_ref, high_bal) - max(low_ref, low_bal))
    cobertura_intervalo_central = float(overlap / span_ref)
    desplazamiento_cuantiles_centrales = float(
        np.mean(np.abs(quantiles_ref[1:-1] - quantiles_bal[1:-1])) / span_ref
    )
    desplazamiento_mediana = float(np.abs(quantiles_ref[3] - quantiles_bal[3]) / span_ref)
    expansion_intervalo_central = float(max(high_bal - low_bal, 0.0) / span_ref)

    counts_seeds_antes = conteo_categorias(seeds)
    counts_seeds_despues = conteo_categorias(seeds[indices_sel])
    semillas = sorted(set(counts_seeds_antes) | set(counts_seeds_despues))
    _, _, retencion_seeds = vector_retencion_por_categoria(counts_seeds_antes, counts_seeds_despues, semillas)

    phases_payload = None
    phase_retention_cv = None
    if fases is not None:
        counts_fases_antes = conteo_categorias(fases)
        counts_fases_despues = conteo_categorias(fases[indices_sel])
        fases_ids = sorted(set(counts_fases_antes) | set(counts_fases_despues))
        _, _, retencion_fases = vector_retencion_por_categoria(counts_fases_antes, counts_fases_despues, fases_ids)
        phase_retention_cv = coeficiente_variacion(retencion_fases)
        phases_payload = {
            "categorias": [int(v) for v in fases_ids],
            "antes": [int(counts_fases_antes.get(v, 0)) for v in fases_ids],
            "despues": [int(counts_fases_despues.get(v, 0)) for v in fases_ids],
            "retencion_pct": [float(v) for v in retencion_fases],
        }

    shares_post = conteo_post / max(int(np.sum(conteo_post)), 1)
    metrics = {
        "tipo_bins": tipo_bins,
        "n_bins_configurados": int(n_bins),
        "n_bins_reales": int(max(len(edges_post) - 1, 1)),
        "n_original": int(len(fitness)),
        "n_balanceado": int(len(indices_sel)),
        "retencion_pct": float(len(indices_sel) / max(len(fitness), 1) * 100.0),
        "cobertura_intervalo_central_ratio": cobertura_intervalo_central,
        "expansion_intervalo_central_ratio": expansion_intervalo_central,
        "desplazamiento_cuantiles_centrales_norm": desplazamiento_cuantiles_centrales,
        "desplazamiento_mediana_norm": desplazamiento_mediana,
        "max_bin_share_despues": float(np.max(shares_post)),
        "top3_bins_share_despues": float(np.sum(np.sort(shares_post)[-3:])),
        "occupied_bins_ratio_despues": float(np.mean(conteo_post > 0)),
        "seed_retencion_cv": float(coeficiente_variacion(retencion_seeds)),
        "fase_retencion_cv": None if phase_retention_cv is None else float(phase_retention_cv),
    }

    return {
        "tipo_bins": tipo_bins,
        "edges": [float(v) for v in edges_post.tolist()],
        "anchuras_bins": [float(v) for v in np.diff(edges_post)],
        "conteo_bins_antes": [int(v) for v in conteo_pre.tolist()],
        "conteo_bins_despues": [int(v) for v in conteo_post.tolist()],
        "seeds": {
            "categorias": [int(v) for v in semillas],
            "antes": [int(counts_seeds_antes.get(v, 0)) for v in semillas],
            "despues": [int(counts_seeds_despues.get(v, 0)) for v in semillas],
            "retencion_pct": [float(v) for v in retencion_seeds],
        },
        "fases": phases_payload,
        "fitness_balanceado": fitness_bal,
        "metricas": metrics,
    }


def normalizar_scores(resultados):
    criterios = [
        ("cobertura_intervalo_central_ratio", True, 0.30),
        ("desplazamiento_cuantiles_centrales_norm", False, 0.25),
        ("max_bin_share_despues", False, 0.20),
        ("occupied_bins_ratio_despues", True, 0.15),
        ("seed_retencion_cv", False, 0.10),
    ]

    scores = {tipo: 0.0 for tipo in resultados}
    desglose = {tipo: {} for tipo in resultados}

    for clave, mayor_es_mejor, peso in criterios:
        valores = {tipo: float(payload["metricas"][clave]) for tipo, payload in resultados.items()}
        minimo = min(valores.values())
        maximo = max(valores.values())

        if np.isclose(minimo, maximo):
            parciales = {tipo: 0.5 for tipo in resultados}
        else:
            if mayor_es_mejor:
                parciales = {tipo: (valor - minimo) / (maximo - minimo) for tipo, valor in valores.items()}
            else:
                parciales = {tipo: (maximo - valor) / (maximo - minimo) for tipo, valor in valores.items()}

        for tipo, parcial in parciales.items():
            parcial = float(parcial)
            scores[tipo] += peso * parcial
            desglose[tipo][clave] = parcial

    mejor_tipo = max(scores, key=scores.get)
    ordenados = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    margen = float(ordenados[0][1] - ordenados[1][1]) if len(ordenados) > 1 else 1.0
    comentario = (
        "La diferencia es pequena; se puede considerar un empate tecnico."
        if margen < 0.05
        else f"La recomendacion favorece {mejor_tipo} de forma apreciable."
    )
    return mejor_tipo, scores, desglose, comentario


def plot_distribucion_bins(resultados, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6), squeeze=False, sharex=True)
    axes = axes[0]
    edges_uniformes = np.asarray(resultados["uniformes"]["edges"], dtype=float)
    xmin = float(edges_uniformes[0])
    xmax = float(edges_uniformes[-1])
    usar_escala_log = False

    for tipo in TIPOS_BINS:
        payload = resultados[tipo]
        antes = np.asarray(payload["conteo_bins_antes"], dtype=float)
        if np.max(antes) <= 0:
            continue
        positivos = antes[antes > 0]
        ratio = float(np.max(antes) / max(np.min(positivos), 1.0)) if positivos.size > 0 else 1.0
        if ratio > 30.0:
            usar_escala_log = True
            break

    for ax, tipo in zip(axes, TIPOS_BINS):
        payload = resultados[tipo]
        edges = np.asarray(payload["edges"], dtype=float)
        anchuras = np.diff(edges)
        centros = edges[:-1] + anchuras / 2.0
        n_reales = len(centros)
        antes = np.asarray(payload["conteo_bins_antes"], dtype=float)[:n_reales]
        despues = np.asarray(payload["conteo_bins_despues"], dtype=float)[:n_reales]

        ax.bar(
            centros,
            antes,
            width=anchuras * 0.94,
            align="center",
            color=COLORES["original"],
            alpha=0.30,
            edgecolor=COLORES["original"],
            linewidth=1.0,
            label="Antes",
        )
        ax.bar(
            centros,
            despues,
            width=anchuras * 0.62,
            align="center",
            color=COLORES[tipo],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            label="Despues",
        )
        ax.set_title(f"{formatear_tipo_bins(tipo)} (bins reales={payload['metricas']['n_bins_reales']})")
        ax.set_xlabel("Fitness")
        ax.set_ylabel("Muestras (escala log)" if usar_escala_log else "Muestras")
        ax.set_xlim(xmin, xmax)
        if usar_escala_log:
            ax.set_yscale("log")
        estilo_ejes(ax)
        ax.legend(loc="best")

    fig.suptitle("Comparacion de distribucion por bins antes vs despues")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_anchura_bins(resultados, outpath):
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for tipo in TIPOS_BINS:
        anchuras = np.asarray(resultados[tipo]["anchuras_bins"], dtype=float)
        x = np.arange(1, len(anchuras) + 1, dtype=float)
        ax.step(x, anchuras, where="mid", linewidth=2.2, label=formatear_tipo_bins(tipo), color=COLORES[tipo])
        ax.scatter(x, anchuras, s=24, color=COLORES[tipo], alpha=0.9)
    ax.set_title("Anchura de los bins por tipo de discretizacion")
    ax.set_xlabel("Indice de bin")
    ax.set_ylabel("Anchura del bin")
    estilo_ejes(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_densidad_fitness(fitness_original, resultados, outpath):
    series = [np.asarray(fitness_original, dtype=float)]
    series.extend(np.asarray(resultados[tipo]["fitness_balanceado"], dtype=float) for tipo in TIPOS_BINS)
    p_low = min(float(np.percentile(arr, 0.5)) for arr in series if arr.size > 0)
    p_high = max(float(np.percentile(arr, 99.5)) for arr in series if arr.size > 0)
    bins = np.linspace(p_low, p_high, 80)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.hist(
        np.asarray(fitness_original, dtype=float),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        color=COLORES["original"],
        label="Original",
    )
    for tipo in TIPOS_BINS:
        ax.hist(
            np.asarray(resultados[tipo]["fitness_balanceado"], dtype=float),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=COLORES[tipo],
            label=f"Balanceado {tipo}",
        )
    ax.set_title("Distribucion de fitness: original vs balanceados")
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Densidad aproximada")
    estilo_ejes(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_retencion_seeds(resultados, outpath):
    categorias = sorted(
        {
            int(v)
            for tipo in TIPOS_BINS
            for v in resultados[tipo]["seeds"]["categorias"]
        }
    )
    x = np.arange(len(categorias), dtype=float)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(13.2, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )

    for tipo in TIPOS_BINS:
        payload = resultados[tipo]["seeds"]
        antes = {int(k): int(v) for k, v in zip(payload["categorias"], payload["antes"])}
        despues = {int(k): int(v) for k, v in zip(payload["categorias"], payload["despues"])}
        ret = {int(k): float(v) for k, v in zip(payload["categorias"], payload["retencion_pct"])}
        y_despues = np.asarray([despues.get(cat, 0) for cat in categorias], dtype=float)
        y_ret = np.asarray([ret.get(cat, 0.0) for cat in categorias], dtype=float)

        ax_top.plot(x, y_despues, marker="o", linewidth=1.8, label=formatear_tipo_bins(tipo), color=COLORES[tipo])
        ax_bottom.plot(x, y_ret, marker="o", linewidth=1.8, label=formatear_tipo_bins(tipo), color=COLORES[tipo])

    ax_top.set_title("Retencion por seed tras el balanceo")
    ax_top.set_ylabel("Muestras conservadas")
    ax_top.set_yscale("log")
    estilo_ejes(ax_top)
    ax_top.legend(loc="best")

    ax_bottom.set_ylabel("Retencion %")
    ax_bottom.set_xlabel("Seed")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([str(cat) for cat in categorias], rotation=90 if len(categorias) > 20 else 0)
    estilo_ejes(ax_bottom)
    ax_bottom.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_retencion_fases(resultados, outpath):
    if any(resultados[tipo]["fases"] is None for tipo in TIPOS_BINS):
        return False

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9.8, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )
    x = np.arange(len(FASES_LABELS), dtype=float)

    for tipo in TIPOS_BINS:
        payload = resultados[tipo]["fases"]
        despues = np.asarray(payload["despues"], dtype=float)
        ret = np.asarray(payload["retencion_pct"], dtype=float)
        ax_top.plot(x, despues, marker="o", linewidth=1.8, label=formatear_tipo_bins(tipo), color=COLORES[tipo])
        ax_bottom.plot(x, ret, marker="o", linewidth=1.8, label=formatear_tipo_bins(tipo), color=COLORES[tipo])

    ax_top.set_title("Retencion por fase relativa")
    ax_top.set_ylabel("Muestras conservadas")
    estilo_ejes(ax_top)
    ax_top.legend(loc="best")

    ax_bottom.set_ylabel("Retencion %")
    ax_bottom.set_xlabel("Fase")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(FASES_LABELS)
    estilo_ejes(ax_bottom)
    ax_bottom.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True


def construir_payload_salida(rutas_npz, resultados, recomendado, scores, desglose, comentario, args):
    filas_csv = []
    payload_resultados = {}
    for tipo in TIPOS_BINS:
        payload = resultados[tipo]
        metricas = dict(payload["metricas"])
        metricas["score_recomendacion"] = float(scores[tipo])
        payload_resultados[tipo] = {
            "metricas": metricas,
            "edges": payload["edges"],
            "conteo_bins_antes": payload["conteo_bins_antes"],
            "conteo_bins_despues": payload["conteo_bins_despues"],
            "seeds": payload["seeds"],
            "fases": payload["fases"],
            "desglose_score": desglose[tipo],
        }
        filas_csv.append(
            {
                "tipo_bins": tipo,
                "score_recomendacion": float(scores[tipo]),
                **metricas,
            }
        )

    payload = {
        "inputs": [str(Path(p).resolve()) for p in rutas_npz],
        "parametros": {
            "n_bins": int(args.n_bins),
            "max_por_bin": int(args.max_por_bin),
            "seed": int(args.seed),
            "estratificar_por_fase": bool(args.estratificar_por_fase),
        },
        "recomendacion": {
            "tipo_bins_recomendado": recomendado,
            "comentario": comentario,
            "scores": {k: float(v) for k, v in scores.items()},
        },
        "resultados": payload_resultados,
    }
    return payload, filas_csv


def main():
    args = parse_args()
    contexto = inferir_contexto_inputs(args.inputs)
    algoritmo = contexto["algoritmo"]
    outdir = inferir_outdir(args.inputs, args.outdir)
    dataset, fases = preparar_dataset(args.inputs, args.estratificar_por_fase)

    resultados = {
        tipo: evaluar_tipo_bins(
            dataset,
            fases,
            tipo_bins=tipo,
            n_bins=int(args.n_bins),
            max_por_bin=int(args.max_por_bin),
            random_state=int(args.seed),
            estratificar_por_fase=bool(args.estratificar_por_fase),
        )
        for tipo in TIPOS_BINS
    }

    recomendado, scores, desglose, comentario = normalizar_scores(resultados)
    payload, filas_csv = construir_payload_salida(args.inputs, resultados, recomendado, scores, desglose, comentario, args)

    prefijo = f"{algoritmo}_"
    ruta_json = outdir / f"{prefijo}comparacion_tipo_bins.json"
    ruta_csv = outdir / f"{prefijo}comparacion_tipo_bins.csv"
    ruta_bins = outdir / f"{prefijo}comparacion_distribucion_bins.png"
    ruta_widths = outdir / f"{prefijo}comparacion_anchura_bins.png"
    ruta_fit = outdir / f"{prefijo}comparacion_densidad_fitness.png"
    ruta_seeds = outdir / f"{prefijo}comparacion_retencion_seeds.png"
    ruta_fases = outdir / f"{prefijo}comparacion_retencion_fases.png"

    escribir_json(ruta_json, payload)
    escribir_csv_dicts(ruta_csv, filas_csv)
    plot_distribucion_bins(resultados, ruta_bins)
    plot_anchura_bins(resultados, ruta_widths)
    plot_densidad_fitness(dataset["fitness"], resultados, ruta_fit)
    plot_retencion_seeds(resultados, ruta_seeds)
    plot_retencion_fases(resultados, ruta_fases)

    print(f"Comparacion guardada en: {outdir}")
    print(f"algoritmo: {algoritmo}")
    print(f"tipo_bins_recomendado: {recomendado}")
    print(f"comentario: {comentario}")
    print(f"json: {ruta_json}")
    print(f"csv: {ruta_csv}")
    print(f"plots: {ruta_bins}, {ruta_widths}, {ruta_fit}, {ruta_seeds}")
    if ruta_fases.exists():
        print(f"plot_fases: {ruta_fases}")


if __name__ == "__main__":
    main()
