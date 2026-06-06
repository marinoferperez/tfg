# BOXPLOTS (fitness y tiempo) -> rendimiento final
# CURVA DE CONVERGENCIA -> dinámica del fitness (mejor/promedio/peor run)
# CURVA DE DIVERSIDAD VS EVALS -> dinámica estructural
# CURVAS CONJUNTAS DE FITNESS VS DIVERSIDAD -> relación fitness vs diversidad
# HISTOGRAMA FITNESS -> estabilidad existente entre las distintas runs
# PREDICTOR CON DIVERSIDAD -> capacidad de mejora de fitness futura


import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from preprocesado_de_datos.utils.utils import cargar_dataset
except ModuleNotFoundError:
    cargar_dataset = None

ADAPTACION_PRIORIDAD = ("age_cec2017", "de_cec2017", "shade_cec2017")
ALGORITMO_PRIORIDAD = ("age", "de", "shade")
COLORES_ALGORITMO = {
    "age": "#4e79a7",
    "de": "#f28e2b",
    "shade": "#59a14f",
}
CEC_OPTIMO_POR_FUNCID = {funcid: funcid * 100.0 for funcid in range(1, 31)}
DIVERSIDAD_POR_PROBLEMA = {
    "cec2017": (
        ("div_dist_euclidea_normalizada", "Diversidad normalizada"),
    ),
}
METRICAS_DIVERSIDAD_NORMALIZADAS = {
    "div_dist_euclidea": (
        "div_dist_euclidea_normalizada",
        "Diversidad normalizada",
    ),
}
METRICAS_NORMALIZADAS = set()
VARIANTES_REINICIO = (
    {
        "clave": "sin_reinicio",
        "umbral": None,
        "label": "Sin reinicio",
        "color": "#d62728",
        "dashtype": "solid",
    },
    {
        "clave": "reinicio_pat001",
        "umbral": 0.01,
        "label": "Ventana 1%",
        "color": "#9467bd",
        "dashtype": "4",
    },
    {
        "clave": "reinicio_pat003",
        "umbral": 0.03,
        "label": "Ventana 3%",
        "color": "#ff7f0e",
        "dashtype": "2",
    },
    {
        "clave": "reinicio_pat005",
        "umbral": 0.05,
        "label": "Ventana 5%",
        "color": "#2ca02c",
        "dashtype": "3",
    },
    {
        "clave": "reinicio_pat007",
        "umbral": 0.07,
        "label": "Ventana 7%",
        "color": "#1f77b4",
        "dashtype": "5",
    },
    {
        "clave": "reinicio_pat010",
        "umbral": 0.10,
        "label": "Ventana 10%",
        "color": "#8c564b",
        "dashtype": "6",
    },
    {
        "clave": "reinicio_005",
        "umbral": 0.05,
        "label": "Reinicio 5%",
        "color": "#2ca02c",
        "dashtype": "2",
    },
    {
        "clave": "reinicio_010",
        "umbral": 0.10,
        "label": "Reinicio 10%",
        "color": "#1f77b4",
        "dashtype": "3",
    },
)
VARIANTE_REINICIO_POR_CLAVE = {variante["clave"]: variante for variante in VARIANTES_REINICIO}
CLAVES_VARIANTES_REINICIO_FILTRADAS = None
METRICA_DIVERSIDAD_CEC = "div_dist_euclidea_normalizada"
ETIQUETA_DIVERSIDAD_CEC = "Diversidad normalizada"
CEC_VARIANTES_DIRS = {
    "sin_reinicio": (ROOT / "results" / "cec" / "cec2017_d10_tam50",),
    "reinicio_pat001": (
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_pat001",
    ),
    "reinicio_pat003": (
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_pat003",
    ),
    "reinicio_pat005": (
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_pat005",
    ),
    "reinicio_pat007": (
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_pat007",
    ),
    "reinicio_pat010": (
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_pat010",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_pat01",
    ),
    "reinicio_005": (
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_005_deltaD",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_005_dimnorm",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_005",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_005_elite1",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_elite1_005",
    ),
    "reinicio_010": (
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_010_deltaD",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_010_dimnorm",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_010",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicion_010",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_010_elite1",
        ROOT / "results" / "cec" / "cec2017_d10_tam50_reinicio_elite1_010",
    ),
}
FASES_PREDICTOR = (
    (0.0, 0.2, "0-20%"),
    (0.2, 0.4, "20-40%"),
    (0.4, 0.6, "40-60%"),
    (0.6, 0.8, "60-80%"),
    (0.8, 1.0, "80-100%"),
)
PLOTS_POR_MODO = {
    "experimento": (
        "boxplot_fitness",
        "boxplot_tiempo",
        "convergencia",
        "convergencia_variante",
        "convergencia_reinicio",
        "diversidad",
        "diversidad_reinicio",
        "reinicios_medios",
        "fitness_diversidad",
        "predictor_diversidad",
        "histograma_fitness_final",
        "tabla_cec",
    ),
    "run": (
        "histograma_fitness_fases",
        "histograma_fitness_dataset",
    ),
}
ALIASES_PLOTS = {
    "boxplot_fitness": "boxplot_fitness",
    "fitness": "boxplot_fitness",
    "boxplot_tiempo": "boxplot_tiempo",
    "tiempo": "boxplot_tiempo",
    "convergencia": "convergencia",
    "curva_convergencia": "convergencia",
    "convergencia_reinicio": "convergencia_reinicio",
    "curva_convergencia_reinicio": "convergencia_reinicio",
    "convergencia_reinicios": "convergencia_reinicio",
    "curva_convergencia_reinicios": "convergencia_reinicio",
    "diversidad": "diversidad",
    "curva_diversidad": "diversidad",
    "diversidad_reinicio": "diversidad_reinicio",
    "curva_diversidad_reinicio": "diversidad_reinicio",
    "diversidad_reinicios": "diversidad_reinicio",
    "curva_diversidad_reinicios": "diversidad_reinicio",
    "reinicios_medios": "reinicios_medios",
    "media_reinicios": "reinicios_medios",
    "n_reinicios_medios": "reinicios_medios",
    "reinicios_medios_por_funcion": "reinicios_medios",
    "fitness_diversidad": "fitness_diversidad",
    "curva_fitness_diversidad": "fitness_diversidad",
    "predictor_diversidad": "predictor_diversidad",
    "predictor_diversidad_mejora_futura": "predictor_diversidad",
    "histograma_fitness_final": "histograma_fitness_final",
    "histograma_final": "histograma_fitness_final",
    "convergencia_variante": "convergencia_variante",
    "curva_convergencia_variante": "convergencia_variante",
    "tabla_cec": "tabla_cec",
    "tabla_resultados_cec": "tabla_cec",
    "histograma_fitness_fases": "histograma_fitness_fases",
    "histograma_fases": "histograma_fitness_fases",
    "histograma_fitness_dataset": "histograma_fitness_dataset",
    "histograma_fitness_dataset_completo": "histograma_fitness_dataset",
    "histograma_dataset": "histograma_fitness_dataset",
}

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


def obtener_transformacion_convergencia(filas):
    problemas = {fila.get("problema") for fila in filas}
    funcids = {fila.get("cec_funcid") for fila in filas if fila.get("cec_funcid") is not None}
    if problemas == {"cec2017"} and len(funcids) == 1:
        funcid = int(next(iter(funcids)))
        optimo = CEC_OPTIMO_POR_FUNCID.get(funcid)
        if optimo is not None:
            return {
                "tipo": "cec_log_gap",
                "funcid": funcid,
                "optimo": float(optimo),
                "ylabel": r"$\log_{10}(f - f^* + 1)$",
            }
    return None


def aplicar_transformacion_convergencia(y, transformacion):
    valores = np.asarray(y, dtype=float)
    if transformacion is None:
        return valores
    if transformacion["tipo"] == "cec_log_gap":
        gap = np.maximum(valores - float(transformacion["optimo"]), 0.0)
        return np.log10(gap + 1.0)
    return valores


def _resolver_ruta_dataset(ruta_metricas):
    if not ruta_metricas:
        return None

    base = Path(ruta_metricas)
    for patron in ("dataset_*.h5", "dataset_*.hdf5"):
        rutas = sorted(base.glob(patron))
        if rutas:
            return rutas[0]
    return None


def _resolver_ruta_resultados_csv(ruta_metricas):
    if not ruta_metricas:
        return None
    rutas = sorted(Path(ruta_metricas).glob("resultados_*.csv"))
    return rutas[0] if rutas else None


def _cargar_dataset_hdf5(ruta_dataset):
    data = {}
    with pd.HDFStore(ruta_dataset, mode="r") as store:
        if "/dataset" in store:
            df = store["dataset"]
            for columna in df.columns:
                data[columna] = df[columna].to_numpy()
            x_cols = sorted([col for col in df.columns if col.startswith("x_")], key=lambda x: int(x.split("_")[1]))
            if x_cols:
                data["x"] = df[x_cols].to_numpy(dtype=float)
    return data


def cargar_dataset_metricas(ruta_dataset):
    ruta_dataset = Path(ruta_dataset)
    sufijo = ruta_dataset.suffix.lower()

    if sufijo in {".h5", ".hdf5"}:
        return _cargar_dataset_hdf5(ruta_dataset)

    if cargar_dataset is not None:
        return cargar_dataset(ruta_dataset)

    raise ValueError(f"Formato de dataset no soportado: {ruta_dataset}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=str, default="results/experimentos_mhs/runs.csv")
    parser.add_argument("--outdir", type=str, default="results/experimentos_mhs/plots")
    parser.add_argument(
        "--modo",
        type=str,
        default="experimento",
        choices=["experimento", "run"],
        help="Genera plots agregados por experimento o solo plots_preprocesado de una run aleatoria.",
    )
    parser.add_argument(
        "--algoritmo",
        type=str,
        default="ambos",
        choices=["age", "de", "shade", "ambos", "todos"],
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
        "--cec-funcid",
        type=int,
        nargs="+",
        default=None,
        help="En modo run + CEC2017, una o varias funciones sobre las que escoger runs aleatorias.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para seleccionar la run aleatoria en modo run.",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        nargs="+",
        default=None,
        help="Filtra por una o varias semillas concretas.",
    )
    parser.add_argument(
        "--delta-futuro",
        type=int,
        default=50,
        help="Delta en generaciones para medir mejora futura: fitness(t)-fitness(t+delta)",
    )
    parser.add_argument(
        "--diversidad-reinicio-agregacion",
        type=str,
        default="representativa",
        choices=["representativa", "promedio"],
        help=(
            "Cómo agregar las curvas de diversidad con reinicio: "
            "'representativa' selecciona una semilla representativa y 'promedio' "
            "promedia las curvas disponibles por configuración."
        ),
    )
    parser.add_argument(
        "--variantes-reinicio",
        type=str,
        nargs="+",
        default=None,
        choices=[variante["clave"] for variante in VARIANTES_REINICIO],
        help="Filtra las variantes de reinicio que se muestran en los plots comparativos.",
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=["todos"],
        help=(
            "Lista de plots a generar. Usa 'todos' por defecto. "
            "Modo experimento: boxplot_fitness, boxplot_tiempo, convergencia, diversidad, "
            "convergencia_reinicio, diversidad_reinicio, "
            "reinicios_medios, "
            "fitness_diversidad, predictor_diversidad, histograma_fitness_final, tabla_cec. "
            "Modo run: histograma_fitness_fases, histograma_fitness_dataset."
        ),
    )
    parser.add_argument(
        "--seeds-representativas-json",
        type=str,
        default=None,
        help="En modo run, JSON generado por seed_representativa_mh.py para seleccionar runs concretas.",
    )
    return parser.parse_args()


def resolver_plots_seleccionados(modo, plots_cli):
    disponibles = set(PLOTS_POR_MODO[modo])
    seleccionados = []

    for plot in plots_cli or ["todos"]:
        plot_normalizado = str(plot).strip().lower()
        if plot_normalizado in {"todos", "all"}:
            return set(PLOTS_POR_MODO[modo])

        canonico = ALIASES_PLOTS.get(plot_normalizado)
        if canonico is None:
            opciones = ", ".join(sorted(ALIASES_PLOTS))
            raise ValueError(f"Plot no reconocido: {plot}. Valores aceptados: {opciones}")
        seleccionados.append(canonico)

    no_disponibles = [plot for plot in seleccionados if plot not in disponibles]
    if no_disponibles:
        opciones_modo = ", ".join(PLOTS_POR_MODO[modo])
        raise ValueError(
            f"Los plots {no_disponibles} no estan disponibles en modo {modo}. "
            f"Disponibles: {opciones_modo}"
        )

    return set(seleccionados)

def ordenar_por_prioridad(valores, prioridad):
    presentes = set(valores)
    ordenados = [v for v in prioridad if v in presentes]
    extras = sorted(presentes.difference(set(ordenados)))
    return ordenados + extras


def normalizar_cec_funcids(cec_funcid):
    if cec_funcid is None:
        return None
    if isinstance(cec_funcid, (list, tuple, set)):
        return [int(v) for v in cec_funcid]
    return [int(cec_funcid)]

def normalizar_umbral_reinicio(valor, ruta_metricas=""):
    txt = str(valor if valor is not None else "").strip().lower()
    if txt in {"", "none", "nan"}:
        ruta_txt = str(ruta_metricas or "").strip().lower()
        if "_udiv0p05" in ruta_txt:
            return 0.05
        if "_udiv0p1" in ruta_txt or "_udiv0p10" in ruta_txt:
            return 0.10
        return None

    umbral = float(txt)
    if np.isclose(umbral, 0.05, atol=1e-9):
        return 0.05
    if np.isclose(umbral, 0.10, atol=1e-9):
        return 0.10
    return umbral


def clasificar_variante_reinicio(umbral):
    for variante in variantes_reinicio_activas():
        umbral_variante = variante["umbral"]
        if umbral_variante is None and umbral is None:
            return variante["clave"]
        if (
            umbral_variante is not None
            and umbral is not None
            and np.isclose(float(umbral), float(umbral_variante), atol=1e-9)
        ):
            return variante["clave"]
    return None


def variantes_reinicio_activas():
    if not CLAVES_VARIANTES_REINICIO_FILTRADAS:
        return VARIANTES_REINICIO
    claves = set(CLAVES_VARIANTES_REINICIO_FILTRADAS)
    return tuple(variante for variante in VARIANTES_REINICIO if variante["clave"] in claves)


def obtener_clave_variante_reinicio(fila):
    clave = str(fila.get("variante_reinicio", "") or "").strip()
    if clave:
        return clave
    return clasificar_variante_reinicio(
        normalizar_umbral_reinicio(
            fila.get("reinicio_elitista_umbral_diversidad", ""),
            ruta_metricas=fila.get("ruta_metricas", ""),
        )
    )


def etiqueta_variante_reinicio(clave):
    if clave in VARIANTE_REINICIO_POR_CLAVE:
        return VARIANTE_REINICIO_POR_CLAVE[clave]["label"]
    if clave is None:
        return "Sin reinicio"
    return str(clave)


def etiqueta_variante_reinicio_algoritmo(algoritmo, clave):
    prefijo = str(algoritmo).upper()
    if clave in {None, "", "sin_reinicio"}:
        return prefijo
    sufijos = {
        "reinicio_pat001": "1",
        "reinicio_pat003": "3",
        "reinicio_pat005": "5",
        "reinicio_pat007": "7",
        "reinicio_pat010": "10",
        "reinicio_005": "5",
        "reinicio_010": "10",
    }
    return f"{prefijo}-{sufijos.get(clave, str(clave))}"


def ordenar_claves_variantes_reinicio(claves):
    prioridad = {variante["clave"]: idx for idx, variante in enumerate(variantes_reinicio_activas())}
    return sorted(
        {clave for clave in claves if clave is not None},
        key=lambda clave: (prioridad.get(clave, len(prioridad)), str(clave)),
    )

def describir_bloque_problema(filas, problema):
    if problema == "cec2017":
        funcids = sorted({int(fila["cec_funcid"]) for fila in filas if fila.get("cec_funcid") is not None})
        if len(funcids) == 1:
            return f"CEC2017 | f{funcids[0]}"
        if len(funcids) > 1:
            return "CEC2017"
    return str(problema).upper()

def resolver_directorio_variante_cec(clave_variante):
    candidatas = CEC_VARIANTES_DIRS.get(clave_variante, ())
    for candidata in candidatas:
        if candidata.exists():
            return candidata
    return None

def cargar_filas_variantes_cec_por_funcion(funcid, algoritmo="ambos"):
    filas = []
    for variante in variantes_reinicio_activas():
        root_variante = resolver_directorio_variante_cec(variante["clave"])
        if root_variante is None:
            continue

        ruta_runs = root_variante / f"f{int(funcid)}" / "runs.csv"
        if not ruta_runs.exists():
            continue

        filas_variante = cargar_runs(ruta_runs, algoritmo, "cec2017")
        for fila in filas_variante:
            fila_nueva = dict(fila)
            fila_nueva["variante_reinicio"] = variante["clave"]
            fila_nueva["reinicio_elitista_umbral_diversidad"] = variante["umbral"]
            filas.append(fila_nueva)
    return filas

def cargar_runs(path, algoritmo, problema):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de resultados: {path}")

    if path.is_dir():
        rutas_runs = []
        if (path / "runs.csv").exists():
            rutas_runs.append(path / "runs.csv")
        rutas_runs.extend(sorted(path.glob("f*/runs.csv"), key=lambda p: int(p.parent.name[1:])))
        if not rutas_runs:
            raise FileNotFoundError(f"No se encontraron ficheros runs.csv en: {path}")
        filas = []
        for ruta_runs in rutas_runs:
            filas.extend(cargar_runs(ruta_runs, algoritmo, problema))
        return filas

    filas = []
    with path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for fila in reader:
            fila_alg = fila["algoritmo"].strip().lower()
            fila_prob = fila["problema"].strip().lower()

            if algoritmo not in {"ambos", "todos"} and fila_alg != algoritmo:
                continue
            if problema != "ambos" and fila_prob != problema:
                continue

            adaptacion = str(fila.get("adaptacion", "")).strip().lower()
            if adaptacion == "":
                adaptacion = f"{fila_alg}_{fila_prob}"

            ruta_metricas = fila.get("ruta_metricas", "").strip()
            umbral_reinicio = normalizar_umbral_reinicio(
                fila.get("reinicio_elitista_umbral_diversidad", ""),
                ruta_metricas=ruta_metricas,
            )

            filas.append(
                {
                    "problema": fila_prob,
                    "algoritmo": fila_alg,
                    "adaptacion": adaptacion,
                    "cec_funcid": int(fila["cec_funcid"]) if str(fila.get("cec_funcid", "")).strip() != "" else None,
                    "qap_instancia": fila.get("qap_instancia", "").strip(),
                    "semilla": int(fila["semilla"]) if str(fila.get("semilla", "")).strip() != "" else None,
                    "fitness": float(fila["fitness"]),
                    "tiempo_s": float(fila["tiempo_s"]),
                    "ruta_metricas": ruta_metricas,
                    "reinicio_elitista_umbral_diversidad": umbral_reinicio,
                    "n_reinicios_elitistas": (
                        int(fila["n_reinicios_elitistas"])
                        if str(fila.get("n_reinicios_elitistas", "")).strip() != ""
                        else None
                    ),
                    "variante_reinicio": clasificar_variante_reinicio(umbral_reinicio),
                }
            )
    return filas


def resolver_ruta_metricas(ruta_metricas):
    ruta_txt = str(ruta_metricas or "").strip()
    if ruta_txt == "":
        return ""

    ruta = Path(ruta_txt)
    if ruta.exists():
        return str(ruta)

    txt = ruta_txt.replace("\\", "/")
    marcadores = (
        "/results/experimentos_mhs_ambos_qap_",
        "/results/experimentos_mhs_ambos_cec2017_",
    )
    for marcador in marcadores:
        if marcador not in txt:
            continue

        sufijo = txt.split(marcador, 1)[1]
        prefijo = txt.split("/results/", 1)[0]
        if marcador.endswith("_qap_"):
            candidatas = [
                Path(prefijo) / "results" / "qap" / f"experimentos_mhs_ambos_qap_{sufijo}",
            ]
        else:
            candidatas = [
                Path(prefijo) / "results" / "cec" / f"experimentos_mhs_ambos_cec2017_{sufijo}",
                Path(prefijo) / "results" / "cec2017" / f"experimentos_mhs_ambos_cec2017_{sufijo}",
            ]
        for candidata in candidatas:
            if candidata.exists():
                return str(candidata)

    return ruta_txt


def resolver_ruta_metricas_desde_fila(fila, results_csv):
    ruta_resuelta = resolver_ruta_metricas(fila.get("ruta_metricas", ""))
    if ruta_resuelta and Path(ruta_resuelta).exists():
        return Path(ruta_resuelta)

    results_dir = Path(results_csv).resolve().parent
    problema = fila.get("problema")
    algoritmo = fila.get("algoritmo")
    semilla = fila.get("semilla")

    candidatas = []

    if problema == "qap":
        instancia = fila.get("qap_instancia", "").strip()
        if instancia:
            candidatas.append(
                results_dir / "metricas_runs" / "qap" / algoritmo / f"{algoritmo}_qap_{instancia}_s{int(semilla)}"
            )
    elif problema == "cec2017":
        funcid = fila.get("cec_funcid")
        if funcid is not None:
            alg_dir = results_dir / f"f{int(funcid)}" / "metricas_runs" / "cec2017" / algoritmo
            if alg_dir.exists():
                patron = f"{algoritmo}_cec2017_f{int(funcid)}_*_s{int(semilla)}"
                candidatas.extend(sorted(alg_dir.glob(patron)))

    for candidata in candidatas:
        if candidata.exists():
            return candidata

    raise FileNotFoundError(
        "No existe la carpeta de metricas de la run seleccionada y no se pudo reconstruir su ruta: "
        f"{fila.get('ruta_metricas', '')}"
    )


def agrupar_filas_cec_por_funcid(filas):
    grupos = {}
    for fila in filas:
        funcid = fila.get("cec_funcid")
        if funcid is None:
            continue
        grupos.setdefault(int(funcid), []).append(fila)
    return dict(sorted(grupos.items(), key=lambda kv: kv[0]))


def seleccionar_run_aleatoria(filas, *, problema, cec_funcid=None, random_state=42):
    if problema == "cec2017":
        if cec_funcid is None:
            raise ValueError("En modo run para CEC2017 debes indicar --cec-funcid.")
        candidatas = [
            fila
            for fila in filas
            if fila.get("problema") == "cec2017" and fila.get("cec_funcid") == int(cec_funcid)
        ]
    elif problema == "qap":
        candidatas = [fila for fila in filas if fila.get("problema") == "qap"]
    else:
        raise ValueError("El modo run solo admite --problema cec2017 o --problema qap.")

    if len(candidatas) == 0:
        raise ValueError("No hay runs disponibles con los filtros indicados para modo run.")

    rng = random.Random(int(random_state))
    return rng.choice(candidatas)


def seleccionar_runs_misma_seed_por_algoritmo(filas, *, problema, cec_funcid=None, random_state=42):
    algoritmos_presentes = ordenar_algoritmos([fila["algoritmo"] for fila in filas])
    if len(algoritmos_presentes) == 0:
        raise ValueError("No hay runs disponibles con los filtros indicados para modo run.")

    filas_filtradas = []
    for fila in filas:
        if problema == "cec2017":
            if fila.get("problema") != "cec2017" or fila.get("cec_funcid") != int(cec_funcid):
                continue
        elif problema == "qap":
            if fila.get("problema") != "qap":
                continue
        else:
            raise ValueError("El modo run solo admite --problema cec2017 o --problema qap.")
        filas_filtradas.append(fila)

    semillas_por_algoritmo = {}
    for algoritmo in algoritmos_presentes:
        semillas = {
            int(fila["semilla"])
            for fila in filas_filtradas
            if fila.get("algoritmo") == algoritmo and fila.get("semilla") is not None
        }
        if len(semillas) == 0:
            raise ValueError(f"No hay runs disponibles para el algoritmo {algoritmo} con los filtros indicados.")
        semillas_por_algoritmo[algoritmo] = semillas

    semillas_comunes = None
    for semillas in semillas_por_algoritmo.values():
        semillas_comunes = semillas if semillas_comunes is None else semillas_comunes & semillas

    if not semillas_comunes:
        raise ValueError("No hay ninguna seed comun entre los algoritmos para los filtros indicados.")

    rng = random.Random(int(random_state))
    semilla_objetivo = rng.choice(sorted(semillas_comunes))

    seleccionadas = []
    for algoritmo in algoritmos_presentes:
        candidatas = [
            fila
            for fila in filas_filtradas
            if fila.get("algoritmo") == algoritmo and int(fila.get("semilla")) == int(semilla_objetivo)
        ]
        if len(candidatas) == 0:
            raise ValueError(
                f"No se encontro una run para algoritmo={algoritmo} con la seed comun seleccionada: {semilla_objetivo}"
            )
        seleccionadas.append(candidatas[0])
    return seleccionadas


def seleccionar_runs_aleatorias_por_algoritmo(filas, *, problema, cec_funcid=None, random_state=42):
    algoritmos_presentes = ordenar_algoritmos([fila["algoritmo"] for fila in filas])
    if len(algoritmos_presentes) == 0:
        raise ValueError("No hay runs disponibles con los filtros indicados para modo run.")
    if len(algoritmos_presentes) > 1:
        return seleccionar_runs_misma_seed_por_algoritmo(
            filas,
            problema=problema,
            cec_funcid=cec_funcid,
            random_state=random_state,
        )

    seleccionadas = []
    for idx, algoritmo in enumerate(algoritmos_presentes):
        filas_alg = [fila for fila in filas if fila.get("algoritmo") == algoritmo]
        seleccionadas.append(
            seleccionar_run_aleatoria(
                filas_alg,
                problema=problema,
                cec_funcid=cec_funcid,
                random_state=int(random_state) + idx,
            )
        )
    return seleccionadas


def _funcid_desde_registro_representativo(registro):
    if registro.get("cec_funcid") is not None and str(registro.get("cec_funcid")).strip() != "":
        return int(registro["cec_funcid"])
    funcion = str(registro.get("funcion", "")).strip().lower()
    if funcion.startswith("f") and funcion[1:].isdigit():
        return int(funcion[1:])
    return None


def cargar_registros_representativos(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("resultados", data.get("seeds", []))
    if not isinstance(data, list):
        raise ValueError("El JSON de semillas representativas debe contener una lista de registros.")
    return data


def seleccionar_runs_desde_registros_representativos(filas, registros, *, problema, funcids=None, algoritmo="todos"):
    funcids_set = None if funcids is None else {int(funcid) for funcid in funcids}
    seleccionadas = []
    avisos = []

    for registro in registros:
        problema_reg = str(registro.get("problema", problema)).strip().lower()
        if problema_reg != problema:
            continue

        algoritmo_reg = str(registro.get("algoritmo", "")).strip().lower()
        if algoritmo not in {"ambos", "todos"} and algoritmo_reg != algoritmo:
            continue

        funcid = _funcid_desde_registro_representativo(registro)
        if problema == "cec2017":
            if funcid is None:
                continue
            if funcids_set is not None and int(funcid) not in funcids_set:
                continue

        semilla = int(registro["seed_representativa"])
        variante_reg = str(registro.get("variante_reinicio", "") or "").strip() or None

        coincidencias = []
        for fila in filas:
            if fila.get("problema") != problema:
                continue
            if str(fila.get("algoritmo", "")).strip().lower() != algoritmo_reg:
                continue
            if problema == "cec2017" and int(fila.get("cec_funcid")) != int(funcid):
                continue
            if int(fila.get("semilla")) != semilla:
                continue
            if variante_reg is not None and obtener_clave_variante_reinicio(fila) != variante_reg:
                continue
            coincidencias.append(fila)

        if coincidencias:
            seleccionadas.append(coincidencias[0])
        else:
            etiqueta_funcion = f"f{funcid}" if funcid is not None else problema
            avisos.append(
                f"No se encontro run para {etiqueta_funcion}/{algoritmo_reg}/"
                f"{variante_reg or 'sin variante'}/seed={semilla}"
            )

    if avisos:
        for aviso in avisos:
            print(f"AVISO: {aviso}", file=sys.stderr)

    return seleccionadas


def generar_plots_preprocesado_por_run(fila, outdir, plots_seleccionados=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_seleccionados = set(plots_seleccionados or PLOTS_POR_MODO["run"])

    problema = fila["problema"]
    algoritmo = fila["algoritmo"]
    clave_variante = obtener_clave_variante_reinicio(fila)
    etiqueta_variante = etiqueta_variante_reinicio(clave_variante)
    sufijo_variante = f"_{clave_variante}" if clave_variante else ""
    run_id = Path(fila.get("ruta_metricas", "")).name or f"{algoritmo}_{problema}_s{int(fila['semilla'])}"
    sufijo = (
        f"_{problema}_f{int(fila['cec_funcid'])}_s{int(fila['semilla'])}{sufijo_variante}"
        if problema == "cec2017"
        else f"_{problema}_{fila['qap_instancia']}_s{int(fila['semilla'])}{sufijo_variante}"
    )
    titulo_base = (
        f"Run {run_id} | {algoritmo.upper()} | {problema.upper()} | seed={fila['semilla']} | {etiqueta_variante}"
        if problema == "qap"
        else (
            f"Run {run_id} | {algoritmo.upper()} | {problema.upper()} "
            f"| f{int(fila['cec_funcid'])} | seed={fila['semilla']} | {etiqueta_variante}"
        )
    )
    titulo_base_figura = (
        f"{algoritmo.upper()} | {problema.upper()} | seed={fila['semilla']} | {etiqueta_variante}"
        if problema == "qap"
        else (
            f"{algoritmo.upper()} | {problema.upper()} f{int(fila['cec_funcid'])} "
            f"| seed={fila['semilla']} | {etiqueta_variante}"
        )
    )

    ruta_hist_fases = outdir / f"histograma_fitness_fases{sufijo}.png"
    ruta_hist_dataset = outdir / f"histograma_fitness_dataset_completo{sufijo}.png"
    generados = []

    if "histograma_fitness_fases" in plots_seleccionados:
        rutas_hist_fases = generar_histograma_fitness_por_fases(
            filas=[fila],
            titulo=f"Distribucion por bloques temporales | {titulo_base_figura}",
            outpath=ruta_hist_fases,
        )
        if rutas_hist_fases:
            generados.extend(rutas_hist_fases)

    if "histograma_fitness_dataset" in plots_seleccionados:
        if generar_histograma_fitness_dataset_completo(
            filas=[fila],
            titulo=f"Histograma de Fitness del Dataset Completo | {titulo_base}",
            outpath=ruta_hist_dataset,
        ):
            generados.append(str(ruta_hist_dataset))

    return generados


def imprimir_resumen_plots_por_run(registros, problema):
    if len(registros) == 0:
        return

    print("PLOTS POR RUN")
    for registro in registros:
        variante = registro.get("variante_reinicio")
        sufijo_variante = f" | {etiqueta_variante_reinicio(variante)}" if variante else ""
        if problema == "cec2017":
            print(f"[f{int(registro['funcid'])}] {registro['algoritmo'].upper()}{sufijo_variante}")
        else:
            print(f"[QAP] {registro['algoritmo'].upper()}{sufijo_variante}")
        print(f"run: {registro['ruta_metricas']}")
        if registro.get("ruta_fases"):
            print(f"fases: {registro['ruta_fases']}")
        if registro.get("ruta_dataset"):
            print(f"dataset: {registro['ruta_dataset']}")


def construir_outdir_por_run(outdir_base, fila):
    base = Path(outdir_base) / "por_run"
    if fila["problema"] == "cec2017":
        clave_variante = obtener_clave_variante_reinicio(fila)
        if clave_variante:
            return base / f"f{int(fila['cec_funcid'])}" / fila["algoritmo"] / clave_variante
        return base / f"f{int(fila['cec_funcid'])}" / fila["algoritmo"]
    instancia = str(fila.get("qap_instancia", "")).strip()
    return base / (instancia if instancia else "qap") / fila["algoritmo"]


def cargar_curva_desde_logbook(ruta_metricas):
    ruta_metricas = resolver_ruta_metricas(ruta_metricas)
    if not ruta_metricas:
        return None
    ruta_logbook = _resolver_ruta_resultados_csv(ruta_metricas)
    if ruta_logbook is None or not ruta_logbook.exists():
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
    ruta_metricas = resolver_ruta_metricas(ruta_metricas)
    if not ruta_metricas:
        return None
    ruta_logbook = _resolver_ruta_resultados_csv(ruta_metricas)
    if ruta_logbook is None or not ruta_logbook.exists():
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

def cargar_curva_metrica_desde_logbook(ruta_metricas, metrica, x_column="eval_id_fin"):
    ruta_metricas = resolver_ruta_metricas(ruta_metricas)
    if not ruta_metricas:
        return None

    ruta_dataset = _resolver_ruta_dataset(ruta_metricas)
    if ruta_dataset is None:
        return None

    ruta_resultados_csv = _resolver_ruta_resultados_csv(ruta_metricas)
    if ruta_resultados_csv is not None:
        diversidad_df = pd.read_csv(ruta_resultados_csv)
        x_col_real = x_column if x_column in diversidad_df.columns else None
        if x_col_real is None and x_column == "evaluaciones" and "eval_id_fin" in diversidad_df.columns:
            x_col_real = "eval_id_fin"
        if x_col_real is None or metrica not in diversidad_df.columns:
            return None
        arr_eval = np.asarray(diversidad_df[x_col_real], dtype=int).reshape(-1)
        arr_val = np.asarray(diversidad_df[metrica], dtype=float).reshape(-1)
        if arr_eval.size != arr_val.size:
            return None
        mascara = np.isfinite(arr_val)
        if not np.any(mascara):
            return None
        return arr_eval[mascara], arr_val[mascara]

    data = cargar_dataset_metricas(ruta_dataset)
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


def resolver_metrica_diversidad_para_curva_conjunta(metrica, etiqueta):
    return METRICAS_DIVERSIDAD_NORMALIZADAS.get(metrica, (metrica, etiqueta))

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

def promediar_curvas_en_malla_regular(curvas, max_puntos=1200):
    min_eval = min(int(np.nanmin(x)) for x, _ in curvas if len(x) > 0)
    max_eval = max(int(np.nanmax(x)) for x, _ in curvas if len(x) > 0)
    if max_eval <= min_eval:
        malla = np.asarray([min_eval], dtype=int)
    else:
        malla = np.unique(
            np.concatenate(
                [
                    np.asarray([0, min_eval, max_eval], dtype=int),
                    np.linspace(min_eval, max_eval, num=max_puntos, dtype=int),
                ]
            )
        )
    y_runs = np.vstack([alinear_serie_a_malla(x, y, malla) for x, y in curvas])
    return malla, np.mean(y_runs, axis=0)

def combinar_series_en_malla(x_1, y_1, x_2, y_2):
    malla = np.unique(np.concatenate([x_1, x_2]))
    y_1_malla = alinear_serie_a_malla(x_1, y_1, malla)
    y_2_malla = alinear_serie_a_malla(x_2, y_2, malla)
    return malla, y_1_malla, y_2_malla

def cargar_diversidad_y_mejora_futura_desde_logbook(ruta_metricas, metrica, delta_pasos):
    ruta_metricas = resolver_ruta_metricas(ruta_metricas)
    if delta_pasos < 1:
        raise ValueError("delta_pasos debe ser >= 1")

    ruta_dataset = _resolver_ruta_dataset(ruta_metricas)
    ruta_logbook = _resolver_ruta_resultados_csv(ruta_metricas)
    if not ruta_metricas or ruta_dataset is None or ruta_logbook is None or not ruta_logbook.exists():
        return None

    div_por_generacion = {}
    if ruta_logbook is not None:
        diversidad_df = pd.read_csv(ruta_logbook)
        columnas_requeridas = {"generacion", "eval_id_fin", metrica}
        if not columnas_requeridas.issubset(set(diversidad_df.columns)):
            return None
        for _, fila in diversidad_df.iterrows():
            gen = int(fila["generacion"])
            div = float(fila[metrica])
            if not np.isfinite(div):
                continue
            div_por_generacion[gen] = (int(fila["eval_id_fin"]), div)
    else:
        data = cargar_dataset_metricas(ruta_dataset)
        if metrica not in data or "generacion" not in data or "eval_id" not in data:
            return None
        arr_gen = np.asarray(data["generacion"], dtype=int).reshape(-1)
        arr_eval = np.asarray(data["eval_id"], dtype=int).reshape(-1)
        arr_div = np.asarray(data[metrica], dtype=float).reshape(-1)

        if not (arr_gen.size == arr_eval.size == arr_div.size):
            return None

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
    evals_arr = np.asarray(evals, dtype=int)

    x = x_div[:-delta_pasos]
    y = y_fit[:-delta_pasos] - y_fit[delta_pasos:]
    y = np.maximum(y, 0.0)
    mejora = y > 0.0
    evals_t = evals_arr[:-delta_pasos]
    evals_totales = int(evals_arr[-1])

    return x, y, mejora, evals_t, evals_totales

def asignar_fases_relativas(evals_t, evals_totales):
    evals_arr = np.asarray(evals_t, dtype=float).reshape(-1)
    if evals_arr.size == 0:
        return np.empty((0,), dtype=int)

    total = max(float(evals_totales), 1.0)
    progreso = np.clip(evals_arr / total, 0.0, 1.0)
    fases = np.full(evals_arr.shape, -1, dtype=int)
    for idx, (inicio, fin, _) in enumerate(FASES_PREDICTOR):
        if idx < len(FASES_PREDICTOR) - 1:
            mascara = (progreso >= inicio) & (progreso < fin)
        else:
            mascara = (progreso >= inicio) & (progreso <= fin)
        fases[mascara] = idx
    return fases

def calcular_bins_cuantiles(x, n_bins=4):
    x_arr = np.asarray(x, dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    if x_arr.size == 0:
        return None

    n_obj = max(1, int(n_bins))
    cuantiles = np.linspace(0.0, 1.0, num=n_obj + 1)
    bordes = np.quantile(x_arr, cuantiles)
    bordes = np.asarray(bordes, dtype=float)
    bordes = np.unique(bordes)
    if bordes.size < 2:
        return None
    return bordes

def etiquetar_bins_desde_bordes(bordes):
    etiquetas = []
    for i in range(len(bordes) - 1):
        a = bordes[i]
        b = bordes[i + 1]
        if i < len(bordes) - 2:
            etiquetas.append(f"[{a:.3g}, {b:.3g})")
        else:
            etiquetas.append(f"[{a:.3g}, {b:.3g}]")
    return etiquetas

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
    series_x_por_metrica = {metrica: [] for metrica, _ in metricas}
    series_y_por_metrica = {metrica: [] for metrica, _ in metricas}

    for algoritmo in orden_algoritmos:
        paneles = []
        for metrica, etiqueta_div in metricas:
            xs = []
            ys = []
            mejoras = []
            fases = []
            evals_norm = []
            evals_abs = []
            for fila in filas_por_algoritmo[algoritmo]:
                datos = cargar_diversidad_y_mejora_futura_desde_logbook(
                    fila.get("ruta_metricas", ""), metrica, delta_futuro
                )
                if datos is None:
                    continue
                x_run, y_run, mejora_run, evals_run, evals_totales_run = datos
                xs.append(x_run)
                ys.append(y_run)
                mejoras.append(mejora_run)
                fases_run = asignar_fases_relativas(evals_run, evals_totales_run)
                evals_norm_run = np.asarray(evals_run, dtype=float) / max(float(evals_totales_run), 1.0)
                fases.append(fases_run)
                evals_norm.append(evals_norm_run)
                evals_abs.append(np.asarray(evals_run, dtype=int))

            if len(xs) == 0:
                paneles.append(None)
                continue

            x = np.concatenate(xs)
            y = np.concatenate(ys)
            mejora = np.concatenate(mejoras)
            fases_arr = np.concatenate(fases)
            evals_norm_arr = np.concatenate(evals_norm)
            evals_abs_arr = np.concatenate(evals_abs)

            color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")
            resumen = calcular_resumen_bins_predictor(x, y, mejora, n_bins=12)
            panel = {
                "metrica": metrica,
                "etiqueta_div": etiqueta_div,
                "nombre_algoritmo": algoritmo.upper(),
                "color": color,
                "x": x,
                "y": y,
                "mejora_bool": mejora,
                "fases": fases_arr,
                "evals_rel": evals_norm_arr,
                "evals_abs": evals_abs_arr,
                "resumen": resumen,
            }
            paneles.append(panel)
            series_x_por_metrica[metrica].append(x)
            series_y_por_metrica[metrica].append(y)
            if resumen is not None:
                _, mejora_media, _, _ = resumen
                series_y_por_metrica[metrica].append(mejora_media)
            hay_grafica = True

        paneles_por_algoritmo[algoritmo] = paneles

    if not hay_grafica:
        return None

    limites_x_por_metrica = {}
    limites_y_por_metrica = {}
    for metrica, _ in metricas:
        if metrica in METRICAS_NORMALIZADAS:
            limites_x_por_metrica[metrica] = (-0.05, 1.05)
        else:
            limites_x_por_metrica[metrica] = calcular_limites_compartidos(
                series_x_por_metrica[metrica], padding_ratio=0.05, padding_min=1e-6
            )
        limites_y_por_metrica[metrica] = calcular_limites_compartidos(
            series_y_por_metrica[metrica], padding_ratio=0.08, padding_min=1e-6
        )

    return {
        "metricas": metricas,
        "orden_algoritmos": orden_algoritmos,
        "paneles_por_algoritmo": paneles_por_algoritmo,
        "limites_x_por_metrica": limites_x_por_metrica,
        "limites_y_por_metrica": limites_y_por_metrica,
    }

def generar_histogramas_distribucion_mejora_futura_por_fases(filas, problema, titulo, outpath, delta_futuro):
    datos_predictor = preparar_predictor_diversidad_mejora_futura(filas, problema, delta_futuro)
    if datos_predictor is None:
        return False

    metricas = datos_predictor["metricas"]
    orden_algoritmos = datos_predictor["orden_algoritmos"]
    paneles_por_algoritmo = datos_predictor["paneles_por_algoritmo"]

    for metrica, _ in metricas:
        for algoritmo in orden_algoritmos:
            for panel in paneles_por_algoritmo[algoritmo]:
                if panel is not None and panel["metrica"] == metrica:
                    break
            else:
                continue

    n_fases = len(FASES_PREDICTOR)
    n_cols = min(3, n_fases)
    n_rows = int(np.ceil(n_fases / n_cols))
    rutas_generadas = []

    for algoritmo in orden_algoritmos:
        for panel in paneles_por_algoritmo[algoritmo]:
            if panel is None:
                continue

            mejoras_por_fase = []
            for idx_fase, _fase in enumerate(FASES_PREDICTOR):
                mascara = panel["fases"] == idx_fase
                mejoras_por_fase.append(np.asarray(panel["y"][mascara], dtype=float))

            series_validas = [arr for arr in mejoras_por_fase if arr.size > 0]
            if len(series_validas) == 0:
                continue

            limites_x_base = calcular_limites_por_percentiles(series_validas, p_inf=0.5, p_sup=99.5)
            if limites_x_base is None:
                continue
            xmin, xmax = limites_x_base
            span_x = max(float(xmax - xmin), 1e-6)
            margen_izq = 0.02 * span_x
            x_left = xmin - margen_izq
            limites_x = (x_left, xmax)
            if limites_x is None:
                continue
            n_bins = max(12, int(np.sqrt(sum(arr.size for arr in series_validas))))
            bin_edges = np.linspace(limites_x[0], limites_x[1], num=n_bins + 1)
            max_conteo = 0
            for arr in series_validas:
                arr_vis = arr[(arr >= xmin) & (arr <= xmax)]
                conteos, _ = np.histogram(arr_vis, bins=bin_edges)
                if conteos.size > 0:
                    max_conteo = max(max_conteo, int(np.max(conteos)))
            limite_y = (0.0, max_conteo + max(max_conteo * 0.08, 1.0))

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6.2 * n_cols, 4.6 * n_rows),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            axes_flat = axes.ravel()
            for idx_fase, (inicio, fin, etiqueta_fase) in enumerate(FASES_PREDICTOR):
                ax = axes_flat[idx_fase]
                datos_fase = mejoras_por_fase[idx_fase]
                color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")
                if datos_fase.size > 0:
                    datos_vis = datos_fase[(datos_fase >= xmin) & (datos_fase <= xmax)]
                    ax.hist(
                        datos_vis,
                        bins=bin_edges,
                        color=color,
                        edgecolor="#ffffff",
                        linewidth=0.9,
                        alpha=0.92,
                    )
                    superponer_curva_histograma(ax, datos_vis, bin_edges)
                ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2)
                ax.set_title(
                    f"Fase {etiqueta_fase} ({int(inicio * 100)}-{int(fin * 100)}% evals, n={datos_fase.size})"
                )
                if idx_fase // n_cols == n_rows - 1:
                    ax.set_xlabel(f"Mejora fitness en t+{delta_futuro}")
                if idx_fase % n_cols == 0:
                    ax.set_ylabel("Frecuencia")
                ax.grid(axis="y", alpha=0.25)
                aplicar_formato_ejes_compacto(ax, ejes="both")
                aplicar_limites_compartidos(ax, "x", limites_x)
                aplicar_limites_compartidos(ax, "y", limite_y)

            for idx in range(n_fases, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle(f"{titulo} | {panel['nombre_algoritmo']}")
            fig.subplots_adjust(left=0.07, right=0.96, bottom=0.10, top=0.88, wspace=0.22, hspace=0.34)

            if len(metricas) == 1:
                ruta_panel = outpath.with_name(f"{outpath.stem}_{algoritmo}_{panel['metrica']}{outpath.suffix}")
            else:
                ruta_panel = outpath.with_name(f"{outpath.stem}_{algoritmo}_{panel['metrica']}{outpath.suffix}")
            fig.savefig(ruta_panel, dpi=180)
            plt.close(fig)
            rutas_generadas.append(ruta_panel)

    return [str(r) for r in rutas_generadas]

def generar_boxplots_mejora_futura_por_fases(filas, problema, titulo, outpath, delta_futuro, n_bins_div=4):
    datos_predictor = preparar_predictor_diversidad_mejora_futura(filas, problema, delta_futuro)
    if datos_predictor is None:
        return False

    metricas = datos_predictor["metricas"]
    orden_algoritmos = datos_predictor["orden_algoritmos"]
    paneles_por_algoritmo = datos_predictor["paneles_por_algoritmo"]
    rutas_generadas = []

    n_fases = len(FASES_PREDICTOR)
    n_cols = min(3, n_fases)
    n_rows = int(np.ceil(n_fases / n_cols))

    for algoritmo in orden_algoritmos:
        for panel in paneles_por_algoritmo[algoritmo]:
            if panel is None:
                continue

            datos_fases = []
            series_y = []
            for idx_fase, _fase in enumerate(FASES_PREDICTOR):
                mascara = panel["fases"] == idx_fase
                x_fase = np.asarray(panel["x"][mascara], dtype=float)
                y_fase = np.asarray(panel["y"][mascara], dtype=float)
                bordes = calcular_bins_cuantiles(x_fase, n_bins=n_bins_div)
                if bordes is None:
                    datos_fases.append((None, None))
                    continue
                etiquetas = etiquetar_bins_desde_bordes(bordes)
                series = []
                etiquetas_validas = []
                for idx_bin in range(len(bordes) - 1):
                    a = bordes[idx_bin]
                    b = bordes[idx_bin + 1]
                    if idx_bin < len(bordes) - 2:
                        mascara_bin = (x_fase >= a) & (x_fase < b)
                    else:
                        mascara_bin = (x_fase >= a) & (x_fase <= b)
                    y_bin = y_fase[mascara_bin]
                    if y_bin.size == 0:
                        continue
                    series.append(y_bin)
                    etiquetas_validas.append(etiquetas[idx_bin])
                    series_y.append(y_bin)
                if len(series) == 0:
                    datos_fases.append((None, None))
                else:
                    datos_fases.append((series, etiquetas_validas))

            if len(series_y) == 0:
                continue

            limites_y = calcular_limites_compartidos(series_y, padding_ratio=0.08, padding_min=1e-6)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6.6 * n_cols, 4.8 * n_rows),
                squeeze=False,
                sharey=True,
            )
            axes_flat = axes.ravel()
            color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")

            for idx_fase, (_inicio, _fin, etiqueta_fase) in enumerate(FASES_PREDICTOR):
                ax = axes_flat[idx_fase]
                series, etiquetas = datos_fases[idx_fase]
                if series is not None:
                    try:
                        bp = ax.boxplot(series, tick_labels=etiquetas, patch_artist=True, showmeans=True)
                    except TypeError:
                        bp = ax.boxplot(series, labels=etiquetas, patch_artist=True, showmeans=True)
                    for box in bp["boxes"]:
                        box.set_facecolor(color)
                        box.set_alpha(0.65)
                ax.set_title(f"Fase {etiqueta_fase}")
                if idx_fase // n_cols == n_rows - 1:
                    ax.set_xlabel("Bin de diversidad (cuantiles)")
                if idx_fase % n_cols == 0:
                    ax.set_ylabel(f"Mejora fitness en t+{delta_futuro}")
                ax.grid(axis="y", alpha=0.25)
                aplicar_formato_ejes_compacto(ax, ejes="y")
                aplicar_limites_compartidos(ax, "y", limites_y)
                ax.tick_params(axis="x", rotation=15)

            for idx in range(n_fases, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle(f"{titulo} | {panel['nombre_algoritmo']} | {panel['etiqueta_div']}")
            fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.88, wspace=0.22, hspace=0.34)

            ruta_panel = outpath.with_name(f"{outpath.stem}_{algoritmo}_{panel['metrica']}{outpath.suffix}")
            fig.savefig(ruta_panel, dpi=180)
            plt.close(fig)
            rutas_generadas.append(ruta_panel)

    return [str(r) for r in rutas_generadas]

def calcular_limites_por_percentiles(series, p_inf=0.5, p_sup=99.5):
    series_finitas = extraer_series_finitas(series)
    if len(series_finitas) == 0:
        return None

    valores = np.concatenate(series_finitas)
    if valores.size == 0:
        return None

    minimo = float(np.percentile(valores, p_inf))
    maximo = float(np.percentile(valores, p_sup))
    if not (np.isfinite(minimo) and np.isfinite(maximo)):
        return None

    if np.isclose(minimo, maximo):
        referencia = max(abs(minimo), 1.0)
        padding = max(referencia * 0.02, 1e-6)
        minimo -= padding
        maximo += padding

    return minimo, maximo


def calcular_limites_histograma(series, p_inf=0.5, p_sup=95.0):
    return calcular_limites_por_percentiles(series, p_inf=p_inf, p_sup=p_sup)


def anotar_muestras_recortadas(ax, datos, xmin, xmax):
    datos = np.asarray(datos, dtype=float)
    if datos.size == 0:
        return

    n_izq = int(np.sum(datos < xmin))
    n_der = int(np.sum(datos > xmax))
    if n_izq == 0 and n_der == 0:
        return

    partes = []
    if n_izq > 0:
        partes.append(f"<xmin: {n_izq}")
    if n_der > 0:
        partes.append(f">xmax: {n_der}")

    ax.text(
        0.98,
        0.96,
        " | ".join(partes),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d0d0"},
    )

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


def calcular_maximo_evaluaciones(series_x):
    maximo = None
    for serie in series_x:
        if serie is None:
            continue
        valores = np.asarray(serie, dtype=float)
        valores = valores[np.isfinite(valores)]
        if valores.size == 0:
            continue
        candidato = float(np.max(valores))
        if maximo is None or candidato > maximo:
            maximo = candidato

    if maximo is None:
        return None
    return max(0.0, maximo)

def _suavizar_conteos_histograma(conteos, sigma_bins=1.15):
    conteos = np.asarray(conteos, dtype=float).ravel()
    if conteos.size <= 1:
        return conteos

    sigma = max(float(sigma_bins), 1e-6)
    radio = max(1, int(np.ceil(3.0 * sigma)))
    posiciones = np.arange(-radio, radio + 1, dtype=float)
    kernel = np.exp(-0.5 * (posiciones / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(conteos, kernel, mode="same")

def superponer_curva_histograma(ax, datos, bin_edges, color="#111111", etiqueta=None):
    datos = np.asarray(datos, dtype=float).ravel()
    datos = datos[np.isfinite(datos)]
    if datos.size < 2 or len(bin_edges) < 2:
        return None

    conteos, _ = np.histogram(datos, bins=bin_edges)
    if conteos.size == 0 or np.sum(conteos) == 0:
        return None

    centros = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    curva = _suavizar_conteos_histograma(
        conteos,
        sigma_bins=min(1.6, max(0.9, len(conteos) / 18.0)),
    )
    if centros.size < 2 or np.allclose(curva, 0.0):
        return None

    x_denso = np.linspace(float(centros[0]), float(centros[-1]), num=max(250, centros.size * 30))
    y_denso = np.interp(x_denso, centros, curva)
    return ax.plot(
        x_denso,
        y_denso,
        color=color,
        linewidth=2.2,
        alpha=0.95,
        label=etiqueta,
        zorder=4,
    )[0]

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
        superponer_curva_histograma(ax, datos, bin_edges, etiqueta="curva suavizada")

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
        ax.legend(loc="best", fontsize=11)

    fig.suptitle(titulo)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    return True


def generar_histogramas_fitness_final_por_algoritmo_y_variante(filas, titulo, outpath):
    fitness_por_algoritmo_variante = {}
    for fila in filas:
        algoritmo = fila["algoritmo"]
        variante = obtener_clave_variante_reinicio(fila)
        if variante is None:
            continue
        fitness_por_algoritmo_variante.setdefault(algoritmo, {}).setdefault(variante, []).append(
            float(fila["fitness"])
        )

    algoritmos = ordenar_algoritmos(list(fitness_por_algoritmo_variante.keys()))
    if len(algoritmos) == 0:
        return []

    rutas_generadas = []
    for algoritmo in algoritmos:
        datos_por_variante = {
            variante: np.asarray(valores, dtype=float)
            for variante, valores in fitness_por_algoritmo_variante.get(algoritmo, {}).items()
            if len(valores) > 0
        }
        variantes = ordenar_claves_variantes_reinicio(datos_por_variante.keys())
        if len(variantes) <= 1:
            continue

        datos_validos = [datos_por_variante[variante] for variante in variantes]
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
            len(variantes),
            1,
            figsize=(10.4, 4.2 * len(variantes)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes = axes[:, 0]

        for idx, variante in enumerate(variantes):
            ax = axes[idx]
            datos = datos_por_variante[variante]
            color = VARIANTE_REINICIO_POR_CLAVE.get(variante, {}).get(
                "color", COLORES_ALGORITMO.get(algoritmo, "#59a14f")
            )
            media = float(np.mean(datos))
            mediana = float(np.median(datos))

            ax.hist(datos, bins=bin_edges, color=color, edgecolor="white", alpha=0.80)
            superponer_curva_histograma(ax, datos, bin_edges, etiqueta="curva suavizada")
            ax.axvline(media, color="#222222", linestyle="--", linewidth=1.2, label="media")
            ax.axvline(mediana, color="#222222", linestyle=":", linewidth=1.2, label="mediana")

            ax.set_title(f"{etiqueta_variante_reinicio(variante)} (n={len(datos)})")
            if idx == len(variantes) - 1:
                ax.set_xlabel("Fitness final")
            ax.set_ylabel("Frecuencia")
            ax.grid(axis="y", alpha=0.25)
            aplicar_formato_ejes_compacto(ax, ejes="x")
            aplicar_limites_compartidos(ax, "x", limites_x)
            aplicar_limites_compartidos(ax, "y", limite_y)
            ax.legend(loc="best", fontsize=11)

        titulo_algoritmo = f"{titulo} | {algoritmo.upper()}" if len(algoritmos) > 1 else titulo
        fig.suptitle(titulo_algoritmo)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        ruta_algoritmo = outpath.with_name(f"{outpath.stem}_{algoritmo}{outpath.suffix}")
        fig.savefig(ruta_algoritmo, dpi=180)
        plt.close(fig)
        rutas_generadas.append(str(ruta_algoritmo))

    return rutas_generadas


def cargar_fitness_dataset_desde_metricas(ruta_metricas):
    ruta_metricas = resolver_ruta_metricas(ruta_metricas)
    if not ruta_metricas:
        return None

    rutas_dataset = sorted(Path(ruta_metricas).glob("dataset_*.h5"))
    if len(rutas_dataset) == 0:
        rutas_dataset = sorted(Path(ruta_metricas).glob("dataset_*.hdf5"))
    if len(rutas_dataset) == 0:
        return None

    fitness_runs = []
    for ruta_dataset in rutas_dataset:
        data = cargar_dataset_metricas(ruta_dataset)
        if "fitness" not in data:
            continue
        arr_fit = np.asarray(data["fitness"], dtype=float).reshape(-1)
        arr_fit = arr_fit[np.isfinite(arr_fit)]
        if arr_fit.size == 0:
            continue
        fitness_runs.append(arr_fit)

    if len(fitness_runs) == 0:
        return None
    return np.concatenate(fitness_runs)

def cargar_fitness_por_fases_desde_metricas(ruta_metricas):
    ruta_metricas = resolver_ruta_metricas(ruta_metricas)
    if not ruta_metricas:
        return None

    rutas_dataset = sorted(Path(ruta_metricas).glob("dataset_*.h5"))
    if len(rutas_dataset) == 0:
        rutas_dataset = sorted(Path(ruta_metricas).glob("dataset_*.hdf5"))
    if len(rutas_dataset) == 0:
        return None

    fitness_runs = []
    fases_runs = []
    for ruta_dataset in rutas_dataset:
        data = cargar_dataset_metricas(ruta_dataset)
        if "fitness" not in data or "eval_id" not in data:
            continue
        arr_fit = np.asarray(data["fitness"], dtype=float).reshape(-1)
        arr_eval = np.asarray(data["eval_id"], dtype=int).reshape(-1)

        if arr_fit.size == 0 or arr_fit.size != arr_eval.size:
            continue

        mascara_finita = np.isfinite(arr_fit)
        arr_fit = arr_fit[mascara_finita]
        arr_eval = arr_eval[mascara_finita]
        if arr_fit.size == 0:
            continue

        evals_totales = int(np.max(arr_eval))
        fases = asignar_fases_relativas(arr_eval, evals_totales)
        mascara_fase = fases >= 0
        if not np.any(mascara_fase):
            continue

        fitness_runs.append(arr_fit[mascara_fase])
        fases_runs.append(fases[mascara_fase])

    if len(fitness_runs) == 0:
        return None
    return np.concatenate(fitness_runs), np.concatenate(fases_runs)

def generar_histograma_fitness_dataset_completo(filas, titulo, outpath):
    fitness_por_algoritmo = {}
    for fila in filas:
        arr_fit = cargar_fitness_dataset_desde_metricas(fila.get("ruta_metricas", ""))
        if arr_fit is None or arr_fit.size == 0:
            continue
        fitness_por_algoritmo.setdefault(fila["algoritmo"], []).append(arr_fit)

    algoritmos = ordenar_algoritmos(list(fitness_por_algoritmo.keys()))
    if len(algoritmos) == 0:
        return False

    datos_por_algoritmo = {}
    for algoritmo in algoritmos:
        series = fitness_por_algoritmo.get(algoritmo, [])
        if len(series) == 0:
            continue
        datos_por_algoritmo[algoritmo] = np.concatenate(series)

    datos_validos = [datos for datos in datos_por_algoritmo.values() if len(datos) > 0]
    if len(datos_validos) == 0:
        return False

    n_bins = max(18, int(np.sqrt(sum(len(datos) for datos in datos_validos)) // 4))
    limites_x_base = calcular_limites_histograma(datos_validos, p_inf=0.5, p_sup=95.0)
    if limites_x_base is None:
        return False
    xmin, xmax = limites_x_base
    span_x = max(float(xmax - xmin), 1e-6)
    limites_x = (xmin - 0.02 * span_x, xmax)
    bin_edges = np.linspace(limites_x[0], limites_x[1], num=n_bins + 1)

    max_conteo = 0
    for datos in datos_validos:
        datos_vis = datos[(datos >= xmin) & (datos <= xmax)]
        conteos, _ = np.histogram(datos_vis, bins=bin_edges)
        if conteos.size > 0:
            max_conteo = max(max_conteo, int(np.max(conteos)))
    limite_y = (0.0, max_conteo + max(max_conteo * 0.08, 1.0))

    fig, axes = plt.subplots(
        len(algoritmos),
        1,
        figsize=(10.4, 4.8 * len(algoritmos)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes[:, 0]

    for i, algoritmo in enumerate(algoritmos):
        ax = axes[i]
        datos = datos_por_algoritmo.get(algoritmo)
        if datos is None or len(datos) == 0:
            continue

        datos_vis = datos[(datos >= xmin) & (datos <= xmax)]
        color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")
        media = float(np.mean(datos))
        mediana = float(np.median(datos))

        ax.hist(datos_vis, bins=bin_edges, color=color, edgecolor="white", linewidth=0.8, alpha=0.82)
        superponer_curva_histograma(ax, datos_vis, bin_edges, etiqueta="curva suavizada")
        ax.axvline(media, color="#222222", linestyle="--", linewidth=1.2, label="media")
        ax.axvline(mediana, color="#222222", linestyle=":", linewidth=1.2, label="mediana")
        anotar_muestras_recortadas(ax, datos, xmin, xmax)

        ax.set_title(f"{algoritmo.upper()} (muestras={len(datos)})")
        if i == len(algoritmos) - 1:
            ax.set_xlabel("Fitness de todas las muestras del dataset")
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.25)
        aplicar_formato_ejes_compacto(ax, ejes="x")
        aplicar_limites_compartidos(ax, "x", limites_x)
        aplicar_limites_compartidos(ax, "y", limite_y)
        ax.legend(loc="best", fontsize=11)

    fig.suptitle(titulo)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_histograma_fitness_por_fases(filas, titulo, outpath):
    transformacion = obtener_transformacion_convergencia(filas)
    fitness_por_algoritmo = {}
    fases_por_algoritmo = {}

    for fila in filas:
        datos = cargar_fitness_por_fases_desde_metricas(fila.get("ruta_metricas", ""))
        if datos is None:
            continue
        arr_fit, arr_fases = datos
        if arr_fit.size == 0 or arr_fit.size != arr_fases.size:
            continue
        arr_fit_plot = aplicar_transformacion_convergencia(arr_fit, transformacion)
        fitness_por_algoritmo.setdefault(fila["algoritmo"], []).append(arr_fit_plot)
        fases_por_algoritmo.setdefault(fila["algoritmo"], []).append(arr_fases)

    algoritmos = ordenar_algoritmos(list(fitness_por_algoritmo.keys()))
    if len(algoritmos) == 0:
        return False

    n_fases = len(FASES_PREDICTOR)
    n_cols = min(3, n_fases)
    n_rows = int(np.ceil(n_fases / n_cols))
    rutas_generadas = []

    for algoritmo in algoritmos:
        series_fit = fitness_por_algoritmo.get(algoritmo, [])
        series_fases = fases_por_algoritmo.get(algoritmo, [])
        if len(series_fit) == 0 or len(series_fit) != len(series_fases):
            continue

        arr_fit = np.concatenate(series_fit)
        arr_fases = np.concatenate(series_fases)
        fitness_fases = [np.asarray(arr_fit[arr_fases == idx], dtype=float) for idx in range(n_fases)]
        series_validas = [arr for arr in fitness_fases if arr.size > 0]
        if len(series_validas) == 0:
            continue

        limites_x_base = calcular_limites_histograma(series_validas, p_inf=0.5, p_sup=95.0)
        if limites_x_base is None:
            continue
        xmin, xmax = limites_x_base
        span_x = max(float(xmax - xmin), 1e-6)
        limites_x = (xmin - 0.02 * span_x, xmax)

        n_bins = max(18, int(np.sqrt(sum(arr.size for arr in series_validas)) // 4))
        bin_edges = np.linspace(limites_x[0], limites_x[1], num=n_bins + 1)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.2 * n_cols, 2.45 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.ravel()
        color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")

        max_conteo = 0
        for arr in series_validas:
            arr_vis = arr[(arr >= xmin) & (arr <= xmax)]
            conteos, _ = np.histogram(arr_vis, bins=bin_edges)
            if conteos.size > 0:
                max_conteo = max(max_conteo, int(np.max(conteos)))
        limite_y = (0.0, max_conteo + max(max_conteo * 0.08, 1.0))

        for idx_fase, (inicio, fin, etiqueta_fase) in enumerate(FASES_PREDICTOR):
            ax = axes_flat[idx_fase]
            datos_fase = fitness_fases[idx_fase]
            if datos_fase.size > 0:
                datos_vis = datos_fase[(datos_fase >= xmin) & (datos_fase <= xmax)]
                ax.hist(
                    datos_vis,
                    bins=bin_edges,
                    color=color,
                    edgecolor="#ffffff",
                    linewidth=0.9,
                    alpha=0.90,
                )
                superponer_curva_histograma(ax, datos_vis, bin_edges)
                anotar_muestras_recortadas(ax, datos_fase, xmin, xmax)
            ax.set_title(
                f"{etiqueta_fase} ({int(inicio * 100)}-{int(fin * 100)}%, n={datos_fase.size})",
                fontsize=9,
            )
            if idx_fase // n_cols == n_rows - 1:
                ax.set_xlabel(
                    transformacion["ylabel"] if transformacion is not None else "Fitness",
                    fontsize=10,
                )
            if idx_fase % n_cols == 0:
                ax.set_ylabel("Frecuencia", fontsize=10)
            ax.grid(axis="y", alpha=0.20)
            aplicar_formato_ejes_compacto(ax, ejes="both")
            aplicar_limites_compartidos(ax, "x", limites_x)
            aplicar_limites_compartidos(ax, "y", limite_y)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(n_fases, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        titulo_algoritmo = f"{titulo} | {algoritmo.upper()}" if len(algoritmos) > 1 else titulo
        fig.suptitle(titulo_algoritmo, fontsize=11)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.86, wspace=0.22, hspace=0.36)

        ruta_panel = outpath.with_name(f"{outpath.stem}_{algoritmo}{outpath.suffix}")
        fig.savefig(ruta_panel, dpi=180)
        plt.close(fig)
        rutas_generadas.append(ruta_panel)

    return [str(r) for r in rutas_generadas]

def generar_curva_convergencia(filas, titulo, outpath):
    transformacion = obtener_transformacion_convergencia(filas)
    filas_por_algoritmo = {}
    for fila in filas:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    orden_algoritmos = ordenar_algoritmos(list(filas_por_algoritmo.keys()))
    series_por_algoritmo = []

    for algoritmo in orden_algoritmos:
        curvas_mejor = []
        curvas_promedio = []
        for fila in filas_por_algoritmo[algoritmo]:
            curva_mejor = cargar_curva_desde_logbook(fila.get("ruta_metricas", ""))
            curva_promedio = cargar_curva_promedio_desde_logbook(fila.get("ruta_metricas", ""))
            if curva_mejor is None or curva_promedio is None:
                continue
            curvas_mejor.append(curva_mejor)
            curvas_promedio.append(curva_promedio)

        if len(curvas_mejor) == 0:
            continue

        x_mejor, y_mejor_runs = alinear_curvas_por_evaluaciones(curvas_mejor)
        x_promedio, y_promedio_runs = alinear_curvas_por_evaluaciones(curvas_promedio)
        mediana_mejor = np.median(y_mejor_runs, axis=0)
        media_promedio = np.mean(y_promedio_runs, axis=0)

        mediana_mejor_plot = aplicar_transformacion_convergencia(mediana_mejor, transformacion)
        media_promedio_plot = aplicar_transformacion_convergencia(media_promedio, transformacion)

        color = COLORES_ALGORITMO.get(algoritmo, "#59a14f")
        nombre = algoritmo.upper()

        series_por_algoritmo.append(
            {
                "algoritmo": algoritmo,
                "nombre": nombre,
                "color": color,
                "n_curvas": len(curvas_mejor),
                "x_mejor": x_mejor,
                "mediana_mejor_plot": mediana_mejor_plot,
                "x_promedio": x_promedio,
                "media_promedio_plot": media_promedio_plot,
            }
        )

    if len(series_por_algoritmo) == 0:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    max_eval = calcular_maximo_evaluaciones(
        [serie["x_mejor"] for serie in series_por_algoritmo]
        + [serie["x_promedio"] for serie in series_por_algoritmo]
    )
    if max_eval is None:
        return False

    limites_x = (0, max_eval)
    limites_y = calcular_limites_compartidos(
        [serie["mediana_mejor_plot"] for serie in series_por_algoritmo]
        + [serie["media_promedio_plot"] for serie in series_por_algoritmo],
        padding_ratio=0.05,
        padding_min=1e-6,
        limites_forzados=(0, None),
    )

    for serie in series_por_algoritmo:
        color = serie["color"]

        ax.step(
            serie["x_mejor"],
            serie["mediana_mejor_plot"],
            where="post",
            color=color,
            linewidth=1.8,
            label=f"{serie['nombre']} mejor mediana",
        )
        ax.step(
            serie["x_promedio"],
            serie["media_promedio_plot"],
            where="post",
            color=color,
            linestyle="--",
            linewidth=1.3,
            alpha=0.95,
            label=f"{serie['nombre']} promedio poblacional",
        )

    ax.set_xlabel("Evaluaciones", fontsize=11)
    if transformacion is None:
        ax.set_ylabel("Fitness", fontsize=11)
    else:
        ax.set_ylabel(transformacion["ylabel"], fontsize=11)
    ax.grid(alpha=0.25)
    aplicar_formato_ejes_compacto(ax, ejes="both")
    aplicar_limites_compartidos(ax, "x", limites_x)
    aplicar_limites_compartidos(ax, "y", limites_y)
    ax.margins(x=0, y=0)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=9, ncols=2)

    fig.suptitle(titulo, fontsize=11)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_tabla_resultados_cec_estandar(filas, outpath):
    filas_cec = [fila for fila in filas if fila.get("problema") == "cec2017"]
    if len(filas_cec) == 0:
        return False

    funcids = sorted({fila.get("cec_funcid") for fila in filas_cec if fila.get("cec_funcid") is not None})
    if len(funcids) != 1:
        return False
    funcid = int(funcids[0])

    filas_por_algoritmo = {}
    for fila in filas_cec:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(float(fila["fitness"]))

    if len(filas_por_algoritmo) == 0:
        return False

    resumen = []
    for algoritmo in ordenar_algoritmos(list(filas_por_algoritmo.keys())):
        valores = np.asarray(filas_por_algoritmo[algoritmo], dtype=float)
        if valores.size == 0:
            continue
        resumen.append(
            {
                "F": funcid,
                "Algoritmo": algoritmo.upper(),
                "Media": float(np.mean(valores)),
                "Std": float(np.std(valores)),
                "Mediana": float(np.median(valores)),
                "Mejor": float(np.min(valores)),
                "Peor": float(np.max(valores)),
            }
        )

    if len(resumen) == 0:
        return False

    orden_indices = sorted(range(len(resumen)), key=lambda idx: (resumen[idx]["Media"], resumen[idx]["Algoritmo"]))
    rank_por_idx = {}
    rank_actual = 1
    for idx in orden_indices:
        rank_por_idx[idx] = rank_actual
        rank_actual += 1
    for idx, fila in enumerate(resumen):
        fila["Rank"] = rank_por_idx[idx]

    outpath.parent.mkdir(parents=True, exist_ok=True)
    campos = ["F", "Algoritmo", "Media", "Std", "Mediana", "Mejor", "Peor", "Rank"]
    with outpath.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=campos, lineterminator="\n")
        writer.writeheader()
        writer.writerows(resumen)
    return True


def generar_curvas_convergencia_por_variante(funcid, outdir, sufijo_base, plots_seleccionados=None):
    if plots_seleccionados is not None and "convergencia_variante" not in plots_seleccionados:
        return []

    filas_variantes = cargar_filas_variantes_cec_por_funcion(funcid, algoritmo="todos")
    if not filas_variantes:
        return []

    por_variante = {}
    for fila in filas_variantes:
        clave = fila.get("variante_reinicio", "sin_reinicio")
        por_variante.setdefault(clave, []).append(fila)

    rutas = []
    for variante in VARIANTES_REINICIO:
        clave = variante["clave"]
        filas_v = por_variante.get(clave)
        if not filas_v:
            continue
        label = variante["label"]
        outpath = outdir / f"curva_convergencia_variante_{clave}{sufijo_base}.png"
        if generar_curva_convergencia(filas=filas_v, titulo=f"Convergencia — {label}", outpath=outpath):
            rutas.append(outpath)
    return rutas


def generar_curvas_convergencia_reinicio(filas, problema, outpath_base, agregacion="representativa"):
    if problema != "cec2017":
        return []
    if agregacion not in {"representativa", "promedio"}:
        raise ValueError(
            "agregacion debe ser 'representativa' o 'promedio' en generar_curvas_convergencia_reinicio"
        )

    funcids = sorted({int(fila["cec_funcid"]) for fila in filas if fila.get("cec_funcid") is not None})
    if len(funcids) != 1:
        return []

    funcid = int(funcids[0])
    transformacion = obtener_transformacion_convergencia(filas)
    algoritmos_objetivo = ordenar_algoritmos([fila["algoritmo"] for fila in filas])
    semillas_objetivo = {
        int(fila["semilla"])
        for fila in filas
        if fila.get("semilla") is not None
    }

    filas_variantes = cargar_filas_variantes_cec_por_funcion(funcid, algoritmo="todos")
    filas_variantes.extend(
        fila
        for fila in filas
        if fila.get("cec_funcid") is not None and int(fila["cec_funcid"]) == funcid
    )

    filas_unicas = {}
    for fila in filas_variantes:
        clave = (
            str(fila.get("ruta_metricas", "")),
            str(fila.get("algoritmo", "")),
            fila.get("cec_funcid"),
            fila.get("semilla"),
            fila.get("variante_reinicio"),
        )
        filas_unicas[clave] = fila
    filas_variantes = list(filas_unicas.values())

    if algoritmos_objetivo:
        filas_variantes = [f for f in filas_variantes if f.get("algoritmo") in set(algoritmos_objetivo)]
    if semillas_objetivo:
        filas_variantes = [
            f for f in filas_variantes
            if f.get("semilla") is not None and int(f["semilla"]) in semillas_objetivo
        ]
    if not filas_variantes:
        return []

    filas_por_algoritmo = {}
    for fila in filas_variantes:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    rutas_generadas = []

    for algoritmo in ordenar_algoritmos(list(filas_por_algoritmo.keys())):
        filas_alg = filas_por_algoritmo[algoritmo]
        filas_por_variante = {}
        for fila in filas_alg:
            clave = fila.get("variante_reinicio")
            if clave is not None:
                filas_por_variante.setdefault(clave, []).append(fila)

        if agregacion == "representativa":
            semilla_rep = _seleccionar_semilla_representativa(filas_por_variante)
            if semilla_rep is None:
                continue
            if semillas_objetivo:
                semilla_rep = min(semillas_objetivo)
        else:
            semilla_rep = None

        series = {}
        for variante in variantes_reinicio_activas():
            curvas = []
            reinicios = []
            for fila in filas_por_variante.get(variante["clave"], []):
                if semilla_rep is not None and (
                    fila.get("semilla") is None or int(fila["semilla"]) != int(semilla_rep)
                ):
                    continue
                curva = cargar_curva_desde_logbook(fila.get("ruta_metricas", ""))
                if curva is not None:
                    curvas.append(curva)
                    eval_antes, _ = _cargar_reinicios_elitistas(fila.get("ruta_metricas", ""))
                    if eval_antes.size > 0:
                        reinicios.extend(eval_antes.tolist())

            if not curvas:
                continue

            if len(curvas) == 1:
                x, y = curvas[0]
            else:
                x, y_runs = alinear_curvas_por_evaluaciones(curvas)
                y = np.median(y_runs, axis=0)
            series[variante["clave"]] = {
                "x": x,
                "y": aplicar_transformacion_convergencia(y, transformacion),
                "color": (
                    COLORES_ALGORITMO.get(algoritmo, variante["color"])
                    if variante["clave"] == "sin_reinicio"
                    else variante["color"]
                ),
                "label": etiqueta_variante_reinicio_algoritmo(algoritmo, variante["clave"]),
                "reinicios": sorted(set(int(ev) for ev in reinicios)),
            }

        if not series:
            continue

        fig, ax = plt.subplots(figsize=(6, 3.5))
        todas_x = [s["x"] for s in series.values()]
        todas_y = [s["y"] for s in series.values()]
        limites_x = (0, calcular_maximo_evaluaciones(todas_x) or 100000)
        limites_y = calcular_limites_compartidos(
            todas_y,
            padding_ratio=0.05,
            padding_min=1e-6,
            limites_forzados=(0, None),
        )

        for variante in variantes_reinicio_activas():
            s = series.get(variante["clave"])
            if s is None:
                continue
            ax.step(s["x"], s["y"], where="post", color=s["color"], linewidth=1.8, label=s["label"])
            _dibujar_hitos_reinicio(
                ax,
                s.get("reinicios", []),
                s["color"],
                f"Hitos {s['label']}",
            )

        ax.set_xlabel("Evaluaciones", fontsize=11)
        ax.set_ylabel(transformacion["ylabel"] if transformacion is not None else "Fitness", fontsize=11)
        ax.grid(alpha=0.20)
        aplicar_formato_ejes_compacto(ax, ejes="both")
        aplicar_limites_compartidos(ax, "x", limites_x)
        aplicar_limites_compartidos(ax, "y", limites_y)
        ax.margins(x=0, y=0)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=9)

        if agregacion == "promedio":
            fig.suptitle(f"Convergencia de {algoritmo.upper()} en CEC2017 f{funcid}", fontsize=11)
        else:
            fig.suptitle(
                f"Convergencia de {algoritmo.upper()} en CEC2017 f{funcid}, semilla {semilla_rep}",
                fontsize=11,
            )
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)

        sufijo_agregacion = "_promedio" if agregacion == "promedio" else ""
        outpath_alg = outpath_base.with_name(
            f"{outpath_base.stem}{sufijo_agregacion}_{algoritmo}{outpath_base.suffix}"
        )
        outpath_alg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath_alg, dpi=180)
        plt.close(fig)
        rutas_generadas.append(str(outpath_alg))

    return rutas_generadas


def generar_curva_diversidad(filas, problema, titulo, outpath):
    metricas = DIVERSIDAD_POR_PROBLEMA.get(problema, ())
    if len(metricas) == 0:
        return False
    # Mantener el estilo de "convergencia": un único panel con la curva promedio por algoritmo.
    # Si hubiese múltiples métricas configuradas para el problema, por consistencia nos quedamos con la primera.
    metrica, etiqueta_y = metricas[0]

    filas_por_algoritmo = {}
    for fila in filas:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    orden_algoritmos = ordenar_algoritmos(list(filas_por_algoritmo.keys()))
    if len(orden_algoritmos) == 0:
        return False

    series_por_algoritmo = []
    series_x = []
    series_y = []

    for algoritmo in orden_algoritmos:
        curvas = []
        for fila in filas_por_algoritmo[algoritmo]:
            curva = cargar_curva_metrica_desde_logbook(fila.get("ruta_metricas", ""), metrica)
            if curva is None:
                continue
            curvas.append(curva)

        if len(curvas) == 0:
            continue

        x, y_runs = alinear_curvas_por_evaluaciones(curvas)
        media = np.mean(y_runs, axis=0)

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
            }
        )
        series_x.append(x)
        series_y.append(media)

    if len(series_por_algoritmo) == 0:
        return False

    max_eval = calcular_maximo_evaluaciones(series_x)
    if max_eval is None:
        return False

    limites_x = (0, max_eval)
    limites_y = calcular_limites_compartidos(
        series_y,
        padding_ratio=0.05,
        padding_min=1e-6,
        limites_forzados=(0, None),
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    for serie in series_por_algoritmo:
        ax.step(
            serie["x"],
            serie["media"],
            where="post",
            color=serie["color"],
            linewidth=1.8,
            label=f"{serie['nombre']} promedio",
        )

    ax.set_xlabel("Evaluaciones", fontsize=11)
    ax.set_ylabel(etiqueta_y, fontsize=11)
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    aplicar_formato_ejes_compacto(ax, ejes="y")
    aplicar_limites_compartidos(ax, "x", limites_x)
    aplicar_limites_compartidos(ax, "y", limites_y)
    ax.margins(x=0, y=0)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=9, ncols=2)

    fig.suptitle(titulo, fontsize=11)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_curva_conjunta_fitness_diversidad(filas, problema, titulo, outpath):
    metricas_base = DIVERSIDAD_POR_PROBLEMA.get(problema, ())
    if len(metricas_base) == 0:
        return False
    metricas = [resolver_metrica_diversidad_para_curva_conjunta(metrica, etiqueta) for metrica, etiqueta in metricas_base]

    filas_por_algoritmo = {}
    for fila in filas:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    orden_algoritmos = ordenar_algoritmos(list(filas_por_algoritmo.keys()))
    if len(orden_algoritmos) == 0:
        return False

    hay_grafica = False
    paneles_por_algoritmo = {}
    series_x_por_metrica = {metrica: [] for metrica, _ in metricas}
    series_fit_por_metrica = {metrica: [] for metrica, _ in metricas}
    series_div_por_metrica = {metrica: [] for metrica, _ in metricas}

    color_fitness = "#1f77b4"
    color_fitness_promedio = "#2ca02c"
    color_diversidad = "#ff7f0e"

    for algoritmo in orden_algoritmos:
        paneles = []
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
            series_x_por_metrica[metrica].append(x)
            series_fit_por_metrica[metrica].extend([y_fit, y_fit_promedio])
            series_div_por_metrica[metrica].append(y_div)
            hay_grafica = True

        paneles_por_algoritmo[algoritmo] = paneles

    if not hay_grafica:
        return False

    limites_x_por_metrica = {}
    limites_fit_y_por_metrica = {}
    limites_div_y_por_metrica = {}
    for metrica, _ in metricas:
        limites_x_por_metrica[metrica] = calcular_limites_compartidos(
            series_x_por_metrica[metrica], padding_ratio=0.02, padding_min=1.0
        )
        limites_fit_y_por_metrica[metrica] = calcular_limites_compartidos(
            series_fit_por_metrica[metrica], padding_ratio=0.05, padding_min=1e-6
        )
        if metrica in METRICAS_NORMALIZADAS:
            limites_div_y_por_metrica[metrica] = (-0.05, 1.05)
        else:
            limites_div_y_por_metrica[metrica] = calcular_limites_compartidos(
                series_div_por_metrica[metrica], padding_ratio=0.05, padding_min=1e-6
            )

    fig, axes = plt.subplots(
        len(orden_algoritmos),
        len(metricas),
        figsize=(8.8 * len(metricas), 5.4 * len(orden_algoritmos)),
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
            aplicar_limites_compartidos(ax, "x", limites_x_por_metrica[panel["metrica"]])
            aplicar_limites_compartidos(ax, "y", limites_fit_y_por_metrica[panel["metrica"]])

            ax2.set_ylabel("Diversidad", color=color_diversidad)
            ax2.tick_params(axis="y", labelcolor=color_diversidad)
            aplicar_limites_compartidos(ax2, "y", limites_div_y_por_metrica[panel["metrica"]])

            ax.legend(
                [linea_fit, linea_fit_promedio, linea_div],
                ["Mejor fitness", "Fitness promedio", "Diversidad"],
                loc="best",
                fontsize=8,
            )

    fig.suptitle(titulo)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.88, wspace=0.34, hspace=0.36)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def _cargar_reinicios_elitistas(ruta_metricas):
    csv_reinicio = Path(str(ruta_metricas)) / "reinicios_elitistas.csv"
    if not csv_reinicio.exists():
        return np.array([], dtype=int), np.array([], dtype=float)
    try:
        df = pd.read_csv(csv_reinicio)
        col_eval = "evaluaciones_antes_reinicio"
        col_div  = METRICA_DIVERSIDAD_CEC
        if col_eval not in df.columns or col_div not in df.columns:
            return np.array([], dtype=int), np.array([], dtype=float)
        return (
            np.asarray(df[col_eval], dtype=int),
            np.asarray(df[col_div],  dtype=float),
        )
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=float)


def _dibujar_hitos_reinicio(ax, evaluaciones_reinicio, color, label):
    if evaluaciones_reinicio is None:
        return False
    evals = np.asarray(evaluaciones_reinicio, dtype=float).reshape(-1)
    evals = evals[np.isfinite(evals)]
    if evals.size == 0:
        return False
    for idx, ev in enumerate(np.unique(evals.astype(int))):
        ax.axvline(
            int(ev),
            color=color,
            linestyle=":",
            linewidth=0.8,
            alpha=0.45,
            label=label if idx == 0 else None,
            zorder=0,
        )
    return True


def _contar_reinicios_run(ruta_metricas, fila=None):
    if fila is not None:
        valor = fila.get("n_reinicios_elitistas")
        if valor not in (None, ""):
            return int(valor)
    eval_antes, _ = _cargar_reinicios_elitistas(ruta_metricas)
    return int(eval_antes.size)


def _detectar_funciones_cec_variantes():
    funciones_por_variante = []
    for variante in variantes_reinicio_activas():
        root_variante = resolver_directorio_variante_cec(variante["clave"])
        if root_variante is None:
            continue
        funciones = {
            int(path.name[1:])
            for path in root_variante.iterdir()
            if path.is_dir() and path.name.startswith("f") and (path / "runs.csv").exists()
        }
        if funciones:
            funciones_por_variante.append(funciones)
    if not funciones_por_variante:
        return []
    funciones_comunes = set.intersection(*funciones_por_variante)
    return sorted(funciones_comunes)


def generar_barras_reinicios_medios_cec(filas, problema, outdir_base):
    if problema != "cec2017":
        return []

    funcids = sorted({int(fila["cec_funcid"]) for fila in filas if fila.get("cec_funcid") is not None})
    if len(funcids) <= 1:
        funcids = _detectar_funciones_cec_variantes()
    if not funcids:
        return []

    filas_totales = []
    for funcid in funcids:
        filas_totales.extend(cargar_filas_variantes_cec_por_funcion(funcid, algoritmo="todos"))
    filas_totales.extend(
        fila
        for fila in filas
        if fila.get("cec_funcid") is not None and int(fila["cec_funcid"]) in set(funcids)
    )
    filas_unicas = {}
    for fila in filas_totales:
        clave = (
            str(fila.get("ruta_metricas", "")),
            str(fila.get("algoritmo", "")),
            fila.get("cec_funcid"),
            fila.get("semilla"),
            fila.get("variante_reinicio"),
        )
        filas_unicas[clave] = fila
    filas_totales = list(filas_unicas.values())
    if not filas_totales:
        return []

    algoritmos_objetivo = ordenar_algoritmos([fila["algoritmo"] for fila in filas])
    if algoritmos_objetivo:
        filas_totales = [fila for fila in filas_totales if fila.get("algoritmo") in set(algoritmos_objetivo)]
    if not filas_totales:
        return []

    resumen = {}
    for fila in filas_totales:
        algoritmo = fila.get("algoritmo")
        variante = fila.get("variante_reinicio")
        funcid = fila.get("cec_funcid")
        if algoritmo is None or variante is None or funcid is None:
            continue
        clave = (str(algoritmo), int(funcid), str(variante))
        resumen.setdefault(clave, []).append(
            _contar_reinicios_run(fila.get("ruta_metricas", ""), fila=fila)
        )

    if not resumen:
        return []

    outdir_base = Path(outdir_base)
    outdir_base.mkdir(parents=True, exist_ok=True)
    rutas_generadas = []

    posiciones = np.arange(len(funcids), dtype=float)
    etiquetas_x = [f"f{int(funcid)}" for funcid in funcids]

    algoritmos_presentes = ordenar_algoritmos(sorted({clave[0] for clave in resumen}))
    for algoritmo in algoritmos_presentes:
        fig, ax = plt.subplots(figsize=(11.5, 5.5))
        hay_barras = False
        variantes_presentes = [
            variante
            for variante in variantes_reinicio_activas()
            if variante["clave"] != "sin_reinicio"
            and any(
                resumen.get((algoritmo, int(funcid), variante["clave"]), [])
                for funcid in funcids
            )
        ]
        if not variantes_presentes:
            plt.close(fig)
            continue
        ancho = min(0.8 / max(len(variantes_presentes), 1), 0.22)
        centro = (len(variantes_presentes) - 1) / 2.0

        for idx_variante, variante in enumerate(variantes_presentes):
            valores = []
            for funcid in funcids:
                conteos = resumen.get((algoritmo, int(funcid), variante["clave"]), [])
                if conteos:
                    valores.append(float(np.mean(np.asarray(conteos, dtype=float))))
                else:
                    valores.append(np.nan)

            valores_arr = np.asarray(valores, dtype=float)
            mascara = np.isfinite(valores_arr)
            if not np.any(mascara):
                continue

            x_variante = posiciones[mascara] + (idx_variante - centro) * ancho
            bars = ax.bar(
                x_variante,
                valores_arr[mascara],
                width=ancho,
                color=variante["color"],
                alpha=0.88,
                label=variante["label"],
            )
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
            hay_barras = True

        if not hay_barras:
            plt.close(fig)
            continue

        ax.set_xticks(posiciones)
        ax.set_xticklabels(etiquetas_x)
        ax.set_xlabel("Función")
        ax.set_ylabel("Nº medio de reinicios")
        ax.set_title(f"{algoritmo.upper()} | Reinicios medios por función")
        ax.grid(axis="y", alpha=0.25)
        aplicar_formato_ejes_compacto(ax, ejes="y")
        ax.legend(loc="upper right", fontsize=9)

        fig.tight_layout()
        outpath = outdir_base / f"reinicios_medios_por_funcion_{algoritmo}_cec2017.png"
        fig.savefig(outpath, dpi=180)
        plt.close(fig)
        rutas_generadas.append(str(outpath))

    return rutas_generadas


def _reconstruir_curva_con_reinicios(ruta_metricas):
    return cargar_curva_metrica_desde_logbook(
        ruta_metricas, METRICA_DIVERSIDAD_CEC, x_column="evaluaciones"
    )


def _seleccionar_semilla_representativa(filas_por_variante):
    """Seed with restart count closest to median in the most informative variant."""
    clave_referencia = next(
        (
            clave
            for clave in (
                "reinicio_pat001",
                "reinicio_pat003",
                "reinicio_pat005",
                "reinicio_pat007",
                "reinicio_pat010",
                "reinicio_005",
                "reinicio_010",
            )
            if filas_por_variante.get(clave)
        ),
        None,
    )
    if clave_referencia is None:
        return None
    filas_referencia = filas_por_variante.get(clave_referencia, [])

    semillas_por_variante = {
        clave: {int(f["semilla"]) for f in fs if f.get("semilla") is not None}
        for clave, fs in filas_por_variante.items()
    }
    semillas_comunes = None
    for semillas in semillas_por_variante.values():
        semillas_comunes = semillas if semillas_comunes is None else semillas_comunes & semillas
    if not semillas_comunes:
        return None

    n_reinicios_por_semilla = {}
    for fila in filas_referencia:
        semilla = fila.get("semilla")
        if semilla is None or int(semilla) not in semillas_comunes:
            continue
        eval_antes, _ = _cargar_reinicios_elitistas(fila.get("ruta_metricas", ""))
        n_reinicios_por_semilla[int(semilla)] = int(eval_antes.size)

    if not n_reinicios_por_semilla:
        return next(iter(sorted(semillas_comunes)))

    conteos = np.array([n_reinicios_por_semilla[s] for s in sorted(n_reinicios_por_semilla)])
    mediana = float(np.median(conteos))
    semilla_rep = min(
        sorted(n_reinicios_por_semilla),
        key=lambda s: (abs(n_reinicios_por_semilla[s] - mediana), s),
    )
    return semilla_rep


def generar_curvas_diversidad_reinicio(filas, problema, outpath_base, agregacion="representativa"):
    if problema != "cec2017":
        return []
    if agregacion not in {"representativa", "promedio"}:
        raise ValueError(
            "agregacion debe ser 'representativa' o 'promedio' en generar_curvas_diversidad_reinicio"
        )

    funcids = sorted({int(fila["cec_funcid"]) for fila in filas if fila.get("cec_funcid") is not None})
    if len(funcids) != 1:
        return []

    algoritmos_objetivo = ordenar_algoritmos([fila["algoritmo"] for fila in filas])
    semillas_objetivo = {
        int(fila["semilla"])
        for fila in filas
        if fila.get("semilla") is not None
    }
    filas_variantes = cargar_filas_variantes_cec_por_funcion(funcids[0], algoritmo="todos")
    filas_variantes.extend(
        fila
        for fila in filas
        if fila.get("cec_funcid") is not None and int(fila["cec_funcid"]) == int(funcids[0])
    )
    filas_unicas = {}
    for fila in filas_variantes:
        clave = (
            str(fila.get("ruta_metricas", "")),
            str(fila.get("algoritmo", "")),
            fila.get("cec_funcid"),
            fila.get("semilla"),
            fila.get("variante_reinicio"),
        )
        filas_unicas[clave] = fila
    filas_variantes = list(filas_unicas.values())
    if algoritmos_objetivo:
        filas_variantes = [f for f in filas_variantes if f.get("algoritmo") in set(algoritmos_objetivo)]
    if semillas_objetivo:
        filas_variantes = [
            f for f in filas_variantes
            if f.get("semilla") is not None and int(f["semilla"]) in semillas_objetivo
        ]
    if not filas_variantes:
        return []

    filas_por_algoritmo = {}
    for fila in filas_variantes:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    rutas_generadas = []

    for algoritmo in ordenar_algoritmos(list(filas_por_algoritmo.keys())):
        filas_alg = filas_por_algoritmo[algoritmo]

        filas_por_variante = {}
        for fila in filas_alg:
            clave = fila.get("variante_reinicio")
            if clave is not None:
                filas_por_variante.setdefault(clave, []).append(fila)

        series = {}
        n_reinicios_por_variante = {}

        if agregacion == "representativa":
            semilla_rep = _seleccionar_semilla_representativa(filas_por_variante)
            if semilla_rep is None:
                continue

            for variante in variantes_reinicio_activas():
                candidatas_seed = [
                    f for f in filas_por_variante.get(variante["clave"], [])
                    if f.get("semilla") is not None and int(f["semilla"]) == int(semilla_rep)
                ]
                fila = next(
                    (
                        f for f in candidatas_seed
                        if str(f.get("ruta_metricas", "")).strip()
                    ),
                    candidatas_seed[0] if candidatas_seed else None,
                )
                if fila is None:
                    continue
                ruta = fila.get("ruta_metricas", "")
                if variante["umbral"] is None:
                    curva = cargar_curva_metrica_desde_logbook(
                        ruta, METRICA_DIVERSIDAD_CEC, x_column="evaluaciones"
                    )
                    n_reinicios_por_variante[variante["clave"]] = 0
                    eval_antes = np.array([], dtype=int)
                else:
                    curva = _reconstruir_curva_con_reinicios(ruta)
                    eval_antes, _ = _cargar_reinicios_elitistas(ruta)
                    n_reinicios_por_variante[variante["clave"]] = int(eval_antes.size)
                if curva is None:
                    continue
                series[variante["clave"]] = {
                    "x": curva[0],
                    "y": curva[1],
                    "n_curvas": 1,
                    "color": (
                        COLORES_ALGORITMO.get(algoritmo, variante["color"])
                        if variante["clave"] == "sin_reinicio"
                        else variante["color"]
                    ),
                    "umbral": variante["umbral"],
                    "reinicios": eval_antes,
                    "label_variante": etiqueta_variante_reinicio_algoritmo(
                        algoritmo, variante["clave"]
                    ),
                }
        else:
            semilla_rep = None
            for variante in variantes_reinicio_activas():
                curvas = []
                conteos_reinicios = []
                reinicios = []
                for fila in filas_por_variante.get(variante["clave"], []):
                    ruta = fila.get("ruta_metricas", "")
                    if variante["umbral"] is None:
                        curva = cargar_curva_metrica_desde_logbook(
                            ruta, METRICA_DIVERSIDAD_CEC, x_column="evaluaciones"
                        )
                        conteos_reinicios.append(0)
                    else:
                        curva = _reconstruir_curva_con_reinicios(ruta)
                        eval_antes, _ = _cargar_reinicios_elitistas(ruta)
                        conteos_reinicios.append(int(eval_antes.size))
                        if eval_antes.size > 0:
                            reinicios.extend(eval_antes.tolist())
                    if curva is not None:
                        curvas.append(curva)

                if not curvas:
                    continue

                x, y_media = promediar_curvas_en_malla_regular(curvas)
                n_reinicios_por_variante[variante["clave"]] = (
                    float(np.mean(np.asarray(conteos_reinicios, dtype=float)))
                    if conteos_reinicios
                    else 0.0
                )
                series[variante["clave"]] = {
                    "x": x,
                    "y": y_media,
                    "n_curvas": len(curvas),
                    "color": (
                        COLORES_ALGORITMO.get(algoritmo, variante["color"])
                        if variante["clave"] == "sin_reinicio"
                        else variante["color"]
                    ),
                    "umbral": variante["umbral"],
                    "reinicios": sorted(set(int(ev) for ev in reinicios)),
                    "label_variante": etiqueta_variante_reinicio_algoritmo(
                        algoritmo, variante["clave"]
                    ),
                }

        if not series:
            continue

        fig, ax = plt.subplots(figsize=(6, 3.5))

        todas_y = [s["y"] for s in series.values()]
        y_max_data = max(float(np.nanmax(y)) for y in todas_y if len(y) > 0)
        umbrales_activos = [s["umbral"] for s in series.values() if s["umbral"] is not None]
        y_max_ref = max([y_max_data] + umbrales_activos) if umbrales_activos else y_max_data
        y_max_div = y_max_ref + max(y_max_ref * 0.08, 0.005)

        for variante in variantes_reinicio_activas():
            s = series.get(variante["clave"])
            if s is None:
                continue
            n_r = n_reinicios_por_variante.get(variante["clave"], 0)
            label = s["label_variante"]
            ax.plot(s["x"], s["y"], color=s["color"], linewidth=1.8, label=label)
            _dibujar_hitos_reinicio(
                ax,
                s.get("reinicios", []),
                s["color"],
                f"Hitos {label}",
            )

        ax.set_xlim(0, 100000)
        ax.set_ylim(0.0, y_max_div)
        ax.set_xlabel("Evaluaciones", fontsize=11)
        ax.set_ylabel(ETIQUETA_DIVERSIDAD_CEC, fontsize=11)
        ax.grid(alpha=0.20)
        aplicar_formato_ejes_compacto(ax, ejes="both")
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9)

        if agregacion == "promedio":
            fig.suptitle(
                f"{algoritmo.upper()} | CEC2017 f{funcids[0]} — "
                "Diversidad por variante de reinicio",
                fontsize=11,
            )
        else:
            fig.suptitle(
                f"{algoritmo.upper()} | CEC2017 f{funcids[0]} — "
                f"Diversidad por variante de reinicio, semilla {semilla_rep}",
                fontsize=11,
            )
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)

        sufijo_agregacion = "_promedio" if agregacion == "promedio" else ""
        outpath_alg = outpath_base.with_name(
            f"{outpath_base.stem}{sufijo_agregacion}_{algoritmo}{outpath_base.suffix}"
        )
        outpath_alg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath_alg, dpi=180)
        plt.close(fig)
        rutas_generadas.append(str(outpath_alg))

    return rutas_generadas


def generar_curvas_diversidad_reinicio_gnuplot(filas, problema, outpath_base):
    if problema != "cec2017":
        return []

    gnuplot_bin = shutil.which("gnuplot")
    if gnuplot_bin is None:
        print(
            "AVISO: no se genera 'diversidad_reinicio' porque gnuplot no esta disponible en el entorno.",
            file=sys.stderr,
        )
        return []

    funcids = sorted({int(fila["cec_funcid"]) for fila in filas if fila.get("cec_funcid") is not None})
    if len(funcids) != 1:
        return []

    algoritmos_objetivo = ordenar_algoritmos([fila["algoritmo"] for fila in filas])
    filas_variantes = cargar_filas_variantes_cec_por_funcion(funcids[0], algoritmo="todos")
    filas_variantes.extend(
        fila
        for fila in filas
        if fila.get("cec_funcid") is not None and int(fila["cec_funcid"]) == int(funcids[0])
    )
    filas_unicas = {}
    for fila in filas_variantes:
        clave = (
            str(fila.get("ruta_metricas", "")),
            str(fila.get("algoritmo", "")),
            fila.get("cec_funcid"),
            fila.get("semilla"),
            fila.get("variante_reinicio"),
        )
        filas_unicas[clave] = fila
    filas_variantes = list(filas_unicas.values())

    if len(algoritmos_objetivo) > 0:
        filas_variantes = [fila for fila in filas_variantes if fila.get("algoritmo") in set(algoritmos_objetivo)]
    if len(filas_variantes) == 0:
        return []

    filas_por_algoritmo = {}
    for fila in filas_variantes:
        filas_por_algoritmo.setdefault(fila["algoritmo"], []).append(fila)

    descripcion_bloque = describir_bloque_problema(filas_variantes, problema)
    rutas_generadas = []

    for algoritmo in ordenar_algoritmos(list(filas_por_algoritmo.keys())):
        series_por_variante = {}
        for variante in variantes_reinicio_activas():
            curvas = []
            for fila in filas_por_algoritmo[algoritmo]:
                if fila.get("variante_reinicio") != variante["clave"]:
                    continue
                curva = cargar_curva_metrica_desde_logbook(
                    fila.get("ruta_metricas", ""),
                    METRICA_DIVERSIDAD_CEC,
                    x_column="evaluaciones",
                )
                if curva is None:
                    continue
                curvas.append(curva)

            if len(curvas) == 0:
                continue

            x, y_runs = alinear_curvas_por_evaluaciones(curvas)
            series_por_variante[variante["clave"]] = {
                "x": x,
                "y": np.mean(y_runs, axis=0),
                "n_curvas": len(curvas),
            }

        if len(series_por_variante) == 0:
            continue

        malla = np.unique(
            np.concatenate([serie["x"] for serie in series_por_variante.values()])
        )
        malla = np.asarray(malla, dtype=int)
        malla = malla[malla > 0]
        if malla.size == 0:
            continue

        columnas = {"evaluaciones": malla}
        for variante in variantes_reinicio_activas():
            serie = series_por_variante.get(variante["clave"])
            if serie is None:
                columnas[variante["clave"]] = np.full(malla.shape, np.nan, dtype=float)
            else:
                columnas[variante["clave"]] = alinear_serie_a_malla(serie["x"], serie["y"], malla)

        df_plot = pd.DataFrame(columnas)
        if not any(np.any(np.isfinite(df_plot[variante["clave"]].to_numpy(dtype=float))) for variante in variantes_reinicio_activas()):
            continue

        outpath_alg = outpath_base.with_name(f"{outpath_base.stem}_{algoritmo}{outpath_base.suffix}")

        with tempfile.TemporaryDirectory(prefix="mhs_diversidad_reinicio_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            ruta_datos = tmpdir_path / "diversidad.tsv"
            ruta_script = tmpdir_path / "plot.gnuplot"
            df_plot.to_csv(ruta_datos, sep="\t", index=False, na_rep="")

            style_lines = []
            plot_terms = []
            for idx_variante, variante in enumerate(variantes_reinicio_activas(), start=1):
                dt = variante["dashtype"]
                dt_gnuplot = "solid" if dt == "solid" else str(dt)
                style_lines.append(
                    f'set style line {idx_variante} lc rgb "{variante["color"]}" lw 2.0 dt {dt_gnuplot}'
                )
                plot_terms.append(
                    f'"{ruta_datos.as_posix()}" using 1:{idx_variante + 1} '
                    f'with lines ls {idx_variante} title "{variante["label"]}"'
                )

            plot_cmd = ", \\\n+     ".join(plot_terms)

            script = f"""
set terminal pngcairo size 1800,1100 enhanced font "Arial,18"
set encoding utf8
set output "{outpath_alg.as_posix()}"
set datafile separator "\\t"
unset title
set xlabel "Evaluaciones"
set ylabel "Diversidad normalizada"
set logscale x
set border 15 lc rgb "#000000" lw 1.2
set tics in
set xtics nomirror
set ytics nomirror
set x2tics
set y2tics
set format x2 ""
set format y2 ""
unset grid
set mxtics 10
set mytics 2
set key right bottom reverse noinvert samplen 3 spacing 1.0
set yrange [0:*]
{chr(10).join(style_lines)}
plot {plot_cmd}
"""
            ruta_script.write_text(script.strip() + "\n", encoding="utf-8")

            try:
                subprocess.run(
                    [gnuplot_bin, str(ruta_script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip()
                stdout = exc.stdout.strip()
                detalle = stderr or stdout or str(exc)
                raise RuntimeError(
                    f"Fallo al generar la curva de diversidad por reinicio con gnuplot para {algoritmo}: {detalle}"
                ) from exc

        rutas_generadas.append(str(outpath_alg))

    return rutas_generadas

def generar_predictor_diversidad_mejora_futura(filas, problema, titulo, outpath, delta_futuro):
    datos_predictor = preparar_predictor_diversidad_mejora_futura(filas, problema, delta_futuro)
    if datos_predictor is None:
        return False

    metricas = datos_predictor["metricas"]
    orden_algoritmos = datos_predictor["orden_algoritmos"]
    paneles_por_algoritmo = datos_predictor["paneles_por_algoritmo"]
    limites_x_por_metrica = datos_predictor["limites_x_por_metrica"]
    limites_y_por_metrica = datos_predictor["limites_y_por_metrica"]

    fig, axes = plt.subplots(
        len(orden_algoritmos),
        len(metricas),
        figsize=(8.8 * len(metricas), 5.4 * len(orden_algoritmos)),
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
            aplicar_limites_compartidos(ax, "x", limites_x_por_metrica[panel["metrica"]])
            aplicar_limites_compartidos(ax, "y", limites_y_por_metrica[panel["metrica"]])

            ax.legend(handles, labels, loc="best", fontsize=8)

    fig.suptitle(titulo)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.88, wspace=0.34, hspace=0.36)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True

def generar_graficos_para_bloque(
    filas,
    outdir,
    algoritmo,
    problema,
    sufijo_archivo,
    delta_futuro,
    plots_seleccionados=None,
    diversidad_reinicio_agregacion="representativa",
):
    fitness_por_adaptacion, tiempo_por_adaptacion, etiquetas = construir_series_por_adaptacion(filas)
    sufijo_titulo = ""
    plots_seleccionados = set(plots_seleccionados or PLOTS_POR_MODO["experimento"])
    predictores_dir = outdir / "predictores"

    ruta_fitness = outdir / f"boxplot_fitness{sufijo_archivo}.png"
    ruta_tiempo = outdir / f"boxplot_tiempo{sufijo_archivo}.png"
    ruta_convergencia = outdir / f"curva_convergencia{sufijo_archivo}.png"
    ruta_convergencia_reinicio = outdir / f"curva_convergencia_reinicio{sufijo_archivo}.png"
    ruta_diversidad = outdir / f"curva_diversidad{sufijo_archivo}.png"
    ruta_diversidad_reinicio = outdir / f"curva_diversidad_reinicio{sufijo_archivo}.png"
    ruta_conjunta_fit_div = outdir / f"curva_fitness_diversidad{sufijo_archivo}.png"
    ruta_predictor_div = predictores_dir / f"predictor_diversidad_mejora_futura{sufijo_archivo}.png"
    ruta_histograma = outdir / f"histograma_fitness_final{sufijo_archivo}.png"
    ruta_tabla_cec = outdir / f"tabla_resultados_cec{sufijo_archivo}.csv"

    rutas = []

    if "boxplot_fitness" in plots_seleccionados:
        generar_boxplot(
            datos_por_etiqueta=fitness_por_adaptacion,
            etiquetas=etiquetas,
            titulo="Boxplot de Fitness por Algoritmo/Adaptacion" + sufijo_titulo,
            ylabel="Fitness",
            outpath=ruta_fitness,
        )
        rutas.append(ruta_fitness)

    if "boxplot_tiempo" in plots_seleccionados:
        generar_boxplot(
            datos_por_etiqueta=tiempo_por_adaptacion,
            etiquetas=etiquetas,
            titulo="Boxplot de Tiempo por Algoritmo/Adaptacion" + sufijo_titulo,
            ylabel="Tiempo (s)",
            outpath=ruta_tiempo,
        )
        rutas.append(ruta_tiempo)

    if "convergencia" in plots_seleccionados and generar_curva_convergencia(
        filas=filas,
        titulo="Curvas de Convergencia por Algoritmo" + sufijo_titulo,
        outpath=ruta_convergencia,
    ):
        rutas.append(ruta_convergencia)

    if "convergencia_reinicio" in plots_seleccionados:
        rutas.extend(
            generar_curvas_convergencia_reinicio(
                filas=filas,
                problema=problema,
                outpath_base=ruta_convergencia_reinicio,
                agregacion=diversidad_reinicio_agregacion,
            )
        )

    if "diversidad" in plots_seleccionados and generar_curva_diversidad(
        filas=filas,
        problema=problema,
        titulo="Curvas de Diversidad vs Evaluaciones" + sufijo_titulo,
        outpath=ruta_diversidad,
    ):
        rutas.append(ruta_diversidad)

    if "diversidad_reinicio" in plots_seleccionados:
        rutas.extend(
            generar_curvas_diversidad_reinicio(
                filas=filas,
                problema=problema,
                outpath_base=ruta_diversidad_reinicio,
                agregacion=diversidad_reinicio_agregacion,
            )
        )

    if "fitness_diversidad" in plots_seleccionados and generar_curva_conjunta_fitness_diversidad(
        filas=filas,
        problema=problema,
        titulo="Curvas Conjuntas de Fitness y Diversidad" + sufijo_titulo,
        outpath=ruta_conjunta_fit_div,
    ):
        rutas.append(ruta_conjunta_fit_div)

    if "predictor_diversidad" in plots_seleccionados:
        predictores_dir.mkdir(parents=True, exist_ok=True)
        if generar_predictor_diversidad_mejora_futura(
            filas=filas,
            problema=problema,
            titulo="Diversidad como Predictor de Mejora Futura" + sufijo_titulo,
            outpath=ruta_predictor_div,
            delta_futuro=delta_futuro,
        ):
            rutas.append(ruta_predictor_div)

    if "histograma_fitness_final" in plots_seleccionados:
        rutas_histograma = generar_histogramas_fitness_final_por_algoritmo_y_variante(
            filas=filas,
            titulo="Histograma de Fitness Final por Algoritmo" + sufijo_titulo,
            outpath=ruta_histograma,
        )
        if rutas_histograma:
            rutas.extend(rutas_histograma)
        elif generar_histograma_fitness_final(
            filas=filas,
            titulo="Histograma de Fitness Final por Algoritmo" + sufijo_titulo,
            outpath=ruta_histograma,
        ):
            rutas.append(ruta_histograma)

    if "tabla_cec" in plots_seleccionados and problema == "cec2017" and generar_tabla_resultados_cec_estandar(
        filas=filas,
        outpath=ruta_tabla_cec,
    ):
        rutas.append(ruta_tabla_cec)

    return rutas

def main():
    global CLAVES_VARIANTES_REINICIO_FILTRADAS
    args = parse_args()
    CLAVES_VARIANTES_REINICIO_FILTRADAS = args.variantes_reinicio
    plots_seleccionados = resolver_plots_seleccionados(args.modo, args.plots)
    outdir = Path(args.outdir)
    outdir_por_funcion = outdir / "por_funcion"
    outdir_por_metaheuristica = outdir / "por_metaheuristica"

    filas = cargar_runs(args.results_csv, args.algoritmo, args.problema)
    if args.problema == "cec2017" and args.cec_funcid is not None:
        funcids_filtrados = set(normalizar_cec_funcids(args.cec_funcid))
        filas = [
            fila
            for fila in filas
            if fila.get("cec_funcid") is not None and int(fila["cec_funcid"]) in funcids_filtrados
        ]
    if args.semilla is not None:
        semillas_filtradas = {int(semilla) for semilla in args.semilla}
        filas = [
            fila
            for fila in filas
            if fila.get("semilla") is not None and int(fila["semilla"]) in semillas_filtradas
        ]
    if len(filas) == 0:
        raise RuntimeError("No hay filas tras aplicar filtros de algoritmo/problema")

    if args.modo == "run":
        if args.problema == "ambos":
            raise ValueError("En modo run debes indicar --problema cec2017 o --problema qap.")
        funcids = normalizar_cec_funcids(args.cec_funcid) if args.problema == "cec2017" else [None]
        if args.problema == "cec2017" and funcids is None:
            raise ValueError("En modo run para CEC2017 debes indicar al menos un valor en --cec-funcid.")

        resumen_runs = []
        if args.seeds_representativas_json:
            registros = cargar_registros_representativos(args.seeds_representativas_json)
            filas_objetivo = seleccionar_runs_desde_registros_representativos(
                filas,
                registros,
                problema=args.problema,
                funcids=funcids if args.problema == "cec2017" else None,
                algoritmo=args.algoritmo,
            )
        else:
            filas_objetivo = []
            for idx_func, funcid in enumerate(funcids):
                filas_objetivo.extend(
                    seleccionar_runs_aleatorias_por_algoritmo(
                        filas,
                        problema=args.problema,
                        cec_funcid=funcid,
                        random_state=int(args.random_state) + idx_func * 1000,
                    )
                )

        for fila in filas_objetivo:
            ruta_metricas = resolver_ruta_metricas_desde_fila(fila, args.results_csv)
            fila = dict(fila)
            fila["ruta_metricas"] = str(ruta_metricas)
            outdir_run = construir_outdir_por_run(outdir, fila)
            generados = generar_plots_preprocesado_por_run(
                fila,
                outdir_run,
                plots_seleccionados=plots_seleccionados,
            )
            ruta_fases = next((ruta for ruta in generados if "histograma_fitness_fases" in str(ruta)), "")
            ruta_dataset = next((ruta for ruta in generados if "histograma_fitness_dataset_completo" in str(ruta)), "")
            resumen_runs.append(
                {
                    "funcid": fila.get("cec_funcid"),
                    "algoritmo": fila["algoritmo"],
                    "variante_reinicio": obtener_clave_variante_reinicio(fila),
                    "ruta_metricas": str(ruta_metricas),
                    "ruta_fases": str(ruta_fases),
                    "ruta_dataset": str(ruta_dataset),
                }
            )
        imprimir_resumen_plots_por_run(resumen_runs, args.problema)
        return

    outdir_por_funcion.mkdir(parents=True, exist_ok=True)
    outdir_por_metaheuristica.mkdir(parents=True, exist_ok=True)

    ficheros_generados = []

    if args.problema == "ambos":
        for problema in ("cec2017", "qap"):
            filas_p = [f for f in filas if f["problema"] == problema]
            if len(filas_p) == 0:
                continue
            if problema == "cec2017":
                for funcid, filas_funcid in agrupar_filas_cec_por_funcid(filas_p).items():
                    outdir_funcid = outdir_por_funcion / f"cec2017_f{int(funcid)}"
                    outdir_funcid.mkdir(parents=True, exist_ok=True)
                    ficheros_generados.extend(
                        generar_graficos_para_bloque(
                            filas=filas_funcid,
                            outdir=outdir_funcid,
                            algoritmo=args.algoritmo,
                            problema=problema,
                            sufijo_archivo=f"_cec2017_f{int(funcid)}",
                            delta_futuro=max(1, args.delta_futuro),
                            plots_seleccionados=plots_seleccionados,
                            diversidad_reinicio_agregacion=args.diversidad_reinicio_agregacion,
                        )
                    )
                    ficheros_generados.extend(
                        generar_curvas_convergencia_por_variante(
                            funcid=funcid,
                            outdir=outdir_funcid,
                            sufijo_base=f"_cec2017_f{int(funcid)}",
                            plots_seleccionados=plots_seleccionados,
                        )
                    )
                continue
            ficheros_generados.extend(
                generar_graficos_para_bloque(
                    filas=filas_p,
                    outdir=outdir_por_funcion / problema,
                    algoritmo=args.algoritmo,
                    problema=problema,
                    sufijo_archivo=f"_{problema}",
                    delta_futuro=max(1, args.delta_futuro),
                    plots_seleccionados=plots_seleccionados,
                    diversidad_reinicio_agregacion=args.diversidad_reinicio_agregacion,
                )
            )
    elif args.problema == "cec2017":
        for funcid, filas_funcid in agrupar_filas_cec_por_funcid(filas).items():
            outdir_funcid = outdir_por_funcion / f"cec2017_f{int(funcid)}"
            outdir_funcid.mkdir(parents=True, exist_ok=True)
            ficheros_generados.extend(
                generar_graficos_para_bloque(
                    filas=filas_funcid,
                    outdir=outdir_funcid,
                    algoritmo=args.algoritmo,
                    problema=args.problema,
                    sufijo_archivo=f"_cec2017_f{int(funcid)}",
                    delta_futuro=max(1, args.delta_futuro),
                    plots_seleccionados=plots_seleccionados,
                    diversidad_reinicio_agregacion=args.diversidad_reinicio_agregacion,
                )
            )
            ficheros_generados.extend(
                generar_curvas_convergencia_por_variante(
                    funcid=funcid,
                    outdir=outdir_funcid,
                    sufijo_base=f"_cec2017_f{int(funcid)}",
                    plots_seleccionados=plots_seleccionados,
                )
            )
    else:
        ficheros_generados.extend(
            generar_graficos_para_bloque(
                filas=filas,
                outdir=outdir_por_funcion / args.problema,
                algoritmo=args.algoritmo,
                problema=args.problema,
                sufijo_archivo="",
                delta_futuro=max(1, args.delta_futuro),
                plots_seleccionados=plots_seleccionados,
                diversidad_reinicio_agregacion=args.diversidad_reinicio_agregacion,
                )
            )

    if args.modo == "experimento" and "reinicios_medios" in plots_seleccionados:
        if args.problema in {"cec2017", "ambos"}:
            filas_cec = [fila for fila in filas if fila.get("problema") == "cec2017"]
            if filas_cec:
                ficheros_generados.extend(
                    generar_barras_reinicios_medios_cec(
                        filas=filas_cec,
                        problema="cec2017",
                        outdir_base=outdir_por_metaheuristica,
                    )
                )

    print(f"Graficos generados en: {outdir_por_funcion}")
    for ruta in ficheros_generados:
        print(f"- {ruta}")

if __name__ == "__main__":
    main()
