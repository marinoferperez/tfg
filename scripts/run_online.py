"""
Ejecuta metaheuristicas online con filtro subrogado sobre CEC2017.

Lanza AGE, DE y SHADE con el filtro RBF online. Los resultados se guardan
por funcion, semilla y configuracion de subrogado, e incluyen metricas
detalladas del filtro (tasa de rechazo, numero de entrenamientos, etc.).
"""

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.cec2017_problem import MAX_EVALS_POR_DIM
from src.utils.experiment_paths import gestiona_algoritmos
from src.utils.experiment_io import (
    construir_resumen,
    gestiona_funcids_cec,
    gestiona_semillas,
    guardar_bloque_resultados,
    mostrar,
    normalizar_ratio_paciencia_reinicio,
    sufijo_ratio_paciencia_reinicio,
    validar_tam_poblacion,
)
from src.metaheuristics.algorithms.online.adapted.age_cec2017_online import GeneticStationaryCEC2017Online
from src.metaheuristics.algorithms.online.adapted.de_cec2017_online import DifferentialEvolutionCEC2017Online
from src.metaheuristics.algorithms.online.adapted.shade_cec2017_online import SHADECEC2017Online
from src.metaheuristics.surrogate.surrogate_controller import ConfiguracionSubrogadoOnline

from src.utils.file_io import leer_json, escribir_csv_dicts, escribir_json


DEFAULT_RBF_PARAMS_JSON = (
    ROOT
    / "src"
    / "surrogates"
    / "configs"
    / "rbf_multiquadric_eps1_smoothing1e-3_neighbors50.json"
)


def parse_args():
    """Lee y devuelve los argumentos de linea de comandos del modo online."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta metaheuristicas online con filtro subrogado sobre CEC2017. "
            "Actualmente soporta AGE, DE y SHADE online."
        )
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Primera semilla del rango secuencial. Por defecto 1.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Numero de semillas consecutivas a ejecutar desde --seed-start. Por defecto 10.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help=(
            "Lista explicita de semillas. Acepta valores separados por espacios "
            "o comas. Si se indica, tiene prioridad sobre --seed-start y --n-seeds."
        ),
    )
    parser.add_argument(
        "--pop-size",
        type=validar_tam_poblacion,
        default=50,
        help="Tamano de poblacion utilizado por las metaheuristicas. Por defecto 50.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Presupuesto maximo de evaluaciones reales. Por defecto: 10000 * dimension.",
    )
    parser.add_argument(
        "--restart",
        "--elitist-restart",
        action="store_true",
        help=(
            "Activa el reinicio elitista. Preserva el mejor individuo y reinicia "
            "cuando el segundo mejor fitness permanece estancado."
        ),
    )
    parser.add_argument(
        "--restart-ratio",
        "--elitist-restart-patience-ratio",
        type=float,
        default=None,
        help="Fraccion de max_evals sin mejora del segundo mejor antes de permitir el reinicio. Requiere --restart.",
    )
    parser.add_argument(
        "--cec-funcid",
        nargs="+",
        required=True,
        help="Funciones CEC: lista (ej. 1 2 3), CSV (ej. 1,2,3) o 'all'.",
    )
    parser.add_argument(
        "--cec-dim",
        type=int,
        choices=[2, 5, 10, 30, 50, 100],
        default=10,
        help="Dimensionalidad del problema CEC2017. Por defecto 10.",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=["age"],
        help=(
            "Metaheuristica online a ejecutar. Acepta age, de, shade, all, "
            "listas separadas por espacios o listas separadas por comas."
        ),
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Si se indica, no registra metricas DEAP ni datasets por run.",
    )
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help=(
            "Si se registran metricas, evita guardar datasets de muestras "
            "evaluadas por run. Mantiene metricas, resumenes y reinicios."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Si se indica, muestra informacion de progreso por terminal. Por defecto False.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Directorio raiz donde guardar resultados y metricas. "
            "Por defecto se genera automaticamente como online_d<dim>_tam<pop>."
        ),
    )
    parser.add_argument(
        "--surrogate-model",
        default="rbf",
        help="Modelo subrogado empleado por el filtro online. Por defecto rbf.",
    )
    parser.add_argument(
        "--surrogate-params-json",
        default=None,
        help=(
            "JSON con hiperparametros del modelo subrogado. Si no se indica y "
            "el modelo es rbf, se usa la configuracion RBF ajustada."
        ),
    )
    parser.add_argument(
        "--surrogate-prob",
        type=float,
        default=0.50,
        help="Probabilidad p de aplicar el filtro subrogado a un candidato. Por defecto 0.50.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.20,
        help="Fraccion de max_evals inicial sin filtro subrogado (calentamiento). Por defecto 0.20.",
    )
    parser.add_argument(
        "--window-ratio",
        type=float,
        default=0.20,
        help="Fraccion de max_evals usada como ventana de entrenamiento del subrogado. Por defecto 0.20.",
    )
    parser.add_argument(
        "--restart-cooldown-evals",
        type=int,
        default=None,
        help=(
            "Evaluaciones reales de enfriamiento tras cada reinicio. "
            "Si no se indica, el enfriamiento queda desactivado (equivale a 0)."
        ),
    )
    parser.add_argument(
        "--save-surrogate-decisions",
        action="store_true",
        help=(
            "Guarda surrogate_decisions.csv con fitness_pred, fitness_ref y "
            "margen_pred_ref para analizar el filtro."
        ),
    )
    parser.add_argument(
        "--save-restart-detail",
        action="store_true",
        help="Guarda reinicios_elitistas.csv con el detalle de cada reinicio.",
    )
    parser.add_argument(
        "--retrain-ratio",
        type=float,
        default=0.25,
        help=(
            "Fraccion de la ventana de entrenamiento que debe renovarse antes "
            "de reentrenar el subrogado. Por ejemplo, 0.25 con window_ratio=0.20 "
            "y max_evals=10000 implica reentrenar cada 500 evaluaciones reales. "
            "Por defecto 0.25."
        ),
    )
    return parser.parse_args()


def gestiona_algoritmos_online(valores):
    """
    Normaliza la seleccion de algoritmos online.

    Permite:
        --algorithm age
        --algorithm de shade
        --algorithm age,de
        --algorithm all
    """
    tokens = []
    for valor in valores or ["age"]:
        for token in str(valor).split(","):
            token = token.strip().lower()
            if token:
                tokens.append(token)

    if not tokens:
        return ["age"]

    if "all" in tokens:
        return ["age", "de", "shade"]

    permitidos = {"age", "de", "shade"}
    algoritmos = []
    for token in tokens:
        if token not in permitidos:
            raise ValueError(
                f"Algoritmo online no soportado: {token}. "
                "Usa age, de, shade o all."
            )
        if token not in algoritmos:
            algoritmos.append(token)

    return algoritmos


def validar_args(args):
    """Valida y normaliza los argumentos parseados; añade args.algoritmos."""
    args.restart_ratio = normalizar_ratio_paciencia_reinicio(args.restart_ratio)
    args.restart = bool(args.restart)

    if args.max_evals is None:
        args.max_evals = MAX_EVALS_POR_DIM * int(args.cec_dim)
    if int(args.max_evals) <= 0:
        raise ValueError("--max-evals debe ser positivo.")

    if args.restart and args.restart_ratio is None:
        raise ValueError("--restart requiere indicar --restart-ratio.")
    if not args.restart and args.restart_ratio is not None:
        raise ValueError("--restart-ratio solo puede utilizarse junto con --restart.")

    args.algoritmos = gestiona_algoritmos(args.algorithm)


def construir_config_subrogado(args, seed):
    """Construye la configuracion del filtro subrogado para una semilla concreta."""
    surrogate_model = str(args.surrogate_model).lower()
    if args.surrogate_params_json is not None:
        modelo_params = dict(leer_json(args.surrogate_params_json))
    elif surrogate_model == "rbf":
        modelo_params = dict(leer_json(DEFAULT_RBF_PARAMS_JSON))
    else:
        modelo_params = {}
    return ConfiguracionSubrogadoOnline(
        modelo_nombre=surrogate_model,
        modelo_params=modelo_params,
        cooldown_reinicio_evals=int(args.restart_cooldown_evals or 0),
        warmup_ratio=float(args.warmup_ratio),
        window_ratio=float(args.window_ratio),
        probabilidad_subrogado=float(args.surrogate_prob),
        max_evals=int(args.max_evals),
        minimizacion=True,
        seed=int(seed),
        retrain_ratio=float(args.retrain_ratio),
    )


def construir_metaheuristica_online(args, seed, algoritmo):
    """Instancia la metaheuristica online indicada con la configuracion de subrogado y reinicio."""
    clases = {
        "age": GeneticStationaryCEC2017Online,
        "de": DifferentialEvolutionCEC2017Online,
        "shade": SHADECEC2017Online,
    }
    if algoritmo not in clases:
        raise NotImplementedError(f"Algoritmo online no soportado: {algoritmo}")

    kwargs = {
        "tam_poblacion": int(args.pop_size),
        "max_evals": int(args.max_evals),
    }
    if args.restart:
        kwargs["reinicio"] = True
        kwargs["reinicio_ratio"] = float(args.restart_ratio)

    return clases[algoritmo](
        surrogate_config=construir_config_subrogado(args, seed),
        **kwargs,
    )


def fila_run(args, algoritmo, funcid, seed, resultado, tiempo_s):
    """Construye el dict-fila de una run online para el CSV de resultados."""
    resumen_online = dict(resultado.get("resumen_online", {}))
    return {
        "algoritmo": str(algoritmo),
        "cec_funcid": int(funcid),
        "semilla": int(seed),
        "fitness": float(resultado["mejor_fitness"]),
        "cec_error": float(resultado["mejor_error"]),
        "tiempo_s": float(tiempo_s),
        "restart": bool(args.restart),
        "restart_ratio": (
            float(args.restart_ratio)
            if args.restart
            else ""
        ),
        "n_reinicios": int(len(resultado.get("reinicios", []))),
        "surrogate_model": str(resumen_online.get("modelo_nombre", args.surrogate_model)),
        "surrogate_prob": float(args.surrogate_prob),
        "warmup_ratio": float(args.warmup_ratio),
        "window_ratio": float(args.window_ratio),
        "restart_cooldown_evals": int(args.restart_cooldown_evals or 0),
        "evals_reales": int(resumen_online.get("evals_reales", 0)),
        "candidatos_generados": int(resumen_online.get("candidatos_generados", 0)),
        "candidatos_con_subrogado": int(resumen_online.get("candidatos_con_subrogado", 0)),
        "candidatos_evaluados_reales": int(
            resumen_online.get("candidatos_evaluados_reales", 0)
        ),
        "candidatos_aceptados_por_subrogado": int(
            resumen_online.get("candidatos_aceptados_por_subrogado", 0)
        ),
        "candidatos_rechazados": int(resumen_online.get("candidatos_rechazados", 0)),
        "porcentaje_rechazo": float(resumen_online.get("porcentaje_rechazo", 0.0)),
        "entrenamientos_rbf": int(resumen_online.get("entrenamientos_rbf", 0)),
        "tiempo_entrenamiento_total": float(
            resumen_online.get("tiempo_entrenamiento_total", 0.0)
        ),
        "tiempo_prediccion_total": float(
            resumen_online.get("tiempo_prediccion_total", 0.0)
        ),
        "tiempo_online_total": float(resumen_online.get("tiempo_online_total", 0.0)),
        "retrain_ratio": float(resumen_online.get("retrain_ratio", args.retrain_ratio)),
        "retrain_interval_efectivo": int(
            resumen_online.get("retrain_interval_efectivo", 0)
        ),
    }


def ejecutar_cec_online(args, algoritmo, semillas, outdir_metricas, funcid):
    """Ejecuta el algoritmo online sobre todas las semillas para una funcion CEC."""
    filas = []
    registrar_metricas = not args.no_metrics
    ratio_paciencia_restart = args.restart_ratio
    restart_activo = bool(args.restart)
    sufijo_reinicio = (
        "_reinicio" + sufijo_ratio_paciencia_reinicio(ratio_paciencia_restart)
        if restart_activo
        else ""
    )
    prob_label = f"p{int(round(float(args.surrogate_prob) * 100)):03d}"
    retrain_label = f"rt{int(round(float(args.retrain_ratio) * 100)):03d}"
    prefijo_adaptacion = {"age": "ao", "de": "do", "shade": "so"}[algoritmo]
    adaptacion = (
        f"{prefijo_adaptacion}_{prob_label}_c{int(args.restart_cooldown_evals or 0)}_{retrain_label}"
    )

    for run_idx, seed in enumerate(semillas, start=1):
        mostrar(
            args,
            f"[ONLINE CEC2017 F{int(funcid)} {run_idx}/{len(semillas)}] "
            f"{algoritmo.upper()} seed={seed} p={float(args.surrogate_prob):.2f}...",
            flush=True,
        )

        metaheuristica = construir_metaheuristica_online(args, seed, algoritmo)
        run_id = (
            f"{algoritmo}_online_cec2017_f{int(funcid)}_d{int(args.cec_dim)}_s{int(seed)}"
            f"_{prob_label}_cd{int(args.restart_cooldown_evals or 0)}_{retrain_label}"
            f"{sufijo_reinicio}"
        )

        t0 = time.perf_counter()
        resultado = metaheuristica.optimize(
            funcid=int(funcid),
            dim=int(args.cec_dim),
            seed=int(seed),
            algname=adaptacion,
            registrar_metricas=registrar_metricas,
            ruta_metricas=(
                str((outdir_metricas / "cec2017" / algoritmo).resolve())
                if (
                    registrar_metricas
                    or args.save_surrogate_decisions
                    or args.save_restart_detail
                )
                else None
            ),
            run_id=run_id,
            cec_workdir=str(outdir_metricas.resolve().parent),
            guardar_decisiones_subrogado=bool(args.save_surrogate_decisions),
            guardar_reinicios_detalle=bool(args.save_restart_detail),
            guardar_dataset=not bool(args.no_dataset),
        )
        tiempo_s = time.perf_counter() - t0

        filas.append(
            fila_run(
                args=args,
                algoritmo=algoritmo,
                funcid=funcid,
                seed=seed,
                resultado=resultado,
                tiempo_s=tiempo_s,
            )
        )

        mostrar(
            args,
            f"[ONLINE CEC2017 F{int(funcid)}] {algoritmo.upper()} seed={seed} "
            f"fitness={float(resultado['mejor_fitness']):.6f} "
            f"error={float(resultado['mejor_error']):.6f} "
            f"tiempo={tiempo_s:.4f}s",
            flush=True,
        )

    return filas


def config_base(args, semillas, funcids):
    """Construye el dict de configuracion base que se guarda junto a los resultados."""
    return {
        "seed_start": int(args.seed_start) if args.seeds is None else None,
        "n_seeds": int(args.n_seeds) if args.seeds is None else len(semillas),
        "seeds": [int(s) for s in semillas],
        "pop_size": int(args.pop_size),
        "max_evals": int(args.max_evals),
        "cec_funcid_raw": list(args.cec_funcid),
        "funcids_resueltas": [int(f) for f in funcids],
        "cec_dim": int(args.cec_dim),
        "algoritmos": [str(a) for a in args.algoritmos],
        "problema": "cec2017",
        "register_metrics": bool(not args.no_metrics),
        "save_dataset": bool(not args.no_dataset),
        "restart": bool(args.restart),
        "restart_ratio": (
            float(args.restart_ratio)
            if args.restart
            else None
        ),
        "surrogate_model": str(args.surrogate_model).lower(),
        "surrogate_params_json": (
            str(Path(args.surrogate_params_json).resolve())
            if args.surrogate_params_json is not None
            else str(DEFAULT_RBF_PARAMS_JSON)
        ),
        "surrogate_prob": float(args.surrogate_prob),
        "warmup_ratio": float(args.warmup_ratio),
        "window_ratio": float(args.window_ratio),
        "restart_cooldown_evals": int(args.restart_cooldown_evals or 0),
        "retrain_ratio": float(args.retrain_ratio),
        "save_surrogate_decisions": bool(args.save_surrogate_decisions),
    }


def main():
    """Punto de entrada del script online."""
    args = parse_args()
    validar_args(args)

    semillas = gestiona_semillas(args)
    funcids_cec = gestiona_funcids_cec(args)

    if args.outdir is None:
        args.outdir = f"metaheuristics_online_d{args.cec_dim}_tam{int(args.pop_size)}"
    outdir_raiz = Path(args.outdir).resolve()
    nombre_experimento = outdir_raiz.name if outdir_raiz.name else "online"
    outdir_cec = (ROOT / "results" / nombre_experimento).resolve()

    mostrar(args, "Configuracion online:", flush=True)
    mostrar(args, f"  algoritmos={','.join(str(a) for a in args.algoritmos)}", flush=True)
    mostrar(args, f"  funciones={','.join(str(f) for f in funcids_cec)}", flush=True)
    mostrar(args, f"  semillas={','.join(str(s) for s in semillas)}", flush=True)
    mostrar(args, f"  cec_dim={int(args.cec_dim)}", flush=True)
    mostrar(args, f"  pop_size={int(args.pop_size)}", flush=True)
    mostrar(args, f"  max_evals={int(args.max_evals)}", flush=True)
    mostrar(args, f"  restart={bool(args.restart)}", flush=True)
    mostrar(
        args,
        "  restart_ratio=" + (str(args.restart_ratio) if args.restart else "n/a"),
        flush=True,
    )
    mostrar(args, f"  surrogate_model={str(args.surrogate_model).lower()}", flush=True)
    mostrar(args, f"  surrogate_prob={float(args.surrogate_prob)}", flush=True)
    mostrar(args, f"  warmup_ratio={float(args.warmup_ratio)}", flush=True)
    mostrar(args, f"  window_ratio={float(args.window_ratio)}", flush=True)
    mostrar(args, f"  restart_cooldown_evals={int(args.restart_cooldown_evals or 0)}", flush=True)
    mostrar(args, f"  retrain_ratio={float(args.retrain_ratio)}", flush=True)
    mostrar(args, f"  save_dataset={not bool(args.no_dataset)}", flush=True)
    mostrar(args, f"  save_surrogate_decisions={bool(args.save_surrogate_decisions)}", flush=True)
    mostrar(args, f"  outdir={outdir_cec}", flush=True)

    filas_runs_cec = []
    for idx, funcid in enumerate(funcids_cec, start=1):
        mostrar(args, f"\nCEC2017 online: F{int(funcid)} ({idx}/{len(funcids_cec)})")
        outdir_funcid = outdir_cec / f"f{int(funcid)}"
        outdir_metricas_funcid = outdir_funcid / "metricas_runs"

        filas_funcid = []
        for alg_idx, algoritmo in enumerate(args.algoritmos, start=1):
            mostrar(
                args,
                f"  Algoritmo online: {algoritmo.upper()} ({alg_idx}/{len(args.algoritmos)})",
                flush=True,
            )
            filas_funcid.extend(
                ejecutar_cec_online(
                    args,
                    algoritmo,
                    semillas,
                    outdir_metricas_funcid,
                    int(funcid),
                )
            )
        filas_runs_cec.extend(filas_funcid)

        config_funcid = config_base(args, semillas, [int(funcid)])
        config_funcid["funcid"] = int(funcid)
        guardar_bloque_resultados(
            outdir_funcid,
            filas_funcid,
            config_funcid,
            incluir_columnas_contexto=False,
        )

    if not filas_runs_cec:
        raise RuntimeError("No se ejecuto ninguna combinacion online.")

    config_cec = config_base(args, semillas, funcids_cec)
    filas_resumen_contexto = construir_resumen(filas_runs_cec, desglosar_contexto=True)
    escribir_csv_dicts(outdir_cec / "runs_media.csv", filas_resumen_contexto)
    escribir_json(
        outdir_cec / "runs_media.json",
        {
            "config": config_cec,
            "resumen": filas_resumen_contexto,
        },
    )

    mostrar(args, f"\nResultados online guardados en: {outdir_cec}", flush=True)


if __name__ == "__main__":
    main()
