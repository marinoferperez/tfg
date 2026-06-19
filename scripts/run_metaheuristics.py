"""
Ejecuta metaheuristicas offline sobre CEC2017.

El script lanza AGE, DE y SHADE puros, guarda los resultados por funcion y
opcionalmente registra metricas detalladas para construir datasets offline.
"""

import argparse
import sys
import time
from pathlib import Path

# Permite ejecutar desde la raiz del repositorio sin instalar el paquete.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.cec2017_problem import MAX_EVALS_POR_DIM
from src.utils.experiment_paths import ALGORITMOS_MH
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
from src.utils.file_io import escribir_csv_dicts, escribir_json


def parse_args():
    """
    Lee los argumentos de linea de comandos.

    Permite seleccionar funciones CEC, semillas, algoritmos, reinicio elitista
    y opciones de registro de metricas/datasets.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta AGE, DE y SHADE offline sobre CEC2017 y guarda resultados "
            "por funcion, semilla y algoritmo."
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
            "Lista explicita de semillas a ejecutar. Acepta valores separados "
            "por espacios o comas. Si se indica, tiene prioridad sobre "
            "--seed-start y --n-seeds."
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
        help="Presupuesto maximo de evaluaciones. Por defecto: 10000 * dimension.",
    )
    parser.add_argument(
        "--restart",
        "--elitist-restart",
        action="store_true",
        help=(
            "Activa el reinicio elitista en CEC2017. El criterio actual "
            "preserva el mejor individuo y reinicia cuando el segundo mejor "
            "fitness permanece estancado durante la paciencia configurada."
        ),
    )
    parser.add_argument(
        "--restart-ratio",
        "--elitist-restart-patience-ratio",
        type=float,
        default=None,
        help=(
            "Fraccion de max_evals que debe transcurrir sin mejora real del "
            "segundo mejor fitness antes de permitir el reinicio. Debe "
            "indicarse cuando se activa --restart."
        ),
    )
    parser.add_argument(
        "--cec-funcid",
        nargs="+",
        required=True,
        help="Funciones CEC a ejecutar: lista (ej. --cec-funcid 1 2 3), CSV (ej. --cec-funcid 1,2,3) o 'all'.",
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
        type=str.lower,
        default="all",
        choices=[*ALGORITMOS_MH, "all"],
        help="Algoritmo a ejecutar. Por defecto all.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Si se indica, no registra metricas DEAP ni genera metricas_runs.",
    )
    parser.add_argument(
        "--save-restart-detail",
        action="store_true",
        help="Guarda reinicios_elitistas.csv con el detalle de cada reinicio.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Si se indica, muestra informacion de progreso por terminal.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/resultados_benchmark_mhs",
        help="Directorio raiz donde guardar resultados y metricas. Por defecto results/resultados_benchmark_mhs.",
    )
    return parser.parse_args()


def crear_metaheuristica_offline(algoritmo, kwargs):
    """
    Instancia el adaptador CEC2017 correspondiente.

    algoritmo: identificador normalizado, uno de age, de o shade.
    kwargs: parametros de construccion de la metaheuristica.
    """
    if algoritmo == "age":
        from src.metaheuristics.algorithms.offline.adapted.age_cec2017 import GeneticStationaryCEC2017

        return GeneticStationaryCEC2017(**kwargs), "age"
    if algoritmo == "de":
        from src.metaheuristics.algorithms.offline.adapted.de_cec2017 import DifferentialEvolutionCEC2017

        return DifferentialEvolutionCEC2017(**kwargs), "de"
    if algoritmo == "shade":
        from src.metaheuristics.algorithms.offline.adapted.shade_cec2017 import SHADECEC2017

        return SHADECEC2017(**kwargs), "shade"
    raise ValueError(f"Algoritmo offline no soportado: {algoritmo}")


def ejecutar_cec(args, semillas, outdir_metricas, algoritmos, funcid):
    """
    Ejecuta las metaheuristicas offline para una funcion CEC concreta.

    args: namespace de argparse.
    semillas: semillas a ejecutar.
    outdir_metricas: directorio base para metricas detalladas.
    algoritmos: algoritmos seleccionados.
    funcid: identificador de funcion CEC2017.
    """
    filas = []
    total_runs = len(algoritmos) * len(semillas)
    run_idx = 0
    registrar_metricas = not args.no_metrics
    ratio_paciencia_restart = normalizar_ratio_paciencia_reinicio(args.restart_ratio)
    restart_activo = bool(args.restart)
    sufijo_reinicio = ""
    if restart_activo:
        sufijo_reinicio = "_reinicio" + sufijo_ratio_paciencia_reinicio(ratio_paciencia_restart)

    for algoritmo in algoritmos:
        for seed in semillas:
            run_idx += 1
            mostrar(
                args,
                f"[CEC2017 F{int(funcid)} {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed}...",
                flush=True,
            )

            kwargs = {}
            if args.pop_size is not None:
                kwargs["tam_poblacion"] = int(args.pop_size)
            if args.max_evals is not None:
                kwargs["max_evals"] = int(args.max_evals)
            if restart_activo:
                kwargs["reinicio"] = True
                kwargs["reinicio_ratio"] = float(ratio_paciencia_restart)
            mtheuristica, algname = crear_metaheuristica_offline(algoritmo, kwargs)

            # Se mide el tiempo de pared de la ejecucion completa.
            t0 = time.perf_counter()
            resultado = mtheuristica.optimize(
                funcid = int(funcid),
                dim = args.cec_dim,
                seed = seed,
                algname = algname,
                registrar_metricas = registrar_metricas,
                ruta_metricas = (
                    str((outdir_metricas / "cec2017" / algoritmo).resolve())
                    if registrar_metricas or args.save_restart_detail
                    else None
                ),
                run_id = (
                    f"{algoritmo}_cec2017_f{int(funcid)}_d{int(args.cec_dim)}_s{int(seed)}"
                    f"{sufijo_reinicio}"
                ),
                cec_workdir = str(outdir_metricas.resolve().parent),
                guardar_reinicios_detalle=bool(args.save_restart_detail),
            )
            dt = time.perf_counter() - t0

            filas.append({
                "algoritmo": algoritmo,
                "cec_funcid": int(funcid),
                "semilla": int(seed),
                "fitness": float(resultado["mejor_fitness"]),
                "cec_error": float(resultado["mejor_error"]),
                "tiempo_s": float(dt),
                "reinicio": bool(restart_activo),
                "reinicio_ratio": (
                    float(ratio_paciencia_restart) if restart_activo else ""
                ),
                "n_reinicios": int(len(resultado.get("reinicios", []))),
            })

            mostrar(
                args,
                f"[CEC2017 F{int(funcid)} {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed} "
                f"fitness={float(resultado['mejor_fitness']):.6f} "
                f"error={float(resultado['mejor_error']):.6f} "
                f"tiempo={dt:.4f}s",
                flush=True,
            )
    return filas

def main():
    """Punto de entrada del script offline."""
    args = parse_args()
    args.restart_ratio = normalizar_ratio_paciencia_reinicio(args.restart_ratio)
    args.restart = bool(args.restart)
    if args.max_evals is None:
        args.max_evals = MAX_EVALS_POR_DIM * int(args.cec_dim)
    if args.restart and args.restart_ratio is None:
        raise ValueError("--restart requiere indicar --restart-ratio.")
    if not args.restart and args.restart_ratio is not None:
        raise ValueError("--restart-ratio solo puede utilizarse junto con --restart.")

    outdir_raiz = Path(args.outdir).resolve()
    nombre_experimento = outdir_raiz.name if outdir_raiz.name else "experimento"
    outdir_cec = (ROOT / "results" / "cec" / nombre_experimento).resolve()

    semillas = gestiona_semillas(args)

    if args.algorithm == "all":
        algoritmos = ALGORITMOS_MH
    else:
        algoritmos = (args.algorithm,)

    funcids_cec = gestiona_funcids_cec(args)
    funcids_txt = ",".join(str(int(f)) for f in funcids_cec)

    mostrar(args, "Configuracion offline:", flush=True)
    mostrar(args, f"  algorithm={args.algorithm}", flush=True)
    mostrar(args, f"  cec_funcid={funcids_txt}", flush=True)
    mostrar(args, f"  cec_dim={int(args.cec_dim)}", flush=True)
    mostrar(args, f"  semillas={','.join(str(s) for s in semillas)}", flush=True)
    mostrar(args, f"  pop_size={int(args.pop_size)}", flush=True)
    mostrar(args, f"  max_evals={int(args.max_evals)}", flush=True)
    mostrar(args, f"  restart={bool(args.restart)}", flush=True)
    mostrar(args, "  restart_ratio=" + (str(args.restart_ratio) if args.restart else "n/a"), flush=True)
    mostrar(args, f"  register_metrics={not args.no_metrics}", flush=True)
    mostrar(args, f"  save_restart_detail={bool(args.save_restart_detail)}", flush=True)
    mostrar(args, f"  outdir={outdir_cec}", flush=True)

    filas_runs_cec = []
    total_funcids = len(funcids_cec)
    for idx, funcid in enumerate(funcids_cec, start=1):
        mostrar(args, f"\nCEC2017 offline: F{int(funcid)} ({idx}/{total_funcids})", flush=True)
        outdir_funcid = outdir_cec / f"f{int(funcid)}"
        outdir_metricas_funcid = outdir_funcid / "metricas_runs"
        filas_funcid = ejecutar_cec(args, semillas, outdir_metricas_funcid, algoritmos, int(funcid))
        filas_runs_cec.extend(filas_funcid)

        config_funcid = {
            "seed_start": int(args.seed_start) if args.seeds is None else None,
            "n_seeds": int(args.n_seeds) if args.seeds is None else len(semillas),
            "seeds": [int(s) for s in semillas],
            "pop_size": int(args.pop_size) if args.pop_size is not None else None,
            "max_evals": int(args.max_evals) if args.max_evals is not None else None,
            "funcid": int(funcid),
            "cec_dim": int(args.cec_dim),
            "algorithm": str(args.algorithm),
            "problema": "cec2017",
            "register_metrics": bool(not args.no_metrics),
            "restart": bool(args.restart),
            "restart_ratio": (
                float(args.restart_ratio)
                if args.restart
                else None
            ),
        }
        guardar_bloque_resultados(
            outdir_funcid,
            filas_funcid,
            config_funcid,
            incluir_columnas_contexto=False,
        )

    if len(filas_runs_cec) == 0:
        raise RuntimeError("No se ejecuto ninguna combinacion. Revisa --algorithm.")

    if filas_runs_cec:
        config_cec = {
            "seed_start": int(args.seed_start) if args.seeds is None else None,
            "n_seeds": int(args.n_seeds) if args.seeds is None else len(semillas),
            "seeds": [int(s) for s in semillas],
            "pop_size": int(args.pop_size) if args.pop_size is not None else None,
            "max_evals": int(args.max_evals) if args.max_evals is not None else None,
            "cec_funcid_raw": list(args.cec_funcid),
            "funcids_resueltas": [int(f) for f in funcids_cec],
            "cec_dim": int(args.cec_dim),
            "algorithm": str(args.algorithm),
            "problema": "cec2017",
            "register_metrics": bool(not args.no_metrics),
            "restart": bool(args.restart),
            "restart_ratio": (
                float(args.restart_ratio)
                if args.restart
                else None
            ),
        }
        filas_resumen_contexto = construir_resumen(filas_runs_cec, desglosar_contexto=True)
        escribir_csv_dicts(outdir_cec / "runs_media.csv", filas_resumen_contexto)
        escribir_json(
            outdir_cec / "runs_media.json",
            {
                "config": config_cec,
                "resumen": filas_resumen_contexto,
            },
        )

        mostrar(args, "\nMedias por funcion y algoritmo:", flush=True)
        for fila in filas_resumen_contexto:
            funcid_txt = f"f{int(fila['cec_funcid'])}" if fila.get("cec_funcid") not in ("", None) else "f?"
            mostrar(
                args,
                f"  {funcid_txt} / {fila['algoritmo']}: "
                f"fitness_media={fila['fitness_media']:.6f}, "
                f"tiempo_media_s={fila['tiempo_media_s']:.6f}",
                flush=True,
            )
        mostrar(args, f"\nResultados guardados en: {outdir_cec}", flush=True)


if __name__ == "__main__":
    main()
