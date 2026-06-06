# ejecuta x semillas por defecto y calcula el promedio de fitness y tiempo.
# usa siempre el logger DEAP ya integrado en los adaptadores.

import argparse
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

# permite ejecutar desde la raiz del repositorio sin instalar paquete
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import escribir_csv_dicts, escribir_json

VENTANA_DIAGNOSTICA_REINICIO = 2500

# parse_args determina las flags y posibles opciones de ejecucion

def normalizar_algoritmo(valor):
    return "all" if str(valor).lower() == "todos" else str(valor).lower()


def validar_tam_poblacion(valor):
    valor = int(valor)
    if valor < 4:
        raise argparse.ArgumentTypeError("debe ser un entero >= 4")
    return valor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=10)
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
        "--tam-poblacion",
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
        "--reinicio",
        "--reinicio-elitista",
        dest="reinicio_elitista",
        action="store_true",
        help=(
            "Activa el reinicio elitista en CEC2017. El criterio actual "
            "preserva el mejor individuo y reinicia cuando el segundo mejor "
            "fitness permanece estancado durante la paciencia configurada."
        ),
    )
    parser.add_argument(
        "--reinicio-ratio",
        "--reinicio-elitista-ratio-paciencia",
        dest="reinicio_elitista_ratio_paciencia",
        type=float,
        default=None,
        help=(
            "Fraccion de max_evals que debe transcurrir sin mejora real del "
            "segundo mejor fitness antes de permitir el reinicio. Debe "
            "indicarse cuando se activa --reinicio."
        ),
    )
    parser.add_argument(
        "--cec-funcid",
        nargs="+",
        required=True,
        help="Funciones CEC a ejecutar: lista (ej. --cec-funcid 1 2 3), CSV (ej. --cec-funcid 1,2,3) o 'all'.",
    )
    parser.add_argument("--cec-dim", type=int, choices=[2, 5, 10, 30, 50], default=10)
    parser.add_argument(
        "--algoritmo",
        type=normalizar_algoritmo,
        default="all",
        choices=["age", "de", "shade", "all"],
        help="Algoritmo a ejecutar",
    )
    parser.add_argument(
        "--sin-metricas",
        action="store_true",
        help="Si se indica, no registra métricas DEAP ni genera metricas_runs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Si se indica, muestra informacion de progreso por terminal.",
    )
    parser.add_argument(
        "--generar-dataset-completo",
        action="store_true",
        help=(
            "Si se indica, concatena las runs por funcion/algoritmo y guarda "
            "dataset_completo.h5 en benchmarking/offline/dataset_preprocesado."
        ),
    )
    parser.add_argument(
        "--solo-dataset-completo",
        action="store_true",
        help=(
            "Si se indica, no ejecuta benchmarks; solo genera dataset_completo "
            "a partir de metricas_runs ya existentes."
        ),
    )

    parser.add_argument("--outdir", type=str, default="results/resultados_benchmark_mhs")
    return parser.parse_args()


def normalizar_ratio_paciencia_reinicio(valor):
    if valor is None:
        return None
    valor = float(valor)
    if not np.isfinite(valor):
        raise ValueError("--reinicio-ratio debe ser finito.")
    if valor <= 0.0 or valor >= 1.0:
        raise ValueError(
            "--reinicio-ratio debe estar en el intervalo abierto (0, 1)."
        )
    return valor


def sufijo_ratio_paciencia_reinicio(valor):
    txt = f"{float(valor):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"_pat{txt}"


def gestiona_funcids_cec(args):
    tokens = []
    for parte in args.cec_funcid:
        for tk in str(parte).split(","):
            tk = tk.strip().lower()
            if tk:
                tokens.append(tk)

    if len(tokens) == 0:
        raise ValueError("--cec-funcid no puede estar vacio.")

    if "all" in tokens:
        if len(tokens) > 1:
            raise ValueError("Si usas --cec-funcid all, no combines con otros valores.")
        return list(range(1, 31))

    vistos = set()
    funcids = []
    for tk in tokens:
        try:
            fid = int(tk)
        except ValueError as exc:
            raise ValueError(f"Valor de --cec-funcid invalido: '{tk}'. Usa enteros en [1, 30] o 'all'.") from exc
        if not 1 <= fid <= 30:
            raise ValueError(f"funcid={fid} fuera de rango. Debe estar en [1, 30].")
        if fid not in vistos:
            vistos.add(fid)
            funcids.append(fid)
    return funcids


def gestiona_semillas(args):
    if args.seeds is None:
        semillas = [int(args.seed_start) + i for i in range(int(args.n_seeds))]
        if len(semillas) == 0:
            raise ValueError("n-seeds debe ser >= 1")
        return semillas

    tokens = []
    for parte in args.seeds:
        for tk in str(parte).split(","):
            tk = tk.strip()
            if tk:
                tokens.append(tk)

    if not tokens:
        raise ValueError("--seeds no puede estar vacio.")

    semillas = []
    vistas = set()
    for tk in tokens:
        try:
            seed = int(tk)
        except ValueError as exc:
            raise ValueError(f"Valor de --seeds invalido: '{tk}'. Usa enteros positivos.") from exc
        if seed < 1:
            raise ValueError(f"Semilla invalida: {seed}. Debe ser >= 1.")
        if seed not in vistas:
            vistas.add(seed)
            semillas.append(seed)
    return semillas


def clave_fila_run(fila):
    return (
        str(fila.get("problema", "")),
        str(fila.get("algoritmo", "")),
        str(fila.get("adaptacion", "")),
        str(fila.get("cec_funcid", "")),
        str(fila.get("qap_instancia", "")),
        int(fila.get("semilla", 0)),
    )


def leer_csv_dicts(path):
    import csv

    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fusionar_runs_existentes(path, filas_nuevas):
    path = Path(path)
    existentes = leer_csv_dicts(path)
    if not existentes:
        return list(filas_nuevas)

    fusionadas = {clave_fila_run(fila): fila for fila in existentes}
    for fila in filas_nuevas:
        fusionadas[clave_fila_run(fila)] = fila
    return sorted(
        fusionadas.values(),
        key=lambda fila: (
            str(fila.get("problema", "")),
            str(fila.get("algoritmo", "")),
            int(float(fila.get("cec_funcid") or 0)),
            str(fila.get("qap_instancia", "")),
            int(float(fila.get("semilla") or 0)),
        ),
    )


def guardar_bloque_resultados(outdir, filas_runs, config_json, incluir_columnas_contexto=True):
    filas_runs = fusionar_runs_existentes(outdir / "runs.csv", filas_runs)
    filas_resumen = construir_resumen(
        filas_runs,
        incluir_columnas_contexto=incluir_columnas_contexto,
    )
    guardar_csv(outdir / "runs.csv", filas_runs)
    guardar_csv(outdir / "resumen_promedios.csv", filas_resumen)
    guardar_json(
        outdir / "resumen_promedios.json",
        {
            "config": config_json,
            "resumen": filas_resumen,
        },
    )
    return filas_resumen

def guardar_csv(path, filas):
    campos = list(filas[0].keys()) if filas else None
    escribir_csv_dicts(path, filas, fieldnames=campos)

def guardar_json(path, payload):
    escribir_json(path, payload)

def mostrar(args, *valores, **kwargs):
    if args.verbose:
        print(*valores, **kwargs)

def resumen_grupo(filas):
    fitness = np.asarray([float(f["fitness"]) for f in filas], dtype=float)
    tiempos = np.asarray([float(f["tiempo_s"]) for f in filas], dtype=float)

    return {
        "n_runs": int(len(filas)),
        "fitness_promedio": float(np.mean(fitness)),
        "tiempo_promedio_s": float(np.mean(tiempos)),
    }

def ejecutar_cec(args, semillas, outdir_metricas, algoritmos, funcid):
    filas = []
    total_runs = len(algoritmos) * len(semillas)
    run_idx = 0
    registrar_metricas = not args.sin_metricas
    ratio_paciencia_reinicio = normalizar_ratio_paciencia_reinicio(args.reinicio_elitista_ratio_paciencia)
    reinicio_activo = bool(args.reinicio_elitista)
    sufijo_reinicio = ""
    if reinicio_activo:
        sufijo_reinicio = "_reinicio" + sufijo_ratio_paciencia_reinicio(ratio_paciencia_reinicio)

    for algoritmo in algoritmos:
        for seed in semillas:
            run_idx += 1
            mostrar(
                args,
                f"[CEC2017 F{int(funcid)} {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed}...",
                flush=True,
            )

            if algoritmo == "age":
                from metaheuristics.age.adapted.genetic_stationary_cec2017 import GeneticStationaryCEC2017
                kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    kwargs["max_evals"] = int(args.max_evals)
                if reinicio_activo:
                    kwargs["reinicio_elitista"] = True
                    kwargs["reinicio_elitista_ratio_paciencia"] = float(ratio_paciencia_reinicio)
                mtheuristica = GeneticStationaryCEC2017(**kwargs)
                algname = "age"
            elif algoritmo == "shade":
                from metaheuristics.shade.adapted.shade_cec2017 import SHADECEC2017
                kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    kwargs["max_evals"] = int(args.max_evals)
                if reinicio_activo:
                    kwargs["reinicio_elitista"] = True
                    kwargs["reinicio_elitista_ratio_paciencia"] = float(ratio_paciencia_reinicio)
                mtheuristica = SHADECEC2017(**kwargs)
                algname = "shade"
            else:
                from metaheuristics.de.adapted.differential_evolution_cec2017 import DifferentialEvolutionCEC2017
                kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    kwargs["max_evals"] = int(args.max_evals)
                if reinicio_activo:
                    kwargs["reinicio_elitista"] = True
                    kwargs["reinicio_elitista_ratio_paciencia"] = float(ratio_paciencia_reinicio)
                mtheuristica = DifferentialEvolutionCEC2017(**kwargs)
                algname = "de"

            t0 = time.perf_counter()
            resultado = mtheuristica.optimize(
                funcid = int(funcid),
                dim = args.cec_dim,
                seed = seed,
                algname = algname,
                registrar_metricas = registrar_metricas,
                ruta_metricas = str((outdir_metricas / "cec2017" / algoritmo).resolve()) if registrar_metricas else None,
                run_id = (
                    f"{algoritmo}_cec2017_f{int(funcid)}_d{int(args.cec_dim)}_s{int(seed)}"
                    f"{sufijo_reinicio}"
                ),
                cec_workdir = str(outdir_metricas.resolve().parent),
            )
            dt = time.perf_counter() - t0

            filas.append({
                "problema": "cec2017",
                "algoritmo": algoritmo,
                "adaptacion": f"{algoritmo}_cec2017",
                "cec_funcid": int(funcid),
                "qap_instancia": "",
                "semilla": int(seed),
                "fitness": float(resultado["mejor_fitness"]),
                "cec_error": float(resultado["mejor_error"]),
                "tiempo_s": float(dt),  
                "ruta_metricas": str(resultado.get("ruta_metricas", "")),
                "reinicio_elitista": bool(reinicio_activo),
                "reinicio_elitista_ratio_estabilidad_diversidad": "",
                "reinicio_elitista_ratio_paciencia": (
                    float(ratio_paciencia_reinicio) if reinicio_activo else ""
                ),
                "reinicio_elitista_ventana_evaluaciones": (
                    int(VENTANA_DIAGNOSTICA_REINICIO) if reinicio_activo else ""
                ),
                "n_reinicios_elitistas": int(len(resultado.get("reinicios_elitistas", []))),
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

def construir_resumen(filas_runs, desglosar_contexto=False, incluir_columnas_contexto=True):
    grupos = defaultdict(list)
    for fila in filas_runs:
        if desglosar_contexto:
            clave = (
                fila["problema"],
                fila["algoritmo"],
                fila["adaptacion"],
                fila.get("cec_funcid", ""),
                fila.get("qap_instancia", ""),
            )
        else:
            clave = (fila["problema"], fila["algoritmo"], fila["adaptacion"])
        grupos[clave].append(fila)

    filas_resumen = []
    for clave, filas in sorted(grupos.items()):
        if desglosar_contexto:
            problema, algoritmo, adaptacion, cec_funcid, qap_instancia = clave
        else:
            problema, algoritmo, adaptacion = clave
            cec_funcid, qap_instancia = "", ""

        resumen = resumen_grupo(filas)
        fila_resumen = {"algoritmo": algoritmo}
        if desglosar_contexto:
            fila_resumen["cec_funcid"] = cec_funcid
        elif incluir_columnas_contexto:
            fila_resumen["cec_funcid"] = cec_funcid
            fila_resumen["qap_instancia"] = qap_instancia
        fila_resumen.update(resumen)
        filas_resumen.append(fila_resumen)
    return filas_resumen


def main():
    args = parse_args()
    from preprocesado_de_datos.exportar_dataset_completo import ejecutar_exportacion_dataset_completo
    args.reinicio_elitista_ratio_paciencia = normalizar_ratio_paciencia_reinicio(
        args.reinicio_elitista_ratio_paciencia
    )
    args.reinicio_elitista = bool(args.reinicio_elitista)
    if args.max_evals is None:
        args.max_evals = 10000 * int(args.cec_dim)
    if args.reinicio_elitista and args.reinicio_elitista_ratio_paciencia is None:
        raise ValueError("--reinicio requiere indicar --reinicio-ratio.")
    if not args.reinicio_elitista and args.reinicio_elitista_ratio_paciencia is not None:
        raise ValueError("--reinicio-ratio solo puede utilizarse junto con --reinicio.")

    outdir_raiz = Path(args.outdir).resolve()
    nombre_experimento = outdir_raiz.name if outdir_raiz.name else "experimento"
    outdir_cec = (ROOT / "results" / "cec" / nombre_experimento).resolve()

    semillas = gestiona_semillas(args)

    if args.algoritmo in ("all", "todos"):
        algoritmos = ("age", "de", "shade")
    else:
        algoritmos = (args.algoritmo,)

    funcids_cec = gestiona_funcids_cec(args)
    funcids_txt = ",".join(str(int(f)) for f in funcids_cec)

    mostrar(args, "Configuracion:", flush=True)
    if args.seeds is None:
        mostrar(args, f"  semillas={semillas[0]}..{semillas[-1]} (n={len(semillas)})", flush=True)
    else:
        mostrar(args, f"  semillas={','.join(str(s) for s in semillas)} (n={len(semillas)})", flush=True)
    mostrar(
        args,
        "  tam_poblacion="
        + (
            str(int(args.tam_poblacion))
            if args.tam_poblacion is not None
            else "default_original_algoritmo"
        ),
        flush=True,
    )
    mostrar(
        args,
        "  max_evals="
        + (
            str(int(args.max_evals))
            if args.max_evals is not None
            else "default_original_algoritmo"
        ),
        flush=True,
    )
    mostrar(args, f"  cec: funcid={funcids_txt}, dim={args.cec_dim}", flush=True)
    mostrar(args, f"  algoritmo={args.algoritmo}", flush=True)
    mostrar(args, f"  registrar_metricas={not args.sin_metricas} (DEAP)", flush=True)
    mostrar(args, f"  reinicio={bool(args.reinicio_elitista)}", flush=True)
    mostrar(
        args,
        "  reinicio_ratio="
        + (
            str(args.reinicio_elitista_ratio_paciencia)
            if args.reinicio_elitista
            else "no_aplica"
        ),
        flush=True,
    )
    mostrar(args, f"  generar_dataset_completo={bool(args.generar_dataset_completo)}", flush=True)
    mostrar(args, f"  solo_dataset_completo={bool(args.solo_dataset_completo)}", flush=True)
    mostrar(args, f"  outdir_solicitado={outdir_raiz}", flush=True)
    mostrar(args, f"  outdir_cec_fijo={outdir_cec}", flush=True)

    if args.generar_dataset_completo and args.sin_metricas and not args.solo_dataset_completo:
        raise ValueError(
            "--generar-dataset-completo requiere metricas_runs disponibles. "
            "No es compatible con --sin-metricas durante la misma ejecucion."
        )

    if args.solo_dataset_completo:
        resultados_exportacion = []
        for funcid in funcids_cec:
            resultados_exportacion.extend(
                ejecutar_exportacion_dataset_completo(
                    outdir_cec,
                    algoritmo=args.algoritmo,
                    funcion=f"f{int(funcid)}",
                )
            )

        if not resultados_exportacion:
            raise RuntimeError("No se generó ningun dataset_completo.")

        mostrar(args, "\nDatasets completos generados:", flush=True)
        for resultado in resultados_exportacion:
            bloque = resultado["funcion"]
            mostrar(
                args,
                f"  {resultado['algoritmo']}/{bloque}: "
                f"runs={resultado['n_runs_entrada']} muestras={resultado['n_muestras']}",
                flush=True,
            )
            mostrar(args, f"    {resultado['dataset_path']}", flush=True)
        return

    filas_runs_cec = []
    total_funcids = len(funcids_cec)
    for idx, funcid in enumerate(funcids_cec, start=1):
        mostrar(
            args,
            f"\nCEC2017: F{int(funcid)} ({idx}/{total_funcids})",
            flush=True,
        )
        outdir_funcid = outdir_cec / f"f{int(funcid)}"
        outdir_metricas_funcid = outdir_funcid / "metricas_runs"
        filas_funcid = ejecutar_cec(args, semillas, outdir_metricas_funcid, algoritmos, int(funcid))
        filas_runs_cec.extend(filas_funcid)

        config_funcid = {
            "seed_start": int(args.seed_start) if args.seeds is None else None,
            "n_seeds": int(args.n_seeds) if args.seeds is None else len(semillas),
            "seeds": [int(s) for s in semillas],
            "tam_poblacion": int(args.tam_poblacion) if args.tam_poblacion is not None else None,
            "max_evals": int(args.max_evals) if args.max_evals is not None else None,
            "funcid": int(funcid),
            "cec_dim": int(args.cec_dim),
            "algoritmo": str(args.algoritmo),
            "problema": "cec2017",
            "registrar_metricas": bool(not args.sin_metricas),
            "reinicio_elitista": bool(args.reinicio_elitista),
            "reinicio_elitista_ratio_estabilidad_diversidad": None,
            "reinicio_elitista_ratio_paciencia": (
                float(args.reinicio_elitista_ratio_paciencia)
                if args.reinicio_elitista
                else None
            ),
            "reinicio_elitista_ventana_evaluaciones": (
                int(VENTANA_DIAGNOSTICA_REINICIO)
                if args.reinicio_elitista
                else None
            ),
        }
        guardar_bloque_resultados(
            outdir_funcid,
            filas_funcid,
            config_funcid,
            incluir_columnas_contexto=False,
        )

        if args.generar_dataset_completo:
            resultados_dataset = ejecutar_exportacion_dataset_completo(
                outdir_cec,
                algoritmo=args.algoritmo,
                funcion=f"f{int(funcid)}",
            )
            for resultado in resultados_dataset:
                mostrar(
                    args,
                    f"  dataset_completo {resultado['algoritmo']}/f{int(funcid)} "
                    f"muestras={resultado['n_muestras']} -> {resultado['dataset_path']}",
                    flush=True,
                )

    if len(filas_runs_cec) == 0:
        raise RuntimeError("No se ejecuto ninguna combinacion. Revisa --algoritmo.")

    if filas_runs_cec:
        config_cec = {
            "seed_start": int(args.seed_start) if args.seeds is None else None,
            "n_seeds": int(args.n_seeds) if args.seeds is None else len(semillas),
            "seeds": [int(s) for s in semillas],
            "tam_poblacion": int(args.tam_poblacion) if args.tam_poblacion is not None else None,
            "max_evals": int(args.max_evals) if args.max_evals is not None else None,
            "cec_funcid_raw": list(args.cec_funcid),
            "funcids_resueltas": [int(f) for f in funcids_cec],
            "cec_dim": int(args.cec_dim),
            "algoritmo": str(args.algoritmo),
            "problema": "cec2017",
            "registrar_metricas": bool(not args.sin_metricas),
            "reinicio_elitista": bool(args.reinicio_elitista),
            "reinicio_elitista_ratio_estabilidad_diversidad": None,
            "reinicio_elitista_ratio_paciencia": (
                float(args.reinicio_elitista_ratio_paciencia)
                if args.reinicio_elitista
                else None
            ),
            "reinicio_elitista_ventana_evaluaciones": (
                int(VENTANA_DIAGNOSTICA_REINICIO)
                if args.reinicio_elitista
                else None
            ),
        }
        filas_resumen_contexto = construir_resumen(filas_runs_cec, desglosar_contexto=True)
        guardar_csv(outdir_cec / "resumen_funciones_promedio.csv", filas_resumen_contexto)
        guardar_json(
            outdir_cec / "resumen_funciones_promedio.json",
            {
                "config": config_cec,
                "resumen": filas_resumen_contexto,
            },
        )

        mostrar(args, "\nPromedios CEC por funcion y algoritmo/adaptacion:", flush=True)
        for fila in filas_resumen_contexto:
            funcid_txt = f"f{int(fila['cec_funcid'])}" if fila.get("cec_funcid") not in ("", None) else "f?"
            mostrar(
                args,
                f"  {funcid_txt} / {fila['algoritmo']}: "
                f"fitness_promedio={fila['fitness_promedio']:.6f}, "
                f"tiempo_promedio_s={fila['tiempo_promedio_s']:.6f}",
                flush=True,
            )
        mostrar(args, f"\nResultados CEC guardados en: {outdir_cec}", flush=True)


if __name__ == "__main__":
    main()
