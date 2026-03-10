# ejecuta x semillas por defecto y calcula el promedio de fitness y tiempo.
# usa siempre el logger DEAP ya integrado en los adaptadores.

import argparse
import csv
import json
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

# permite ejecutar desde la raiz del repositorio sin instalar paquete
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaheuristics.age.adapted.genetic_stationary_cec2017 import GeneticStationaryCEC2017
from metaheuristics.age.adapted.genetic_stationary_qap import GeneticStationaryQAP
from metaheuristics.de.adapted.differential_evolution_cec2017 import DifferentialEvolutionCEC2017
from metaheuristics.de.adapted.differential_evolution_qap import DifferentialEvolutionQAP

# parse_args determina las flags y posibles opciones de ejecucion
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=10)

    parser.add_argument(
        "--tam-poblacion",
        type=int,
        default=None,
        help="Override opcional. Si no se indica, cada algoritmo usa su valor original.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Override opcional. Si no se indica, cada algoritmo usa su valor original.",
    )

    parser.add_argument(
        "--cec-funcid",
        nargs="+",
        default=["1"],
        help="Funciones CEC a ejecutar: lista (ej. --cec-funcid 1 2 3), CSV (ej. --cec-funcid 1,2,3) o 'all'.",
    )
    parser.add_argument("--cec-dim", type=int, default=10)
    parser.add_argument("--qap-path", type=str, default="problems/qaplib_small/nug12.dat")
    parser.add_argument(
        "--algoritmo",
        type=str,
        default="ambos",
        choices=["age", "de", "ambos"],
        help="Algoritmo a ejecutar",
    )
    parser.add_argument(
        "--problema",
        type=str,
        default="ambos",
        choices=["cec2017", "qap", "ambos"],
        help="Problema a ejecutar",
    )

    parser.add_argument("--outdir", type=str, default="results/experimentos_mhs")
    return parser.parse_args()


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


def guardar_bloque_resultados(outdir, filas_runs, config_json):
    filas_resumen = construir_resumen(filas_runs)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        with path.open("w", encoding="utf-8", newline="") as f_out:
            f_out.write("")
        return

    campos = list(filas[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=campos, lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)

def guardar_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, ensure_ascii=False, indent=2)

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

    for algoritmo in algoritmos:
        for seed in semillas:
            run_idx += 1
            print(
                f"[CEC2017 F{int(funcid)} {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed}...",
                flush=True,
            )

            if algoritmo == "age":
                kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    kwargs["max_evals"] = int(args.max_evals)
                mtheuristica = GeneticStationaryCEC2017(**kwargs)
                algname = "age"
            else:
                kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    kwargs["max_evals"] = int(args.max_evals)
                mtheuristica = DifferentialEvolutionCEC2017(**kwargs)
                algname = "de"

            t0 = time.perf_counter()
            resultado = mtheuristica.optimize(
                funcid = int(funcid),
                dim = args.cec_dim,
                seed = seed,
                algname = algname,
                registrar_metricas = True,
                ruta_metricas = str(outdir_metricas / "cec2017" / algoritmo),
                run_id = f"{algoritmo}_cec2017_f{int(funcid)}_d{int(args.cec_dim)}_s{int(seed)}",
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
            })

            print(
                f"[CEC2017 F{int(funcid)} {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed} "
                f"fitness={float(resultado['mejor_fitness']):.6f} "
                f"error={float(resultado['mejor_error']):.6f} "
                f"tiempo={dt:.4f}s",
                flush=True,
            )
    return filas

def ejecutar_qap(args, semillas, outdir_metricas, algoritmos):
    filas = []
    qap_path = Path(args.qap_path)
    instancia = qap_path.stem
    total_runs = len(algoritmos) * len(semillas)
    run_idx = 0

    for algoritmo in algoritmos:
        for seed in semillas:
            run_idx += 1
            print(f"[QAP {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed}...", flush=True)

            if algoritmo == "age":
                kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    kwargs["max_evals"] = int(args.max_evals)
                mtheuristica = GeneticStationaryQAP(**kwargs)
            else:
                kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    kwargs["max_evals"] = int(args.max_evals)
                mtheuristica = DifferentialEvolutionQAP(**kwargs)

            t0 = time.perf_counter()
            resultado = mtheuristica.optimize(
                qap_path=str(qap_path),
                seed=seed,
                registrar_metricas=True,
                ruta_metricas=str(outdir_metricas / "qap" / algoritmo),
                run_id=f"{algoritmo}_qap_{instancia}_s{int(seed)}",
            )
            dt = time.perf_counter() - t0

            filas.append({
                "problema": "qap",
                "algoritmo": algoritmo,
                "adaptacion": f"{algoritmo}_qap",
                "cec_funcid": "",
                "qap_instancia": instancia,
                "semilla": int(seed),
                "fitness": float(resultado["mejor_fitness"]),
                "tiempo_s": float(dt),  # TIEMPO REAL
                "cec_error": "",
                "ruta_metricas": str(resultado.get("ruta_metricas", "")),
            })

            print(
                f"[QAP {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed} "
                f"fitness={float(resultado['mejor_fitness']):.6f} "
                f"tiempo={dt:.4f}s",
                flush=True,
            )
    return filas

def construir_resumen(filas_runs, desglosar_contexto=False):
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
        filas_resumen.append(
            {
                "problema": problema,
                "algoritmo": algoritmo,
                "adaptacion": adaptacion,
                "cec_funcid": cec_funcid,
                "qap_instancia": qap_instancia,
                **resumen,
            }
        )
    return filas_resumen


def main():
    args = parse_args()
    outdir = Path(args.outdir)

    semillas = [args.seed_start + i for i in range(args.n_seeds)]
    if len(semillas) == 0:
        raise ValueError("n-seeds debe ser >= 1")

    if args.algoritmo == "ambos":
        algoritmos = ("age", "de")
    else:
        algoritmos = (args.algoritmo,)

    if args.problema in ("cec2017", "ambos"):
        funcids_cec = gestiona_funcids_cec(args)
        funcids_txt = ",".join(str(int(f)) for f in funcids_cec)
    else:
        funcids_cec = [1]
        funcids_txt = ",".join(str(v) for v in args.cec_funcid)

    print("Configuracion:", flush=True)
    print(f"  semillas={semillas[0]}..{semillas[-1]} (n={len(semillas)})", flush=True)
    print(
        "  tam_poblacion="
        + (
            str(int(args.tam_poblacion))
            if args.tam_poblacion is not None
            else "default_original_algoritmo"
        ),
        flush=True,
    )
    print(
        "  max_evals="
        + (
            str(int(args.max_evals))
            if args.max_evals is not None
            else "default_original_algoritmo"
        ),
        flush=True,
    )
    print(f"  cec: funcid={funcids_txt}, dim={args.cec_dim}", flush=True)
    print(f"  qap_path={args.qap_path}", flush=True)
    print(f"  algoritmo={args.algoritmo}", flush=True)
    print(f"  problema={args.problema}", flush=True)
    print("  registrar_metricas=True (DEAP)", flush=True)
    print(f"  outdir={outdir}", flush=True)

    filas_runs = []
    if args.problema in ("cec2017", "ambos"):
        total_funcids = len(funcids_cec)
        for idx, funcid in enumerate(funcids_cec, start=1):
            print(
                f"\nCEC2017: F{int(funcid)} ({idx}/{total_funcids})",
                flush=True,
            )
            outdir_funcid = outdir / f"f{int(funcid)}"
            outdir_metricas_funcid = outdir_funcid / "metricas_runs"
            filas_funcid = ejecutar_cec(args, semillas, outdir_metricas_funcid, algoritmos, int(funcid))
            filas_runs.extend(filas_funcid)

            config_funcid = {
                "seed_start": int(args.seed_start),
                "n_seeds": int(args.n_seeds),
                "tam_poblacion": int(args.tam_poblacion) if args.tam_poblacion is not None else None,
                "max_evals": int(args.max_evals) if args.max_evals is not None else None,
                "funcid": int(funcid),
                "cec_dim": int(args.cec_dim),
                "algoritmo": str(args.algoritmo),
                "problema": "cec2017",
                "registrar_metricas": True,
            }
            guardar_bloque_resultados(outdir_funcid, filas_funcid, config_funcid)

    if args.problema in ("qap", "ambos"):
        outdir_metricas_qap = outdir / "metricas_runs"
        filas_runs.extend(ejecutar_qap(args, semillas, outdir_metricas_qap, algoritmos))

    if len(filas_runs) == 0:
        raise RuntimeError("No se ejecuto ninguna combinacion. Revisa --algoritmo y --problema.")

    config_global = {
        "seed_start": int(args.seed_start),
        "n_seeds": int(args.n_seeds),
        "tam_poblacion": int(args.tam_poblacion) if args.tam_poblacion is not None else None,
        "max_evals": int(args.max_evals) if args.max_evals is not None else None,
        "cec_funcid_raw": list(args.cec_funcid),
        "funcids_resueltas": [int(f) for f in funcids_cec],
        "cec_dim": int(args.cec_dim),
        "qap_path": str(args.qap_path),
        "algoritmo": str(args.algoritmo),
        "problema": str(args.problema),
        "registrar_metricas": True,
    }
    filas_resumen_global = guardar_bloque_resultados(outdir, filas_runs, config_global)

    filas_resumen_contexto = construir_resumen(filas_runs, desglosar_contexto=True)
    guardar_csv(outdir / "resumen_promedios_contexto.csv", filas_resumen_contexto)
    guardar_json(
        outdir / "resumen_promedios_contexto.json",
        {
            "config": config_global,
            "resumen": filas_resumen_contexto,
        },
    )

    print("\nPromedios por algoritmo/adaptacion:", flush=True)
    for fila in filas_resumen_global:
        print(
            f"  {fila['adaptacion']}: "
            f"fitness_promedio={fila['fitness_promedio']:.6f}, "
            f"tiempo_promedio_s={fila['tiempo_promedio_s']:.6f}",
            flush=True,
        )
    print(f"\nResultados guardados en: {outdir}", flush=True)


if __name__ == "__main__":
    main()
