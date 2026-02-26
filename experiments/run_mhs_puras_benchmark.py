#!/usr/bin/env python3
# runner simplificado para MHs puras (AGE/DE) en CEC2017 y QAP.
# ejecuta 10 semillas por defecto y calcula el promedio de fitness y tiempo.
# usa siempre el logger DEAP ya integrado en los adaptadores.

import argparse
import csv
import json
from collections import defaultdict
import sys
import time
from pathlib import Path

import numpy as np

# permite ejecutar desde la raiz del repositorio sin instalar paquete
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaheuristics.age.adapted.genetic_stationary_cec2017 import GeneticStationaryCEC2017
from metaheuristics.age.adapted.genetic_stationary_qap import GeneticStationaryQAP
from metaheuristics.de.adapted.differential_evolution_cec2017 import DifferentialEvolutionCEC2017
from metaheuristics.de.adapted.differential_evolution_qap import DifferentialEvolutionQAP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta AGE/DE (CEC2017 + QAP) con 10 semillas y resume promedios"
    )
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

    parser.add_argument("--cec-funcid", type=int, default=1)
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


def guardar_csv(path, filas):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(filas) == 0:
        with path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow([])
        return

    campos = list(filas[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=campos)
        writer.writeheader()
        for fila in filas:
            writer.writerow(fila)


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


def ejecutar_cec(args, semillas, outdir_metricas, algoritmos):
    filas = []
    total_runs = len(algoritmos) * len(semillas)
    run_idx = 0
    
    for algoritmo in algoritmos:
        for seed in semillas:
            run_idx += 1
            print(
                f"[CEC2017 {run_idx}/{total_runs}] iniciando {algoritmo.upper()} con seed={seed}...",
                flush=True,
            )
            if algoritmo == "age":
                age_kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    age_kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    age_kwargs["max_evals"] = int(args.max_evals)
                solver = GeneticStationaryCEC2017(**age_kwargs)
                algname = f"age_cec_f{int(args.cec_funcid)}_d{int(args.cec_dim)}_s{int(seed)}"
            else:
                de_kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    de_kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    de_kwargs["max_evals"] = int(args.max_evals)
                solver = DifferentialEvolutionCEC2017(**de_kwargs)
                algname = f"de_cec_f{int(args.cec_funcid)}_d{int(args.cec_dim)}_s{int(seed)}"

            t0 = time.perf_counter()
            resultado = solver.optimize(
                funcid=args.cec_funcid,
                dim=args.cec_dim,
                seed=seed,
                algname=algname,
                registrar_metricas=True,
                ruta_metricas=str(outdir_metricas / "cec2017" / algoritmo),
                run_id=f"{algoritmo}_cec2017_f{int(args.cec_funcid)}_d{int(args.cec_dim)}_s{int(seed)}",
            )
            dt = time.perf_counter() - t0

            fila = {
                "problema": "cec2017",
                "algoritmo": algoritmo,
                "adaptacion": f"{algoritmo}_cec2017",
                "semilla": int(seed),
                "fitness": float(resultado["mejor_fitness"]),
                "tiempo_s": float(dt),
                "cec_error": float(resultado["mejor_error"]),
                "qap_instancia": "",
                "ruta_metricas": str(resultado.get("ruta_metricas", "")),
            }

            filas.append(fila)
            print(
                f"[CEC2017 {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed} "
                f"fitness={fila['fitness']:.6f} error={fila['cec_error']:.6f} tiempo={dt:.4f}s",
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
            print(
                f"[QAP {run_idx}/{total_runs}] iniciando {algoritmo.upper()} con seed={seed}...",
                flush=True,
            )
            if algoritmo == "age":
                age_kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    age_kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    age_kwargs["max_evals"] = int(args.max_evals)
                solver = GeneticStationaryQAP(**age_kwargs)
            else:
                de_kwargs = {"seed": seed}
                if args.tam_poblacion is not None:
                    de_kwargs["tam_poblacion"] = int(args.tam_poblacion)
                if args.max_evals is not None:
                    de_kwargs["max_evals"] = int(args.max_evals)
                solver = DifferentialEvolutionQAP(**de_kwargs)

            t0 = time.perf_counter()
            resultado = solver.optimize(
                qap_path=str(qap_path),
                seed=seed,
                registrar_metricas=True,
                ruta_metricas=str(outdir_metricas / "qap" / algoritmo),
                run_id=f"{algoritmo}_qap_{instancia}_s{int(seed)}",
            )
            dt = time.perf_counter() - t0

            fila = {
                "problema": "qap",
                "algoritmo": algoritmo,
                "adaptacion": f"{algoritmo}_qap",
                "semilla": int(seed),
                "fitness": float(resultado["mejor_fitness"]),
                "tiempo_s": float(dt),
                "cec_error": "",
                "qap_instancia": instancia,
                "ruta_metricas": str(resultado.get("ruta_metricas", "")),
            }

            filas.append(fila)
            print(
                f"[QAP {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed} "
                f"fitness={fila['fitness']:.6f} tiempo={dt:.4f}s",
                flush=True,
            )
    return filas


def construir_resumen(filas_runs):
    grupos = defaultdict(list)
    for fila in filas_runs:
        clave = (fila["problema"], fila["algoritmo"], fila["adaptacion"])
        grupos[clave].append(fila)

    filas_resumen = []
    for (problema, algoritmo, adaptacion), filas in sorted(grupos.items()):
        resumen = resumen_grupo(filas)
        filas_resumen.append(
            {
                "problema": problema,
                "algoritmo": algoritmo,
                "adaptacion": adaptacion,
                **resumen,
            }
        )
    return filas_resumen


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outdir_metricas = outdir / "metricas_runs"

    semillas = [args.seed_start + i for i in range(args.n_seeds)]
    if len(semillas) == 0:
        raise ValueError("n-seeds debe ser >= 1")

    if args.algoritmo == "ambos":
        algoritmos = ("age", "de")
    else:
        algoritmos = (args.algoritmo,)

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
    print(f"  cec: funcid={args.cec_funcid}, dim={args.cec_dim}", flush=True)
    print(f"  qap_path={args.qap_path}", flush=True)
    print(f"  algoritmo={args.algoritmo}", flush=True)
    print(f"  problema={args.problema}", flush=True)
    print("  registrar_metricas=True (DEAP)", flush=True)
    print(f"  outdir={outdir}", flush=True)

    filas_runs = []
    if args.problema in ("cec2017", "ambos"):
        filas_runs.extend(ejecutar_cec(args, semillas, outdir_metricas, algoritmos))
    if args.problema in ("qap", "ambos"):
        filas_runs.extend(ejecutar_qap(args, semillas, outdir_metricas, algoritmos))

    if len(filas_runs) == 0:
        raise RuntimeError("No se ejecuto ninguna combinacion. Revisa --algoritmo y --problema.")

    filas_resumen = construir_resumen(filas_runs)

    guardar_csv(outdir / "runs.csv", filas_runs)
    guardar_csv(outdir / "resumen_promedios.csv", filas_resumen)
    guardar_json(
        outdir / "resumen_promedios.json",
        {
            "config": {
                "seed_start": int(args.seed_start),
                "n_seeds": int(args.n_seeds),
                "tam_poblacion": int(args.tam_poblacion) if args.tam_poblacion is not None else None,
                "max_evals": int(args.max_evals) if args.max_evals is not None else None,
                "cec_funcid": int(args.cec_funcid),
                "cec_dim": int(args.cec_dim),
                "qap_path": str(args.qap_path),
                "algoritmo": str(args.algoritmo),
                "problema": str(args.problema),
                "registrar_metricas": True,
            },
            "resumen": filas_resumen,
        },
    )

    print("\nPromedios por algoritmo/adaptacion:", flush=True)
    for fila in filas_resumen:
        print(
            f"  {fila['adaptacion']}: "
            f"fitness_promedio={fila['fitness_promedio']:.6f}, "
            f"tiempo_promedio_s={fila['tiempo_promedio_s']:.6f}",
            flush=True,
        )
    print(f"\nResultados guardados en: {outdir}", flush=True)


if __name__ == "__main__":
    main()
