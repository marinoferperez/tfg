from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agrega por configuracion los resultados de validacion interna "
            "guardados por ajustar_hiperparametros.py."
        )
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--model", required=True, choices=["rbf", "rsm"])
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--winner-json", default=None)
    return parser


def canonicalizar_parametros(params: dict) -> str:
    return json.dumps(params, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def rutas_metricas_raiz(benchmark_dir: Path, model: str) -> list[Path]:
    return sorted(
        path
        for path in benchmark_dir.rglob(f"{model}_metricas.json")
        if path.parent.name == model
    )


def cargar_resultados(benchmark_dir: Path, model: str) -> tuple[list[dict], list[Path]]:
    filas = []
    rutas = rutas_metricas_raiz(benchmark_dir, model)
    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron metricas raiz para model={model!r} en {benchmark_dir}."
        )

    for path in rutas:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for run in payload.get("runs", []):
            resultados = run.get("tuning_resultados")
            if resultados is None:
                raise ValueError(
                    f"{path} no contiene tuning_resultados. "
                    "Ejecuta el runner con --store-tuning-results."
                )
            for resultado in resultados:
                metricas = resultado.get("metricas") or {}
                filas.append(
                    {
                        "params": dict(resultado["params"]),
                        "score": float(resultado["score"]),
                        "spearman": float(metricas.get("spearman", float("nan"))),
                        "nrmse": float(metricas.get("nrmse", float("nan"))),
                        "nmae": float(metricas.get("nmae", float("nan"))),
                        "train_time_s": float(resultado.get("train_time_s", float("nan"))),
                        "predict_time_s": float(resultado.get("predict_time_s", float("nan"))),
                        "error": resultado.get("error"),
                    }
                )
    return filas, rutas


def media_finita(valores: list[float]) -> float:
    validos = np.asarray([valor for valor in valores if math.isfinite(valor)], dtype=float)
    if validos.size == 0:
        return float("nan")
    return float(np.mean(validos))


def desviacion_finita(valores: list[float]) -> float:
    validos = np.asarray([valor for valor in valores if math.isfinite(valor)], dtype=float)
    if validos.size == 0:
        return float("nan")
    return float(np.std(validos))


def agregar_por_configuracion(filas: list[dict]) -> list[dict]:
    grupos = defaultdict(list)
    for fila in filas:
        grupos[canonicalizar_parametros(fila["params"])].append(fila)

    resumen = []
    for params_json, resultados in grupos.items():
        scores = [fila["score"] for fila in resultados]
        spearman = [fila["spearman"] for fila in resultados]
        nrmse = [fila["nrmse"] for fila in resultados]
        nmae = [fila["nmae"] for fila in resultados]
        train_time_s = [fila["train_time_s"] for fila in resultados]
        predict_time_s = [fila["predict_time_s"] for fila in resultados]
        resumen.append(
            {
                "params": json.loads(params_json),
                "params_json": params_json,
                "n_casos": len(resultados),
                "n_validos": sum(math.isfinite(valor) for valor in scores),
                "n_errores": sum(bool(fila["error"]) for fila in resultados),
                "spearman_interno_medio": media_finita(spearman),
                "spearman_interno_std": desviacion_finita(spearman),
                "nrmse_interno_medio": media_finita(nrmse),
                "nmae_interno_medio": media_finita(nmae),
                "train_time_s_interno_medio": media_finita(train_time_s),
                "train_time_s_interno_std": desviacion_finita(train_time_s),
                "predict_time_s_interno_medio": media_finita(predict_time_s),
                "predict_time_s_interno_std": desviacion_finita(predict_time_s),
            }
        )

    resumen.sort(
        key=lambda fila: (
            -fila["spearman_interno_medio"]
            if math.isfinite(fila["spearman_interno_medio"])
            else float("inf"),
            fila["params_json"],
        )
    )
    for posicion, fila in enumerate(resumen, start=1):
        fila["posicion"] = posicion
    return resumen


def escribir_csv(path: Path, filas: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "posicion",
        "params_json",
        "n_casos",
        "n_validos",
        "n_errores",
        "spearman_interno_medio",
        "spearman_interno_std",
        "nrmse_interno_medio",
        "nmae_interno_medio",
        "train_time_s_interno_medio",
        "train_time_s_interno_std",
        "predict_time_s_interno_medio",
        "predict_time_s_interno_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=campos, lineterminator="\n")
        writer.writeheader()
        for fila in filas:
            writer.writerow({campo: fila[campo] for campo in campos})


def main() -> None:
    args = build_parser().parse_args()
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"No existe benchmark-dir: {benchmark_dir}")

    out_csv = (
        Path(args.out_csv).expanduser().resolve()
        if args.out_csv
        else benchmark_dir / f"{args.model}_resumen_ajuste_interno.csv"
    )
    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else benchmark_dir / f"{args.model}_resumen_ajuste_interno.json"
    )
    winner_json = (
        Path(args.winner_json).expanduser().resolve()
        if args.winner_json
        else benchmark_dir / f"{args.model}_mejor_config.json"
    )

    filas, rutas = cargar_resultados(benchmark_dir, args.model)
    resumen = agregar_por_configuracion(filas)
    if not resumen:
        raise ValueError("No se encontraron resultados internos para agregar.")

    escribir_csv(out_csv, resumen)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "model": args.model,
                "benchmark_dir": str(benchmark_dir),
                "n_archivos_metricas": len(rutas),
                "n_evaluaciones_internas": len(filas),
                "n_configuraciones": len(resumen),
                "configuraciones": resumen,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    winner_json.parent.mkdir(parents=True, exist_ok=True)
    winner_json.write_text(
        json.dumps(resumen[0]["params"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Modelo: {args.model}")
    print(f"Archivos de metricas: {len(rutas)}")
    print(f"Evaluaciones internas: {len(filas)}")
    print(f"Configuraciones: {len(resumen)}")
    print(f"Mejor configuracion: {resumen[0]['params_json']}")
    print(f"Spearman interno medio: {resumen[0]['spearman_interno_medio']:.6f}")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")
    print(f"Ganadora: {winner_json}")


if __name__ == "__main__":
    main()
