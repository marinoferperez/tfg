import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANGO_INF_CEC = -100.0
RANGO_SUP_CEC = 100.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Migra runs CEC2017 antiguas para unificar logbook y diversidad en "
            "resultados_*.csv, eliminando dataset_*_metricas.csv si se solicita."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Raíz del experimento, carpeta fX o carpeta metricas_runs/cec2017/<algoritmo>. "
            "El script localizará automáticamente las carpetas de run."
        ),
    )
    parser.add_argument(
        "--eliminar-antiguos",
        action="store_true",
        help="Elimina logbook.csv y dataset_*_metricas.csv tras generar resultados_*.csv.",
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Sobrescribe resultados_*.csv aunque ya exista.",
    )
    return parser.parse_args()


def inferir_run_dirs(root: Path):
    if root.is_file():
        raise ValueError("--root debe ser un directorio.")

    # run dir directo
    if any(root.glob("dataset_*.h5")) or any(root.glob("dataset_*_simple.csv")):
        return [root]

    candidatos = []
    for patron in [
        "cec2017/*/*",
        "metricas_runs/cec2017/*/*",
        "f*/metricas_runs/cec2017/*/*",
        "*/*/*",
        "*/*",
    ]:
        candidatos.extend(root.glob(patron))

    run_dirs = []
    vistos = set()
    for path in candidatos:
        if not path.is_dir():
            continue
        if not (any(path.glob("dataset_*.h5")) or any(path.glob("dataset_*_simple.csv"))):
            continue
        key = str(path.resolve())
        if key not in vistos:
            vistos.add(key)
            run_dirs.append(path)
    return sorted(run_dirs)


def resolver_dataset_df(run_dir: Path):
    rutas_h5 = sorted(run_dir.glob("dataset_*.h5"))
    if rutas_h5:
        return pd.read_hdf(rutas_h5[0], key="dataset"), rutas_h5[0]

    rutas_csv = sorted(run_dir.glob("dataset_*_simple.csv"))
    if rutas_csv:
        return pd.read_csv(rutas_csv[0]), rutas_csv[0]

    raise FileNotFoundError(f"No se encontró dataset en {run_dir}")


def resolver_metricas_generacion_csv(run_dir: Path):
    rutas = sorted(run_dir.glob("dataset_*_metricas.csv"))
    return rutas[0] if rutas else None


def resolver_csv_origen(run_dir: Path):
    rutas_resultados = sorted(run_dir.glob("resultados_*.csv"))
    if rutas_resultados:
        return rutas_resultados[0]
    ruta_logbook = run_dir / "logbook.csv"
    if ruta_logbook.exists():
        return ruta_logbook
    raise FileNotFoundError(f"No se encontró resultados_*.csv ni logbook.csv en {run_dir}")


def construir_nombre_resultados(desde_dataset: Path):
    stem = desde_dataset.stem
    if stem.endswith("_simple"):
        stem = stem[: -len("_simple")]
    if not stem.startswith("dataset_"):
        return "resultados_mh_resultados.csv"
    return f"resultados_{stem[len('dataset_'):]}.csv"


def calcular_diversidad(df_dataset: pd.DataFrame, eval_inicio: int, eval_fin: int):
    columnas_x = [c for c in df_dataset.columns if c.startswith("x_")]
    if len(columnas_x) == 0:
        return np.nan, np.nan

    bloque = df_dataset.loc[
        (df_dataset["eval_id"] >= int(eval_inicio)) & (df_dataset["eval_id"] <= int(eval_fin)),
        columnas_x,
    ]
    if bloque.shape[0] < 2:
        return np.nan, np.nan

    poblacion = bloque.to_numpy(dtype=float)
    centroide = np.mean(poblacion, axis=0)
    dists = np.linalg.norm(poblacion - centroide, axis=1)
    div = float(np.mean(dists))
    dimension = int(poblacion.shape[1])
    div_norm = div / ((RANGO_SUP_CEC - RANGO_INF_CEC) * dimension)
    return div, div_norm


def enriquecer_resultados_desde_metricas_csv(
    df_resultados: pd.DataFrame,
    ruta_metricas_csv: Path,
    dimension: int,
):
    df = df_resultados.copy()
    if "generacion" not in df.columns:
        raise ValueError("El CSV origen no tiene columna 'generacion'.")

    df_metricas = pd.read_csv(ruta_metricas_csv)
    columnas_requeridas = {
        "generacion",
        "eval_id_inicio",
        "eval_id_fin",
        "div_dist_euclidea",
    }
    if not columnas_requeridas.issubset(df_metricas.columns):
        raise ValueError(f"{ruta_metricas_csv.name} no contiene las columnas requeridas.")

    columnas_merge = [
        "generacion",
        "eval_id_inicio",
        "eval_id_fin",
        "div_dist_euclidea",
    ]
    df = df.merge(
        df_metricas[columnas_merge],
        on="generacion",
        how="left",
        validate="one_to_one",
    )

    dimension = max(int(dimension), 1)
    df["div_dist_euclidea_normalizada"] = (
        pd.to_numeric(df["div_dist_euclidea"], errors="coerce")
        / ((RANGO_SUP_CEC - RANGO_INF_CEC) * dimension)
    )
    return df


def enriquecer_resultados_desde_dataset(df_resultados: pd.DataFrame, df_dataset: pd.DataFrame):
    df = df_resultados.copy()
    if "evaluaciones" not in df.columns:
        raise ValueError("El CSV origen no tiene columna 'evaluaciones'.")

    evals = pd.to_numeric(df["evaluaciones"], errors="coerce")
    if evals.isna().any():
        raise ValueError("Hay filas con 'evaluaciones' no numéricas.")
    evals = evals.astype(int).to_numpy()

    eval_inicio = []
    eval_fin = []
    divs = []
    divs_norm = []

    prev_end = 0
    for current_end in evals:
        if current_end < prev_end:
            raise ValueError("Las evaluaciones no son monótonas.")
        inicio = prev_end + 1
        fin = int(current_end)
        div, div_norm = calcular_diversidad(df_dataset, inicio, fin)
        eval_inicio.append(inicio)
        eval_fin.append(fin)
        divs.append(div)
        divs_norm.append(div_norm)
        prev_end = fin

    df["eval_id_inicio"] = eval_inicio
    df["eval_id_fin"] = eval_fin
    df["div_dist_euclidea"] = divs
    df["div_dist_euclidea_normalizada"] = divs_norm
    return df


def migrar_run(run_dir: Path, sobrescribir: bool, eliminar_antiguos: bool):
    df_dataset, ruta_dataset = resolver_dataset_df(run_dir)
    ruta_origen = resolver_csv_origen(run_dir)
    ruta_destino = run_dir / construir_nombre_resultados(ruta_dataset)

    if ruta_destino.exists() and ruta_origen.resolve() != ruta_destino.resolve() and not sobrescribir:
        return {
            "run_dir": str(run_dir),
            "estado": "omitido",
            "motivo": f"{ruta_destino.name} ya existe",
        }

    df_resultados = pd.read_csv(ruta_origen)
    ruta_metricas_csv = resolver_metricas_generacion_csv(run_dir)
    if ruta_metricas_csv is not None:
        dimension = len([c for c in df_dataset.columns if c.startswith("x_")])
        df_migrado = enriquecer_resultados_desde_metricas_csv(
            df_resultados,
            ruta_metricas_csv,
            dimension=dimension,
        )
    else:
        df_migrado = enriquecer_resultados_desde_dataset(df_resultados, df_dataset)
    df_migrado.to_csv(ruta_destino, index=False)

    eliminados = []
    if eliminar_antiguos:
        ruta_logbook = run_dir / "logbook.csv"
        if ruta_logbook.exists() and ruta_logbook.resolve() != ruta_destino.resolve():
            ruta_logbook.unlink()
            eliminados.append(ruta_logbook.name)
        for ruta_metricas in run_dir.glob("dataset_*_metricas.csv"):
            ruta_metricas.unlink()
            eliminados.append(ruta_metricas.name)

    return {
        "run_dir": str(run_dir),
        "estado": "ok",
        "csv_salida": str(ruta_destino),
        "origen": str(ruta_origen),
        "eliminados": eliminados,
    }


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    run_dirs = inferir_run_dirs(root)
    if not run_dirs:
        raise SystemExit(f"No se encontraron runs bajo {root}")

    resultados = []
    errores = []
    for run_dir in run_dirs:
        try:
            resultados.append(
                migrar_run(
                    run_dir,
                    sobrescribir=bool(args.sobrescribir),
                    eliminar_antiguos=bool(args.eliminar_antiguos),
                )
            )
        except Exception as exc:
            errores.append((str(run_dir), str(exc)))

    ok = sum(1 for r in resultados if r["estado"] == "ok")
    omitidos = sum(1 for r in resultados if r["estado"] == "omitido")

    print(f"Runs procesadas: {len(run_dirs)}")
    print(f"OK: {ok}")
    print(f"Omitidas: {omitidos}")
    print(f"Errores: {len(errores)}")
    for run_dir, msg in errores[:20]:
        print(f"[ERROR] {run_dir}: {msg}")


if __name__ == "__main__":
    main()
