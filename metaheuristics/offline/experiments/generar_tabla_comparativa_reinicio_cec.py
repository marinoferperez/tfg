#!/usr/bin/env python3
"""Genera una tabla comparativa CEC2017 entre una base y una variante.

El script lee los ``runs.csv`` de dos directorios de resultados y compara la
mediana de ``cec_error`` usando las funciones y semillas comunes. Esta politica
permite comparar pilotos con menos semillas contra una ejecucion base completa
sin sesgar la base por usar semillas extra.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

import numpy as np


ALGORITMOS = ("age", "de", "shade")
NOMBRE_ALGORITMO = {
    "age": "AGE",
    "de": "DE",
    "shade": "SHADE",
}


def familia_cec2017(funcid):
    funcid = int(funcid)
    if funcid in (1, 2, 3):
        return "Unimodal"
    if 4 <= funcid <= 10:
        return "Multimodal"
    if 11 <= funcid <= 20:
        return "Hibrida"
    if 21 <= funcid <= 30:
        return "Compuesta"
    return ""


def familia_latex(nombre):
    if nombre == "Hibrida":
        return "H\\'ibrida"
    return nombre


def parse_lista_enteros(raw):
    valores = []
    for parte in raw or []:
        for token in str(parte).split(","):
            token = token.strip()
            if token:
                valores.append(int(token))
    return valores


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera CSV y LaTeX comparando una ejecucion base CEC2017 frente a "
            "una variante, usando funciones y semillas comunes por defecto."
        )
    )
    parser.add_argument(
        "--root-base",
        required=True,
        help="Directorio raiz de resultados base, con subdirectorios f*/runs.csv.",
    )
    parser.add_argument(
        "--root-variante",
        required=True,
        help="Directorio raiz de resultados de la variante a comparar.",
    )
    parser.add_argument(
        "--label-base",
        default="Sin reinicio",
        help="Etiqueta de la configuracion base en la tabla LaTeX.",
    )
    parser.add_argument(
        "--label-variante",
        default="Variante",
        help="Etiqueta de la variante en la tabla LaTeX.",
    )
    parser.add_argument(
        "--output-dir",
        default="informes/benchmarks/generated/reinicio_primero_mejor",
        help="Directorio donde se guardan los CSV y LaTeX generados.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefijo de los ficheros generados.",
    )
    parser.add_argument(
        "--funciones",
        nargs="*",
        default=None,
        help=(
            "Funciones CEC a incluir. Acepta espacios o comas. Si no se indica, "
            "se usan las funciones comunes entre base y variante."
        ),
    )
    parser.add_argument(
        "--semillas",
        nargs="*",
        default=None,
        help=(
            "Semillas a incluir. Acepta espacios o comas. Si no se indica, se "
            "usan las semillas comunes por funcion y algoritmo."
        ),
    )
    parser.add_argument(
        "--n-semillas-aleatorias",
        type=int,
        default=None,
        help=(
            "Si se indica, selecciona de forma reproducible este numero de "
            "semillas. Si tambien se indica --semillas, muestrea dentro de esa "
            "lista; si no, muestrea dentro de las semillas comunes disponibles."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla del muestreo aleatorio usado por --n-semillas-aleatorias.",
    )
    parser.add_argument(
        "--metric",
        default="cec_error",
        choices=("cec_error", "fitness"),
        help="Metrica de runs.csv sobre la que calcular la mediana.",
    )
    return parser.parse_args()


def leer_runs(root, metric):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio: {root}")

    datos = {}
    runs_paths = sorted(
        root.glob("f*/runs.csv"),
        key=lambda path: int(path.parent.name[1:]),
    )
    if not runs_paths:
        raise RuntimeError(f"No se encontraron f*/runs.csv en {root}")

    for path in runs_paths:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for fila in csv.DictReader(fh):
                algoritmo = str(fila.get("algoritmo", "")).strip().lower()
                if algoritmo not in ALGORITMOS:
                    continue
                valor_raw = str(fila.get(metric, "")).strip()
                if valor_raw == "":
                    continue
                funcid = int(float(fila["cec_funcid"]))
                semilla = int(float(fila["semilla"]))
                datos[(funcid, algoritmo, semilla)] = float(valor_raw)
    return datos


def escribir_csv(path, filas):
    if not filas:
        raise ValueError(f"No hay filas para escribir en {path}")
    fieldnames = list(filas[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)


def mediana(valores):
    return float(statistics.median([float(v) for v in valores]))


def funciones_comunes(base, variante, funciones_explicit):
    funciones_base = {func for func, _, _ in base}
    funciones_variante = {func for func, _, _ in variante}
    if funciones_explicit:
        funciones = sorted(set(funciones_explicit))
        faltan = [f for f in funciones if f not in funciones_base or f not in funciones_variante]
        if faltan:
            raise RuntimeError(f"Funciones no disponibles en ambas ejecuciones: {faltan}")
        return funciones
    funciones = sorted(funciones_base & funciones_variante)
    if not funciones:
        raise RuntimeError("No hay funciones comunes entre base y variante.")
    return funciones


def semillas_comunes_globales(base, variante, funciones):
    comunes = None
    for funcid in funciones:
        for algoritmo in ALGORITMOS:
            semillas_base = {s for f, a, s in base if f == funcid and a == algoritmo}
            semillas_variante = {s for f, a, s in variante if f == funcid and a == algoritmo}
            semillas = semillas_base & semillas_variante
            comunes = set(semillas) if comunes is None else comunes & semillas
    if not comunes:
        raise RuntimeError(
            "No hay semillas comunes globales para todas las funciones y algoritmos."
        )
    return sorted(comunes)


def seleccionar_semillas_aleatorias(candidatas, n_semillas, random_state):
    candidatas = sorted(set(int(s) for s in candidatas))
    if int(n_semillas) <= 0:
        raise ValueError("--n-semillas-aleatorias debe ser > 0.")
    if int(n_semillas) > len(candidatas):
        raise ValueError(
            f"No se pueden seleccionar {n_semillas} semillas de solo "
            f"{len(candidatas)} candidatas."
        )
    rng = np.random.default_rng(int(random_state))
    seleccion = rng.choice(candidatas, size=int(n_semillas), replace=False)
    return sorted(int(s) for s in seleccion)


def formatear_numero_latex(valor):
    valor = float(valor)
    if not math.isfinite(valor):
        return "--"
    if abs(valor) < 10000:
        return f"\\num{{{valor:.3f}}}"
    return (
        "\\num[scientific-notation=true, round-mode=figures, "
        f"round-precision=3]{{{valor:.2e}}}"
    )


def formatear_mejor(base, variante, valor):
    txt = formatear_numero_latex(valor)
    if float(valor) == min(float(base), float(variante)):
        return f"\\textbf{{{txt}}}"
    return txt


def construir_filas(base, variante, funciones, semillas_explicit):
    long_rows = []
    wide_rows = []
    for funcid in funciones:
        wide = {
            "familia": familia_cec2017(funcid),
            "cec_funcid": int(funcid),
            "funcion": f"f{int(funcid)}",
        }
        for algoritmo in ALGORITMOS:
            seeds_base = {s for f, a, s in base if f == funcid and a == algoritmo}
            seeds_variante = {s for f, a, s in variante if f == funcid and a == algoritmo}
            if semillas_explicit:
                semillas = sorted(set(semillas_explicit))
                faltan = [
                    s for s in semillas
                    if (funcid, algoritmo, s) not in base
                    or (funcid, algoritmo, s) not in variante
                ]
                if faltan:
                    raise RuntimeError(
                        f"Faltan semillas para f{funcid}/{algoritmo}: {faltan}"
                    )
            else:
                semillas = sorted(seeds_base & seeds_variante)
            if not semillas:
                raise RuntimeError(
                    f"No hay semillas comunes para f{funcid}/{algoritmo}."
                )

            valores_base = [base[(funcid, algoritmo, s)] for s in semillas]
            valores_variante = [variante[(funcid, algoritmo, s)] for s in semillas]
            med_base = mediana(valores_base)
            med_variante = mediana(valores_variante)
            delta = med_variante - med_base
            ratio = math.nan if med_base == 0.0 else med_variante / med_base

            long_rows.append(
                {
                    "familia": familia_cec2017(funcid),
                    "cec_funcid": int(funcid),
                    "funcion": f"f{int(funcid)}",
                    "algoritmo": algoritmo,
                    "n_seeds": len(semillas),
                    "semillas": " ".join(str(s) for s in semillas),
                    "base_mediana": med_base,
                    "variante_mediana": med_variante,
                    "delta_variante_menos_base": delta,
                    "ratio_variante_sobre_base": ratio,
                }
            )

            label = NOMBRE_ALGORITMO[algoritmo]
            wide[f"{label}_base"] = med_base
            wide[f"{label}_variante"] = med_variante
            wide[f"{label}_delta"] = delta
            wide[f"{label}_n_seeds"] = len(semillas)
        wide_rows.append(wide)
    return long_rows, wide_rows


def escribir_latex(path, wide_rows, args):
    lines = [
        "% AUTO-GENERATED: comparativa CEC2017 base vs variante",
        "\\begin{table}[H]",
        "    \\centering",
        (
            "    \\caption{Mediana del error final de la configuracion base "
            f"({args.label_base}) y la variante ({args.label_variante}), "
            "calculada sobre las semillas comunes disponibles.}"
        ),
        f"    \\label{{tab:{args.output_prefix}}}",
        "    \\adjustbox{max width=\\textwidth}{%",
        "    \\begin{tabular}{llrrrrrr}",
        "        \\toprule",
        (
            "        & & \\multicolumn{2}{c}{AGE} & \\multicolumn{2}{c}{DE} "
            "& \\multicolumn{2}{c}{SHADE} \\\\"
        ),
        "        \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        (
            f"        Familia & Funcion & {args.label_base} & {args.label_variante} "
            f"& {args.label_base} & {args.label_variante} "
            f"& {args.label_base} & {args.label_variante} \\\\"
        ),
        "        \\midrule",
    ]

    for row in wide_rows:
        cells = [familia_latex(row["familia"]), row["funcion"]]
        for algoritmo in ALGORITMOS:
            label = NOMBRE_ALGORITMO[algoritmo]
            base = row[f"{label}_base"]
            variante = row[f"{label}_variante"]
            cells.append(formatear_mejor(base, variante, base))
            cells.append(formatear_mejor(base, variante, variante))
        lines.append("        " + " & ".join(cells) + " \\\\")

    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}%",
            "    }",
            "    \\vspace{0.3em}",
            (
                "    \\footnotesize{Nota: la comparacion se realiza con las "
                "funciones y semillas comunes entre ambas ejecuciones. En negrita "
                "se marca el menor valor dentro de cada par algoritmo--funcion.}"
            ),
            "\\end{table}",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def escribir_semillas(path, semillas):
    Path(path).write_text(" ".join(str(s) for s in semillas) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    funciones = parse_lista_enteros(args.funciones)
    semillas = parse_lista_enteros(args.semillas)

    base = leer_runs(args.root_base, args.metric)
    variante = leer_runs(args.root_variante, args.metric)
    funciones_finales = funciones_comunes(base, variante, funciones)
    if args.n_semillas_aleatorias is not None:
        candidatas = semillas if semillas else semillas_comunes_globales(
            base,
            variante,
            funciones_finales,
        )
        semillas = seleccionar_semillas_aleatorias(
            candidatas,
            args.n_semillas_aleatorias,
            args.random_state,
        )
    long_rows, wide_rows = construir_filas(base, variante, funciones_finales, semillas)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{args.output_prefix}.csv"
    wide_path = outdir / f"{args.output_prefix}_wide.csv"
    tex_path = outdir / f"{args.output_prefix}.tex"
    semillas_path = outdir / f"{args.output_prefix}_semillas.txt"

    escribir_csv(csv_path, long_rows)
    escribir_csv(wide_path, wide_rows)
    escribir_latex(tex_path, wide_rows, args)
    if semillas:
        escribir_semillas(semillas_path, semillas)

    print(csv_path)
    print(wide_path)
    print(tex_path)
    if semillas:
        print(semillas_path)
        print("semillas=" + " ".join(str(s) for s in semillas))


if __name__ == "__main__":
    main()
