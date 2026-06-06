import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


CONFIG_DEFAULT = (
    ("virgen", Path("results/cec/cec2017_d10_tam50")),
    ("reinicio_005", Path("results/cec/cec2017_d10_tam50_reinicio_005")),
    ("reinicio_010", Path("results/cec/cec2017_d10_tam50_reinicio_010")),
)
ORDEN_ALGORITMOS = ("age", "de", "shade")
NOMBRE_ALGORITMO = {
    "age": "AGE",
    "de": "DE",
    "shade": "SHADE",
}
SUFIJO_ESTRATEGIA_CORTO = {
    "virgen": "",
    "reinicio_005": "-5",
    "reinicio_010": "-10",
}
FAMILIAS = (
    ("Unimodales", (1, 3)),
    ("Multimodales", (5, 10)),
    ("Híbridas", (17,)),
    ("Compuestas", (29,)),
)
FUNCION_A_FAMILIA = {
    funcid: familia
    for familia, funcids in FAMILIAS
    for funcid in funcids
}
ORDEN_FAMILIAS = {familia: idx for idx, (familia, _) in enumerate(FAMILIAS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Consolida resultados CEC2017 y genera la tabla de resultados por familias "
            "sobre el subconjunto de funciones disponible."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="informes/benchmarks/generated",
        help="Directorio de salida para CSV y LaTeX.",
    )
    return parser.parse_args()


def escribir_csv(path, filas):
    if not filas:
        raise ValueError(f"No hay filas para escribir en {path}")
    fieldnames = list(filas[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)


def cargar_runs_experimento(estrategia_id, root, orden_estrategia):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio de resultados: {root}")

    filas = []
    funciones = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("f") and (path / "runs.csv").exists()],
        key=lambda p: int(p.name[1:]),
    )
    if not funciones:
        raise RuntimeError(f"No se encontraron carpetas f*/runs.csv en {root}")

    for path_funcion in funciones:
        ruta_runs = path_funcion / "runs.csv"
        with ruta_runs.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for fila in reader:
                funcid = int(fila["cec_funcid"])
                if funcid not in FUNCION_A_FAMILIA:
                    continue

                cec_error = str(fila.get("cec_error", "")).strip()
                if cec_error == "":
                    raise ValueError(f"Falta cec_error en {ruta_runs}")

                algoritmo = fila["algoritmo"].strip().lower()
                filas.append(
                    {
                        "algoritmo": algoritmo,
                        "algoritmo_tex": NOMBRE_ALGORITMO[algoritmo],
                        "estrategia": estrategia_id,
                        "orden_estrategia": int(orden_estrategia),
                        "configuracion_id": f"{algoritmo}__{estrategia_id}",
                        "configuracion_corta": f"{NOMBRE_ALGORITMO[algoritmo]}{SUFIJO_ESTRATEGIA_CORTO[estrategia_id]}",
                        "familia": FUNCION_A_FAMILIA[funcid],
                        "orden_familia": ORDEN_FAMILIAS[FUNCION_A_FAMILIA[funcid]],
                        "cec_funcid": funcid,
                        "semilla": int(fila["semilla"]),
                        "cec_error": float(cec_error),
                    }
                )
    return filas


def validar_datos(filas):
    if not filas:
        raise RuntimeError("No hay filas para procesar.")

    vistos = set()
    for fila in filas:
        clave = (fila["algoritmo"], fila["estrategia"], fila["cec_funcid"], fila["semilla"])
        if clave in vistos:
            raise ValueError(f"Hay filas duplicadas por algoritmo + estrategia + funcion + semilla: {clave}")
        vistos.add(clave)

    funciones_por_config = defaultdict(set)
    semillas_por_config_func = defaultdict(set)
    for fila in filas:
        funciones_por_config[fila["configuracion_id"]].add(fila["cec_funcid"])
        semillas_por_config_func[(fila["configuracion_id"], fila["cec_funcid"])].add(fila["semilla"])

    referencia_funciones = None
    for configuracion_id, funciones in sorted(funciones_por_config.items()):
        funciones = sorted(funciones)
        if referencia_funciones is None:
            referencia_funciones = funciones
            continue
        if funciones != referencia_funciones:
            raise ValueError(
                f"Las funciones disponibles no coinciden entre configuraciones. "
                f"{configuracion_id}: {funciones} vs referencia {referencia_funciones}"
            )

    conteos_semillas = {clave: len(semillas) for clave, semillas in semillas_por_config_func.items()}
    if len(set(conteos_semillas.values())) != 1:
        raise ValueError(f"El numero de semillas no es uniforme entre configuraciones y funciones: {conteos_semillas}")

    familias_esperadas = {familia for familia, _ in FAMILIAS}
    for configuracion_id, funciones in funciones_por_config.items():
        familias_presentes = {FUNCION_A_FAMILIA[funcid] for funcid in funciones}
        if familias_presentes != familias_esperadas:
            raise ValueError(
                f"Las familias disponibles no coinciden para {configuracion_id}: "
                f"{sorted(familias_presentes)} vs esperadas {sorted(familias_esperadas)}"
            )


def media(valores):
    return statistics.fmean(valores)


def desviacion_poblacional(valores):
    if len(valores) <= 1:
        return 0.0
    return statistics.pstdev(valores)


def resumir_por_familia(filas):
    tabla = []
    agrupado = defaultdict(list)
    for fila in filas:
        clave = (
            fila["configuracion_id"],
            fila["configuracion_corta"],
            fila["algoritmo"],
            fila["familia"],
            fila["orden_estrategia"],
            fila["orden_familia"],
        )
        agrupado[clave].append(fila)

    for clave, subfilas in agrupado.items():
        valores = [float(f["cec_error"]) for f in subfilas]
        tabla.append(
            {
                "configuracion_id": clave[0],
                "configuracion": clave[1],
                "algoritmo": clave[2],
                "familia": clave[3],
                "orden_estrategia": clave[4],
                "orden_familia": clave[5],
                "n_funciones_familia": len({int(f["cec_funcid"]) for f in subfilas}),
                "n_runs": len(subfilas),
                "media": media(valores),
                "mediana": statistics.median(valores),
                "desviacion": desviacion_poblacional(valores),
                "mejor": min(valores),
                "peor": max(valores),
            }
        )
    return tabla


def formatear_numero(valor, *, decimales=3):
    return f"{float(valor):.{decimales}f}"


def generar_tabla_latex(tabla):
    lineas = [
        "% AUTO-GENERATED: tabla CEC por familias",
        "% Do not edit manually.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Resultados agregados por familias sobre CEC2017 (dimensi\\'on 10, poblaci\\'on 50), calculados sobre $cec\\_error$ y restringidos al subconjunto de funciones realmente evaluado.}",
        "\\label{tab:familias_preliminar}",
        "\\adjustbox{max width=\\textwidth}{%",
        "\\begin{tabular}{l l "
        "S[table-format=6.3] "
        "S[table-format=6.3] "
        "S[table-format=6.3] "
        "S[table-format=3.3] "
        "S[table-format=7.3]}",
        "\\toprule",
        "\\textbf{Familia} & \\textbf{Configuraci\\'on} & "
        "\\multicolumn{1}{c}{\\textbf{Media error}} & "
        "\\multicolumn{1}{c}{\\textbf{Mediana error}} & "
        "\\multicolumn{1}{c}{$\\boldsymbol{\\sigma}$} & "
        "\\multicolumn{1}{c}{\\textbf{Mejor error}} & "
        "\\multicolumn{1}{c}{\\textbf{Peor error}} \\\\",
        "\\midrule",
    ]

    filas_por_familia = defaultdict(list)
    for row in tabla:
        filas_por_familia[row["familia"]].append(row)

    for idx_fam, (familia, _) in enumerate(FAMILIAS):
        filas_fam = filas_por_familia.get(familia, [])
        if not filas_fam:
            continue
        filas_fam.sort(
            key=lambda row: (
                ORDEN_ALGORITMOS.index(row["algoritmo"]),
                row["orden_estrategia"],
            )
        )
        for idx_row, row in enumerate(filas_fam):
            familia_celda = f"\\multirow{{{len(filas_fam)}}}{{*}}{{{familia}}}" if idx_row == 0 else ""
            lineas.append(
                " & ".join(
                    [
                        familia_celda,
                        row["configuracion"],
                        formatear_numero(row["media"]),
                        formatear_numero(row["mediana"]),
                        formatear_numero(row["desviacion"]),
                        formatear_numero(row["mejor"]),
                        formatear_numero(row["peor"]),
                    ]
                )
                + " \\\\"
            )
        if idx_fam != len(FAMILIAS) - 1:
            lineas.append("\\midrule")

    lineas.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\vspace{2pt}",
            "\\begin{flushleft}\\small",
            "\\textbf{Nota.} La agregaci\\'on se realiz\\'o sobre $cec\\_error$. Las familias se construyeron solo con las funciones seleccionadas para este an\\'alisis: unimodales $\\{f1,f3\\}$, multimodales $\\{f5,f10\\}$, h\\'ibridas $\\{f17\\}$ y compuestas $\\{f29\\}$. Por tanto, esta tabla resume familias parciales y no la cobertura completa de CEC2017.",
            "\\end{flushleft}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lineas)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    filas = []
    for orden_estrategia, (estrategia_id, root) in enumerate(CONFIG_DEFAULT):
        filas.extend(cargar_runs_experimento(estrategia_id, root, orden_estrategia))

    validar_datos(filas)
    tabla = resumir_por_familia(filas)
    tabla.sort(
        key=lambda row: (
            ORDEN_ALGORITMOS.index(row["algoritmo"]),
            row["orden_estrategia"],
            row["orden_familia"],
        )
    )

    tabla_csv = outdir / "cec_familias.csv"
    tabla_tex = outdir / "cec_familias.tex"
    escribir_csv(tabla_csv, tabla)
    tabla_tex.write_text(generar_tabla_latex(tabla), encoding="utf-8")

    print(f"Tabla CSV: {tabla_csv}")
    print(f"Tabla LaTeX: {tabla_tex}")


if __name__ == "__main__":
    main()
