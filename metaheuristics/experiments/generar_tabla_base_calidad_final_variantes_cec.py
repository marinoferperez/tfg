import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


CONFIG_DEFAULT = (
    ("virgen", "Sin reinicio", Path("results/cec/cec2017_d10_tam50")),
    ("reinicio_005", "Reinicio 5\\%", Path("results/cec/cec2017_d10_tam50_reinicio_005_deltaD")),
    ("reinicio_010", "Reinicio 10\\%", Path("results/cec/cec2017_d10_tam50_reinicio_010_deltaD")),
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

FUNCIONES_EVALUADAS = (1, 3, 5, 10, 17, 29)
FAMILIAS_PARCIALES = (
    ("Unimodales", (1, 3)),
    ("Multimodales", (5, 10)),
    ("Híbridas", (17,)),
    ("Compuestas", (29,)),
)
FAMILIA_POR_FUNCION = {
    funcid: familia
    for familia, funcids in FAMILIAS_PARCIALES
    for funcid in funcids
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera la tabla base de calidad final (mediana de cec_error) para las "
            "tres configuraciones (sin reinicio, reinicio 5%, reinicio 10%) de cada metaheurística."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="informes/benchmarks/generated/reinicio_primero_mejor",
        help="Directorio de salida para CSV y LaTeX.",
    )
    parser.add_argument(
        "--root-base",
        type=str,
        default=str(CONFIG_DEFAULT[0][2]),
        help="Directorio raíz de resultados sin reinicio.",
    )
    parser.add_argument(
        "--root-reinicio-5",
        type=str,
        default=str(CONFIG_DEFAULT[1][2]),
        help="Directorio raíz de resultados con reinicio 5%%.",
    )
    parser.add_argument(
        "--root-reinicio-10",
        type=str,
        default=str(CONFIG_DEFAULT[2][2]),
        help="Directorio raíz de resultados con reinicio 10%%.",
    )
    return parser.parse_args()


def escribir_csv(path: Path, filas):
    if not filas:
        raise ValueError(f"No hay filas para escribir en {path}")
    fieldnames = list(filas[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)


def cargar_cec_error_por_semilla(root: Path, *, estrategia_id: str):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio de resultados: {root}")

    filas = []
    funciones = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("f") and (path / "runs.csv").exists()
        ],
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
                if funcid not in FUNCIONES_EVALUADAS:
                    continue
                algoritmo = fila["algoritmo"].strip().lower()
                if algoritmo not in NOMBRE_ALGORITMO:
                    continue
                cec_error = str(fila.get("cec_error", "")).strip()
                if cec_error == "":
                    raise ValueError(f"Falta cec_error en {ruta_runs}")
                filas.append(
                    {
                        "algoritmo": algoritmo,
                        "estrategia": estrategia_id,
                        "cec_funcid": funcid,
                        "funcion": f"f{funcid}",
                        "familia": FAMILIA_POR_FUNCION[funcid],
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
        config_id = f"{fila['algoritmo']}__{fila['estrategia']}"
        funciones_por_config[config_id].add(int(fila["cec_funcid"]))
        semillas_por_config_func[(config_id, int(fila["cec_funcid"]))].add(int(fila["semilla"]))

    referencia_funciones = None
    for config_id, funcs in sorted(funciones_por_config.items()):
        funcs = sorted(funcs)
        if referencia_funciones is None:
            referencia_funciones = funcs
            continue
        if funcs != referencia_funciones:
            raise ValueError(
                f"Las funciones disponibles no coinciden entre configuraciones. "
                f"{config_id}: {funcs} vs referencia {referencia_funciones}"
            )
    if referencia_funciones != list(FUNCIONES_EVALUADAS):
        raise ValueError(
            f"Las funciones disponibles no coinciden con las esperadas: "
            f"{referencia_funciones} vs {list(FUNCIONES_EVALUADAS)}"
        )

    conteos_semillas = {k: len(v) for k, v in semillas_por_config_func.items()}
    if len(set(conteos_semillas.values())) != 1:
        raise ValueError(f"El numero de semillas no es uniforme entre configuraciones y funciones: {conteos_semillas}")


def mediana(valores):
    return float(statistics.median([float(v) for v in valores]))


def construir_tabla_medianas(filas):
    por_funcion_y_config = defaultdict(list)
    for fila in filas:
        cfg = f"{NOMBRE_ALGORITMO[fila['algoritmo']]}{SUFIJO_ESTRATEGIA_CORTO[fila['estrategia']]}"
        por_funcion_y_config[(int(fila["cec_funcid"]), cfg)].append(float(fila["cec_error"]))

    configuraciones = [
        f"{NOMBRE_ALGORITMO[alg]}{SUFIJO_ESTRATEGIA_CORTO[est]}"
        for alg in ORDEN_ALGORITMOS
        for est, _, _ in CONFIG_DEFAULT
    ]

    tabla = []
    for funcid in FUNCIONES_EVALUADAS:
        row = {
            "familia": FAMILIA_POR_FUNCION[funcid],
            "cec_funcid": int(funcid),
            "funcion": f"f{int(funcid)}",
        }
        for cfg in configuraciones:
            valores = por_funcion_y_config.get((int(funcid), cfg))
            if not valores:
                raise ValueError(f"Faltan datos para funcion={funcid}, configuracion={cfg}")
            row[cfg] = mediana(valores)
        tabla.append(row)
    return tabla, configuraciones


def formatear_numero(valor, *, decimales=3, cifras=3):
    """Usa decimal fijo si cabe en table-format=4.3 y científica solo para valores grandes."""
    val = float(valor)
    if abs(val) < 10000:
        return f"\\num{{{val:.{decimales}f}}}"
    return (
        "\\num[scientific-notation=true, round-mode=figures, "
        f"round-precision={cifras}]{{{val:.{max(cifras - 1, 0)}e}}}"
    )


def generar_tabla_latex(tabla, configuraciones):
    spec_num = "r"
    algoritmo_de_columna = {}
    for algoritmo in ORDEN_ALGORITMOS:
        nombre = NOMBRE_ALGORITMO[algoritmo]
        algoritmo_de_columna[nombre] = nombre
        algoritmo_de_columna[f"{nombre}-5"] = nombre
        algoritmo_de_columna[f"{nombre}-10"] = nombre

    colspec = " ".join([spec_num for _ in configuraciones])
    encabezados = " & ".join([f"\\multicolumn{{1}}{{c}}{{\\textbf{{{c}}}}}" for c in configuraciones])

    lineas = [
        "% AUTO-GENERATED: tabla base calidad final (variantes por reinicio)",
        "% Do not edit manually.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Mediana del error final respecto al \\'optimo conocido obtenida por cada metaheur\\'istica y umbral de reinicio en las funciones seleccionadas de CEC2017 (dimensi\\'on 10, poblaci\\'on 50). Se a\\~nade la metaheur\\'istica ganadora por fila para facilitar la lectura sin romper la alineaci\\'on decimal.}",
        "\\label{tab:cec_base_calidad_final_variantes}",
        "\\makebox[\\textwidth][c]{%",
        "\\adjustbox{width=1.08\\textwidth}{%",
        "\\begin{tabular}{c " + colspec + " l}",
        "\\toprule",
        "\\textbf{Funci\\'on} & " + encabezados + " & \\textbf{Ganadora(s)} \\\\",
        "\\midrule",
    ]
    filas_ordenadas = sorted(tabla, key=lambda row: int(row["cec_funcid"]))
    for idx, row in enumerate(filas_ordenadas):
        minimo = min(float(row[c]) for c in configuraciones)
        cols_ganadoras = [c for c in configuraciones if float(row[c]) == minimo]
        mhs_ganadoras = []
        for col in cols_ganadoras:
            mh = algoritmo_de_columna[col]
            if mh not in mhs_ganadoras:
                mhs_ganadoras.append(mh)
        ganadoras_txt = "/".join(mhs_ganadoras)
        lineas.append(
            " & ".join(
                [row["funcion"]] + [formatear_numero(row[c]) for c in configuraciones] + [ganadoras_txt]
            )
            + " \\\\"
        )
        if idx != len(filas_ordenadas) - 1:
            lineas.append("\\midrule")

    lineas.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}}",
            "\\vspace{2pt}",
            "\\begin{flushleft}\\small",
            "\\textbf{Nota.} Las estad\\'isticas se calcularon sobre $cec\\_error$, agregando las 51 semillas disponibles para cada funci\\'on seleccionada y configuraci\\'on. \\textbf{Ganadora(s)} indica la metaheur\\'istica con menor mediana en la fila (considerando sus tres variantes). Si hay empate exacto entre metaheur\\'isticas, se listan todas. Los valores se muestran con tres decimales cuando caben en el formato de tabla; la notaci\\'on cient\\'ifica se reserva para errores de mayor magnitud.",
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

    config = (
        ("virgen", "Sin reinicio", Path(args.root_base)),
        ("reinicio_005", "Reinicio 5\\%", Path(args.root_reinicio_5)),
        ("reinicio_010", "Reinicio 10\\%", Path(args.root_reinicio_10)),
    )

    filas = []
    for estrategia_id, _estrategia_tex, root in config:
        filas.extend(cargar_cec_error_por_semilla(root, estrategia_id=estrategia_id))

    validar_datos(filas)
    tabla, configuraciones = construir_tabla_medianas(filas)

    tabla_csv = outdir / "cec_base_calidad_final_variantes.csv"
    tabla_tex = outdir / "cec_base_calidad_final_variantes.tex"
    escribir_csv(tabla_csv, tabla)
    tabla_tex.write_text(generar_tabla_latex(tabla, configuraciones), encoding="utf-8")
    print(f"Tabla CSV: {tabla_csv}")
    print(f"Tabla LaTeX: {tabla_tex}")


if __name__ == "__main__":
    main()
