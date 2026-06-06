import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


CONFIG_DEFAULT = (
    ("virgen", "Sin reinicio", Path("results/cec/cec2017_d10_tam50")),
    ("reinicio_005", "Reinicio 5\\%", Path("results/cec/cec2017_d10_tam50_reinicio_005")),
    ("reinicio_010", "Reinicio 10\\%", Path("results/cec/cec2017_d10_tam50_reinicio_010")),
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Consolida resultados CEC2017 de estrategias virgen/reinicio y genera "
            "la tabla global por estrategia en CSV y LaTeX."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="informes/benchmarks/generated",
        help="Directorio de salida para CSV y LaTeX.",
    )
    return parser.parse_args()


def _sort_key_configuracion(row):
    return (
        ORDEN_ALGORITMOS.index(row["algoritmo"]),
        row["orden_estrategia"],
    )


def escribir_csv(path, filas):
    if not filas:
        raise ValueError(f"No hay filas para escribir en {path}")
    fieldnames = list(filas[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)


def cargar_runs_experimento(estrategia_id, estrategia_tex, root, orden_estrategia):
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
                if funcid not in FUNCIONES_EVALUADAS:
                    continue
                cec_error = str(fila.get("cec_error", "")).strip()
                if cec_error == "":
                    raise ValueError(f"Falta cec_error en {ruta_runs}")

                filas.append(
                    {
                        "algoritmo": fila["algoritmo"].strip().lower(),
                        "algoritmo_tex": NOMBRE_ALGORITMO[fila["algoritmo"].strip().lower()],
                        "estrategia": estrategia_id,
                        "estrategia_tex": estrategia_tex,
                        "orden_estrategia": int(orden_estrategia),
                        "configuracion_id": f"{fila['algoritmo'].strip().lower()}__{estrategia_id}",
                        "configuracion_tex": f"{NOMBRE_ALGORITMO[fila['algoritmo'].strip().lower()]} + {estrategia_tex}",
                        "cec_funcid": funcid,
                        "semilla": int(fila["semilla"]),
                        "cec_error": float(cec_error),
                        "fitness": float(fila["fitness"]),
                        "tiempo_s": float(fila["tiempo_s"]),
                        "ruta_runs": str(ruta_runs.resolve()),
                    }
                )
    return filas


def validar_datos(filas):
    if not filas:
        raise RuntimeError("No hay filas para procesar.")

    vistos = set()
    duplicados = []
    for fila in filas:
        clave = (fila["algoritmo"], fila["estrategia"], fila["cec_funcid"], fila["semilla"])
        if clave in vistos:
            duplicados.append(clave)
        vistos.add(clave)
    if duplicados:
        raise ValueError(f"Hay filas duplicadas por algoritmo + estrategia + funcion + semilla: {duplicados[:10]}")

    funciones_por_estrategia = defaultdict(set)
    for fila in filas:
        funciones_por_estrategia[fila["estrategia"]].add(int(fila["cec_funcid"]))
    funciones_referencia = None
    for estrategia, funciones_set in funciones_por_estrategia.items():
        funciones = sorted(funciones_set)
        if funciones_referencia is None:
            funciones_referencia = funciones
            continue
        if funciones != funciones_referencia:
            raise ValueError(
                f"Las funciones disponibles no coinciden entre estrategias. "
                f"{estrategia}: {funciones} vs referencia: {funciones_referencia}"
            )
    if funciones_referencia != list(FUNCIONES_EVALUADAS):
        raise ValueError(
            f"Las funciones disponibles no coinciden con las esperadas: {funciones_referencia} vs {list(FUNCIONES_EVALUADAS)}"
        )

    estrategias = sorted(funciones_por_estrategia)
    esperadas = set((alg, est, fid) for alg in ORDEN_ALGORITMOS for est in estrategias for fid in funciones_referencia)
    presentes = set((fila["algoritmo"], fila["estrategia"], int(fila["cec_funcid"])) for fila in filas)
    if esperadas != presentes:
        faltan = sorted(esperadas.difference(presentes))
        raise ValueError(f"Faltan combinaciones algoritmo + estrategia + funcion: {faltan}")

    semillas_por_combinacion = defaultdict(set)
    for fila in filas:
        clave = (fila["algoritmo"], fila["estrategia"], int(fila["cec_funcid"]))
        semillas_por_combinacion[clave].add(int(fila["semilla"]))
    conteos = {clave: len(semillas) for clave, semillas in semillas_por_combinacion.items()}
    if len(set(conteos.values())) != 1:
        raise ValueError(f"El numero de semillas no es uniforme entre combinaciones: {conteos}")

    semillas_por_algoritmo_funcion_estrategia = defaultdict(set)
    for fila in filas:
        clave = (fila["algoritmo"], int(fila["cec_funcid"]), fila["estrategia"])
        semillas_por_algoritmo_funcion_estrategia[clave].add(int(fila["semilla"]))

    semillas_referencia = None
    agrupado = defaultdict(dict)
    for (algoritmo, cec_funcid, estrategia), semillas in semillas_por_algoritmo_funcion_estrategia.items():
        agrupado[(algoritmo, cec_funcid)][estrategia] = sorted(semillas)
    for (algoritmo, cec_funcid), semillas_por_estrategia in agrupado.items():
        if semillas_referencia is None and semillas_por_estrategia:
            semillas_referencia = next(iter(semillas_por_estrategia.values()))
        for estrategia, semillas in semillas_por_estrategia.items():
            if semillas != semillas_referencia:
                raise ValueError(
                    f"Las semillas no coinciden para algoritmo={algoritmo}, funcion={cec_funcid}, estrategia={estrategia}: "
                    f"{semillas} vs referencia {semillas_referencia}"
                )


def media(valores):
    return statistics.fmean(valores)


def desviacion_poblacional(valores):
    if len(valores) <= 1:
        return 0.0
    return statistics.pstdev(valores)


def resumir_global(filas):
    filas_resumen = []
    agrupado = defaultdict(list)
    for fila in filas:
        clave = (
            fila["configuracion_id"],
            fila["algoritmo"],
            fila["algoritmo_tex"],
            fila["estrategia"],
            fila["estrategia_tex"],
            fila["orden_estrategia"],
        )
        agrupado[clave].append(fila)
    for clave, subfilas in agrupado.items():
        valores = [float(f["cec_error"]) for f in subfilas]
        filas_resumen.append(
            {
                "configuracion_id": clave[0],
                "algoritmo": clave[1],
                "algoritmo_tex": clave[2],
                "estrategia": clave[3],
                "estrategia_tex": clave[4],
                "configuracion_corta": f"{clave[2]}{SUFIJO_ESTRATEGIA_CORTO[clave[3]]}",
                "orden_estrategia": clave[5],
                "n_funciones": len({int(f["cec_funcid"]) for f in subfilas}),
                "n_runs": len(subfilas),
                "media_global": media(valores),
                "mediana_global": statistics.median(valores),
                "desviacion_global": desviacion_poblacional(valores),
                "mejor_global": min(valores),
                "peor_global": max(valores),
            }
        )
    return filas_resumen


def calcular_ranking_por_funcion(filas):
    filas_ranking = []
    por_funcion = defaultdict(list)
    for fila in filas:
        por_funcion[int(fila["cec_funcid"])].append(fila)

    for funcid in sorted(por_funcion):
        por_config = defaultdict(list)
        for fila in por_funcion[funcid]:
            clave = (
                fila["configuracion_id"],
                fila["algoritmo"],
                fila["algoritmo_tex"],
                fila["estrategia"],
                fila["estrategia_tex"],
                fila["orden_estrategia"],
            )
            por_config[clave].append(float(fila["cec_error"]))

        resumen = []
        for clave, valores in por_config.items():
            resumen.append(
                {
                    "cec_funcid": int(funcid),
                    "configuracion_id": clave[0],
                    "algoritmo": clave[1],
                    "algoritmo_tex": clave[2],
                    "estrategia": clave[3],
                    "estrategia_tex": clave[4],
                    "orden_estrategia": clave[5],
                    "mediana_funcion": statistics.median(valores),
                    "media_funcion": media(valores),
                    "mejor_funcion": min(valores),
                }
            )
        resumen.sort(
            key=lambda row: (
                row["mediana_funcion"],
                row["media_funcion"],
                row["mejor_funcion"],
                row["configuracion_id"],
            )
        )
        for idx, row in enumerate(resumen, start=1):
            row["rank_funcion"] = idx
            row["victoria_funcion"] = 1 if idx == 1 else 0
            filas_ranking.append(row)
    return filas_ranking


def combinar_resumen_y_ranking(filas_global, filas_ranking):
    ranking_global = defaultdict(lambda: {"suma_ranks": 0.0, "n": 0, "victorias": 0})
    for fila in filas_ranking:
        ref = ranking_global[fila["configuracion_id"]]
        ref["suma_ranks"] += float(fila["rank_funcion"])
        ref["n"] += 1
        ref["victorias"] += int(fila["victoria_funcion"])

    tabla = []
    for fila in filas_global:
        ref = ranking_global[fila["configuracion_id"]]
        nueva = dict(fila)
        nueva["ranking_medio"] = ref["suma_ranks"] / max(ref["n"], 1)
        nueva["victorias"] = ref["victorias"]
        tabla.append(nueva)
    return tabla


def formatear_numero_tex(valor, *, decimales=3):
    valor = float(valor)
    if not math.isfinite(valor):
        return "---"
    if valor == 0:
        return "0"
    abs_val = abs(valor)
    if abs_val >= 1e4 or abs_val < 1e-2:
        txt = f"{valor:.2e}"
        mantisa, exponente = txt.split("e")
        exp = int(exponente)
        return f"${mantisa} \\times 10^{{{exp}}}$"
    return f"{valor:.{decimales}f}"


def formatear_error_siunitx(valor, *, decimales=3):
    valor = float(valor)
    if not math.isfinite(valor):
        return "{}"
    return f"{valor:.{decimales}f}"


def formatear_mejor_siunitx(valor, *, decimales=3):
    return formatear_error_siunitx(valor, decimales=decimales)


def formatear_peor_siunitx(valor, *, decimales=3):
    return formatear_error_siunitx(valor, decimales=decimales)


def generar_tabla_latex(tabla):
    lineas = [
        "% AUTO-GENERATED: tabla global CEC por estrategia",
        "% Do not edit manually.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Resumen global de las configuraciones evaluadas sobre CEC2017 (dimensi\\'on 10, poblaci\\'on 50), agregando las seis funciones seleccionadas y usando $cec\\_error$ como m\\'etrica comparable entre funciones.}",
        "\\label{tab:cec_global_estrategias}",
        "\\adjustbox{max width=\\textwidth}{%",
        "\\begin{tabular}{l "
        "S[table-format=5.3] "
        "S[table-format=3.3] "
        "S[table-format=6.3] "
        "S[table-format=1.3] "
        "S[table-format=7.3]}",
        "\\toprule",
        "\\textbf{Configuraci\\'on} & "
        "\\multicolumn{1}{c}{\\textbf{Media error}} & "
        "\\multicolumn{1}{c}{\\textbf{Mediana error}} & "
        "\\multicolumn{1}{c}{$\\boldsymbol{\\sigma}$} & "
        "\\multicolumn{1}{c}{\\textbf{Mejor error}} & "
        "\\multicolumn{1}{c}{\\textbf{Peor error}} \\\\",
        "\\midrule",
    ]

    filas_por_algoritmo = defaultdict(list)
    for row in tabla:
        filas_por_algoritmo[row["algoritmo"]].append(row)

    for idx_alg, algoritmo in enumerate(ORDEN_ALGORITMOS):
        filas_alg = filas_por_algoritmo.get(algoritmo, [])
        if not filas_alg:
            continue
        for row in filas_alg:
            celdas = [
                row["configuracion_corta"],
                formatear_error_siunitx(row["media_global"]),
                formatear_error_siunitx(row["mediana_global"]),
                formatear_error_siunitx(row["desviacion_global"]),
                formatear_mejor_siunitx(row["mejor_global"]),
                formatear_peor_siunitx(row["peor_global"]),
            ]
            lineas.append(" & ".join(celdas) + " \\\\")
        if idx_alg != len(ORDEN_ALGORITMOS) - 1:
            lineas.append("\\midrule")

    lineas.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\vspace{2pt}",
            "\\begin{flushleft}\\small",
            "\\textbf{Nota.} Las estad\\'isticas de esta tabla se calcularon sobre $cec\\_error$, agregando las seis funciones seleccionadas $\\{f1,f3,f5,f10,f17,f29\\}$ y las 51 semillas de cada configuraci\\'on.",
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
    for orden_estrategia, (estrategia_id, estrategia_tex, root) in enumerate(CONFIG_DEFAULT):
        filas.extend(cargar_runs_experimento(estrategia_id, estrategia_tex, root, orden_estrategia))

    validar_datos(filas)
    tabla_global = resumir_global(filas)
    ranking_funcion = calcular_ranking_por_funcion(filas)
    tabla_final = combinar_resumen_y_ranking(tabla_global, ranking_funcion)

    tabla_final.sort(
        key=lambda row: (
            ORDEN_ALGORITMOS.index(row["algoritmo"]),
            row["orden_estrategia"],
        )
    )

    tabla_csv = outdir / "cec_global_estrategias.csv"
    ranking_csv = outdir / "cec_ranking_por_funcion.csv"
    tabla_tex = outdir / "cec_global_estrategias.tex"

    tabla_sin_ranking = []
    for row in tabla_final:
        fila = dict(row)
        fila.pop("ranking_medio", None)
        fila.pop("victorias", None)
        tabla_sin_ranking.append(fila)

    escribir_csv(tabla_csv, tabla_sin_ranking)
    escribir_csv(ranking_csv, ranking_funcion)
    tabla_tex.write_text(generar_tabla_latex(tabla_final), encoding="utf-8")

    print(f"Tabla CSV: {tabla_csv}")
    print(f"Ranking por funcion CSV: {ranking_csv}")
    print(f"Tabla LaTeX: {tabla_tex}")


if __name__ == "__main__":
    main()
