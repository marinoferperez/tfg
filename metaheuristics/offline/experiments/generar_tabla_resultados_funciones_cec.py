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
            "Consolida resultados CEC2017 y genera la tabla de resultados por función "
            "sobre las funciones realmente evaluadas."
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
                if funcid not in FUNCIONES_EVALUADAS:
                    continue

                fitness = str(fila.get("fitness", "")).strip()
                if fitness == "":
                    raise ValueError(f"Falta fitness en {ruta_runs}")
                cec_error = str(fila.get("cec_error", "")).strip()
                if cec_error == "":
                    raise ValueError(f"Falta cec_error en {ruta_runs}")

                algoritmo = fila["algoritmo"].strip().lower()
                filas.append(
                    {
                        "algoritmo": algoritmo,
                        "estrategia": estrategia_id,
                        "orden_estrategia": int(orden_estrategia),
                        "configuracion_id": f"{algoritmo}__{estrategia_id}",
                        "configuracion": f"{NOMBRE_ALGORITMO[algoritmo]}{SUFIJO_ESTRATEGIA_CORTO[estrategia_id]}",
                        "cec_funcid": funcid,
                        "funcion": f"f{funcid}",
                        "semilla": int(fila["semilla"]),
                        "fitness": float(fitness),
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

    if referencia_funciones != list(FUNCIONES_EVALUADAS):
        raise ValueError(
            f"Las funciones disponibles no coinciden con las esperadas: {referencia_funciones} vs {list(FUNCIONES_EVALUADAS)}"
        )

    conteos_semillas = {clave: len(semillas) for clave, semillas in semillas_por_config_func.items()}
    if len(set(conteos_semillas.values())) != 1:
        raise ValueError(f"El numero de semillas no es uniforme entre configuraciones y funciones: {conteos_semillas}")


def media(valores):
    return statistics.fmean(valores)


def desviacion_poblacional(valores):
    if len(valores) <= 1:
        return 0.0
    return statistics.pstdev(valores)


def resumir_por_funcion(filas):
    tabla = []
    agrupado = defaultdict(list)
    for fila in filas:
        clave = (
            fila["configuracion_id"],
            fila["configuracion"],
            fila["algoritmo"],
            fila["funcion"],
            fila["cec_funcid"],
            fila["orden_estrategia"],
        )
        agrupado[clave].append(fila)

    for clave, subfilas in agrupado.items():
        valores = [float(f["fitness"]) for f in subfilas]
        errores = [float(f["cec_error"]) for f in subfilas]
        tabla.append(
            {
                "configuracion_id": clave[0],
                "configuracion": clave[1],
                "algoritmo": clave[2],
                "funcion": clave[3],
                "cec_funcid": clave[4],
                "orden_estrategia": clave[5],
                "n_runs": len(subfilas),
                "media": media(valores),
                "mediana": statistics.median(valores),
                "desviacion": desviacion_poblacional(valores),
                "mejor": min(valores),
                "peor": max(valores),
                "error_media": media(errores),
                "error_mediana": statistics.median(errores),
                "error_desviacion": desviacion_poblacional(errores),
                "error_mejor": min(errores),
                "error_peor": max(errores),
            }
        )
    return tabla


def formatear_numero(valor, *, decimales=3):
    return f"{float(valor):.{decimales}f}"


def generar_tabla_latex_por_funcion(funcid, filas_funcion):
    funcion = f"f{funcid}"
    lineas = [
        f"% AUTO-GENERATED: tabla CEC para {funcion}",
        "% Do not edit manually.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Resultados agregados para {funcion} sobre CEC2017 (dimensi\\'on 10, poblaci\\'on 50), calculados sobre fitness bruto y agregando las 51 semillas de cada configuraci\\'on.}}",
        f"\\label{{tab:cec_{funcion}}}",
        "\\adjustbox{width=0.88\\textwidth}{%",
        "\\begin{tabular}{l "
        "S[table-format=7.3] "
        "S[table-format=7.3] "
        "S[table-format=7.3] "
        "S[table-format=7.3] "
        "S[table-format=7.3]}",
        "\\toprule",
        "\\textbf{Configuraci\\'on} & "
        "\\multicolumn{1}{c}{\\textbf{Media fitness}} & "
        "\\multicolumn{1}{c}{\\textbf{Mediana fitness}} & "
        "\\multicolumn{1}{c}{$\\boldsymbol{\\sigma}$} & "
        "\\multicolumn{1}{c}{\\textbf{Mejor fitness}} & "
        "\\multicolumn{1}{c}{\\textbf{Peor fitness}} \\\\",
        "\\midrule",
    ]

    filas_por_algoritmo = defaultdict(list)
    for row in filas_funcion:
        filas_por_algoritmo[row["algoritmo"]].append(row)

    for idx_alg, algoritmo in enumerate(ORDEN_ALGORITMOS):
        filas_alg = filas_por_algoritmo.get(algoritmo, [])
        if not filas_alg:
            continue
        filas_alg.sort(key=lambda row: row["orden_estrategia"])
        for row in filas_alg:
            lineas.append(
                " & ".join(
                    [
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
        if idx_alg != len(ORDEN_ALGORITMOS) - 1:
            lineas.append("\\midrule")

    familias_txt = ", ".join(
        f"{familia.lower()} $\\{{{','.join(f'f{fid}' for fid in funcids)}\\}}$"
        for familia, funcids in FAMILIAS_PARCIALES
    )
    lineas.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\vspace{2pt}",
            "\\begin{flushleft}\\small",
            "\\textbf{Nota.} Las estad\\'isticas se calcularon sobre el fitness bruto de esta funci\\'on, agregando las 51 semillas de cada configuraci\\'on.",
            "\\end{flushleft}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lineas)


def resumir_linea_base_sin_reinicio(tabla_funciones):
    filas_por_funcion = defaultdict(list)
    for row in tabla_funciones:
        if row["orden_estrategia"] == 0:
            filas_por_funcion[int(row["cec_funcid"])].append(row)

    tabla_calidad = []
    ranking_por_algoritmo = defaultdict(list)
    victorias_por_algoritmo = defaultdict(int)
    segundos_por_algoritmo = defaultdict(int)
    peores_por_algoritmo = defaultdict(int)

    for funcid in FUNCIONES_EVALUADAS:
        filas_funcion = sorted(
            filas_por_funcion[funcid],
            key=lambda row: (
                float(row["error_mediana"]),
                float(row["error_media"]),
                float(row["error_mejor"]),
                row["algoritmo"],
            ),
        )
        if len(filas_funcion) != len(ORDEN_ALGORITMOS):
            raise ValueError(f"Faltan algoritmos sin reinicio para f{funcid}: {filas_funcion}")

        rankings = {}
        posicion = 1
        while posicion <= len(filas_funcion):
            inicio = posicion - 1
            mediana_ref = float(filas_funcion[inicio]["error_mediana"])
            fin = inicio
            while fin + 1 < len(filas_funcion) and float(filas_funcion[fin + 1]["error_mediana"]) == mediana_ref:
                fin += 1

            rank_promedio = (posicion + (fin + 1)) / 2.0
            for idx in range(inicio, fin + 1):
                row = filas_funcion[idx]
                rankings[row["algoritmo"]] = rank_promedio
            posicion = fin + 2

        mejor_mediana = float(filas_funcion[0]["error_mediana"])
        peor_mediana = float(filas_funcion[-1]["error_mediana"])
        for row in filas_funcion:
            algoritmo = row["algoritmo"]
            ranking_por_algoritmo[algoritmo].append(rankings[algoritmo])
            if float(row["error_mediana"]) == mejor_mediana:
                victorias_por_algoritmo[algoritmo] += 1
            elif float(row["error_mediana"]) == peor_mediana:
                peores_por_algoritmo[algoritmo] += 1
            else:
                segundos_por_algoritmo[algoritmo] += 1

        por_algoritmo = {row["algoritmo"]: row for row in filas_funcion}
        mejores_algoritmos = [
            NOMBRE_ALGORITMO[row["algoritmo"]]
            for row in filas_funcion
            if float(row["error_mediana"]) == mejor_mediana
        ]
        tabla_calidad.append(
            {
                "cec_funcid": funcid,
                "funcion": f"f{funcid}",
                "familia": FAMILIA_POR_FUNCION[funcid],
                "age_error_mediana": por_algoritmo["age"]["error_mediana"],
                "de_error_mediana": por_algoritmo["de"]["error_mediana"],
                "shade_error_mediana": por_algoritmo["shade"]["error_mediana"],
                "mejor_mh": "/".join(mejores_algoritmos),
                "rank_age": rankings["age"],
                "rank_de": rankings["de"],
                "rank_shade": rankings["shade"],
            }
        )

    tabla_ranking = []
    for algoritmo in ORDEN_ALGORITMOS:
        ranks = ranking_por_algoritmo[algoritmo]
        tabla_ranking.append(
            {
                "algoritmo": algoritmo,
                "metaheuristica": NOMBRE_ALGORITMO[algoritmo],
                "ranking_medio": media(ranks),
                "ranking_mediano": statistics.median(ranks),
                "veces_mejor": victorias_por_algoritmo[algoritmo],
                "veces_segunda": segundos_por_algoritmo[algoritmo],
                "veces_peor": peores_por_algoritmo[algoritmo],
            }
        )
    tabla_ranking.sort(
        key=lambda row: (
            float(row["ranking_medio"]),
            float(row["ranking_mediano"]),
            -int(row["veces_mejor"]),
            ORDEN_ALGORITMOS.index(row["algoritmo"]),
        )
    )
    posicion = 1
    while posicion <= len(tabla_ranking):
        inicio = posicion - 1
        ref = (
            float(tabla_ranking[inicio]["ranking_medio"]),
            float(tabla_ranking[inicio]["ranking_mediano"]),
        )
        fin = inicio
        while fin + 1 < len(tabla_ranking) and (
            float(tabla_ranking[fin + 1]["ranking_medio"]),
            float(tabla_ranking[fin + 1]["ranking_mediano"]),
        ) == ref:
            fin += 1
        for idx in range(inicio, fin + 1):
            tabla_ranking[idx]["posicion"] = posicion
        posicion = fin + 2

    return tabla_calidad, tabla_ranking


def generar_tabla_latex_linea_base_calidad(tabla):
    spec_num = "S[table-format=4.3]"

    def celda_error(row, algoritmo):
        valor = float(row[f"{algoritmo}_error_mediana"])
        celda = formatear_numero(valor)
        minimo = min(
            float(row["age_error_mediana"]),
            float(row["de_error_mediana"]),
            float(row["shade_error_mediana"]),
        )
        if valor == minimo:
            return f"{{\\bfseries {celda}}}"
        return celda

    lineas = [
        "% AUTO-GENERATED: tabla calidad inicial sin reinicio",
        "% Do not edit manually.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Mediana del error final de AGE, DE y SHADE sin reinicio en las funciones seleccionadas de CEC2017, calculada sobre las 51 semillas.}",
        "\\label{tab:cec_base_calidad_final}",
        "\\adjustbox{width=0.5\\textwidth}{%",
        "\\begin{tabular}{l l "
        f"{spec_num} "
        f"{spec_num} "
        f"{spec_num}}}",
        "\\toprule",
        "\\textbf{Familia} & "
        "\\textbf{Funci\\'on} & "
        "\\multicolumn{1}{c}{\\textbf{AGE}} & "
        "\\multicolumn{1}{c}{\\textbf{DE}} & "
        "\\multicolumn{1}{c}{\\textbf{SHADE}} \\\\",
        "\\midrule",
    ]

    filas_por_familia = defaultdict(list)
    for row in tabla:
        filas_por_familia[row["familia"]].append(row)

    for idx_familia, (familia, funcids) in enumerate(FAMILIAS_PARCIALES):
        filas_familia = sorted(
            filas_por_familia.get(familia, []),
            key=lambda row: int(row["cec_funcid"]),
        )
        for idx_row, row in enumerate(filas_familia):
            if len(filas_familia) == 1:
                familia_txt = familia
            else:
                familia_txt = f"\\multirow{{{len(filas_familia)}}}{{*}}{{{familia}}}" if idx_row == 0 else ""
            lineas.append(
                " & ".join(
                    [
                        familia_txt,
                        row["funcion"],
                        celda_error(row, "age"),
                        celda_error(row, "de"),
                        celda_error(row, "shade"),
                    ]
                )
                + " \\\\"
        )
        if idx_familia != len(FAMILIAS_PARCIALES) - 1:
            lineas.append("\\midrule")

    lineas.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lineas)


def generar_tabla_latex_linea_base_ranking(tabla):
    lineas = [
        "% AUTO-GENERATED: tabla ranking inicial sin reinicio",
        "% Do not edit manually.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Ranking agregado inicial de AGE, DE y SHADE sin reinicio, calculado por funci\\'on a partir de la mediana de $cec\\_error$.}",
        "\\label{tab:cec_base_ranking}",
        "\\begin{tabular}{S[table-format=1.0] l S[table-format=1.3]}",
        "\\toprule",
        "\\multicolumn{1}{c}{\\textbf{Pos.}} & "
        "\\textbf{MH} & "
        "\\multicolumn{1}{c}{\\textbf{Ranking medio}} \\\\",
        "\\midrule",
    ]
    for row in tabla:
        lineas.append(
            " & ".join(
                [
                    str(int(row["posicion"])),
                    row["metaheuristica"],
                    f"{float(row['ranking_medio']):.3f}",
                ]
            )
            + " \\\\"
        )

    lineas.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\begin{flushleft}\\small",
            "\\textbf{Nota.} Para cada funci\\'on se ordenan AGE, DE y SHADE seg\\'un la mediana de $cec\\_error$ sin reinicio. En caso de empate exacto en la mediana se asigna el rango promedio a las metaheur\\'isticas empatadas. El ranking medio promedia esos puestos sobre $\\{f1,f3,f5,f10,f17,f29\\}$. Los datos proceden de \\texttt{results/cec/cec2017\\_d10\\_tam50/f*/runs.csv}.",
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
    tabla = resumir_por_funcion(filas)
    tabla.sort(
        key=lambda row: (
            row["cec_funcid"],
            ORDEN_ALGORITMOS.index(row["algoritmo"]),
            row["orden_estrategia"],
        )
    )

    tabla_csv = outdir / "cec_funciones.csv"
    escribir_csv(tabla_csv, tabla)
    print(f"Tabla CSV consolidada: {tabla_csv}")

    tabla_base_calidad, tabla_base_ranking = resumir_linea_base_sin_reinicio(tabla)
    tabla_base_calidad_csv = outdir / "cec_base_calidad_final.csv"
    tabla_base_ranking_csv = outdir / "cec_base_ranking.csv"
    tabla_base_calidad_tex = outdir / "cec_base_calidad_final.tex"
    tabla_base_ranking_tex = outdir / "cec_base_ranking.tex"
    escribir_csv(tabla_base_calidad_csv, tabla_base_calidad)
    escribir_csv(tabla_base_ranking_csv, tabla_base_ranking)
    tabla_base_calidad_tex.write_text(generar_tabla_latex_linea_base_calidad(tabla_base_calidad), encoding="utf-8")
    tabla_base_ranking_tex.write_text(generar_tabla_latex_linea_base_ranking(tabla_base_ranking), encoding="utf-8")
    print(f"Tabla CSV linea base calidad: {tabla_base_calidad_csv}")
    print(f"Tabla LaTeX linea base calidad: {tabla_base_calidad_tex}")
    print(f"Tabla CSV linea base ranking: {tabla_base_ranking_csv}")
    print(f"Tabla LaTeX linea base ranking: {tabla_base_ranking_tex}")

    filas_por_funcion = defaultdict(list)
    for row in tabla:
        filas_por_funcion[row["cec_funcid"]].append(row)

    for funcid in FUNCIONES_EVALUADAS:
        filas_funcion = filas_por_funcion.get(funcid, [])
        if not filas_funcion:
            continue
        tabla_tex = outdir / f"cec_funcion_f{funcid}.tex"
        tabla_tex.write_text(generar_tabla_latex_por_funcion(funcid, filas_funcion), encoding="utf-8")
        print(f"Tabla LaTeX: {tabla_tex}")


if __name__ == "__main__":
    main()
