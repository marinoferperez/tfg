import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from generar_tabla_resultados_globales_cec import (
    CONFIG_DEFAULT,
    ORDEN_ALGORITMOS,
    SUFIJO_ESTRATEGIA_CORTO,
    cargar_runs_experimento,
    formatear_error_siunitx,
    validar_datos,
    escribir_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera la tabla de ranking global de configuraciones CEC2017 "
            "a partir de la mediana del fitness final por funcion."
        )
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="informes/benchmarks/generated",
        help="Directorio de salida para CSV y LaTeX.",
    )
    return parser.parse_args()


def generar_tabla_latex(tabla):
    lineas = [
        "% AUTO-GENERATED: tabla ranking global CEC",
        "% Do not edit manually.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Ranking global de las configuraciones evaluadas sobre CEC2017 (dimensi\\'on 10, poblaci\\'on 50), ordenado por ranking medio ascendente.}",
        "\\label{tab:cec_ranking_global}",
        "\\adjustbox{max width=\\textwidth}{%",
        "\\begin{tabular}{S[table-format=1.0] l S[table-format=1.3] S[table-format=1.0]}",
        "\\toprule",
        "\\multicolumn{1}{c}{\\textbf{Pos. final}} & "
        "\\textbf{Configuraci\\'on} & "
        "\\multicolumn{1}{c}{\\textbf{Ranking medio}} & "
        "\\multicolumn{1}{c}{\\textbf{Victorias}} \\\\",
        "\\midrule",
    ]

    for row in tabla:
        lineas.append(
            " & ".join(
                [
                    str(int(row["pos_final"])),
                    row["configuracion_corta"],
                    f"{float(row['ranking_medio']):.3f}",
                    str(int(row["victorias"])),
                ]
            )
            + " \\\\"
        )

    lineas.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\vspace{2pt}",
            "\\begin{flushleft}\\small",
            "\\textbf{Nota.} El ranking por funci\\'on se calcul\\'o ordenando las nueve configuraciones seg\\'un la mediana del fitness final en cada funci\\'on; en caso de empate, se desempata por media de fitness, despu\\'es por el mejor fitness y, si persiste el empate exacto, por el identificador de configuraci\\'on para mantener un orden total reproducible. El ranking medio es el promedio de esos puestos sobre las funciones seleccionadas $\\{f1,f3,f5,f10,f17,f29\\}$. Las victorias cuentan el n\\'umero de funciones en las que cada configuraci\\'on obtuvo el primer puesto tras aplicar ese criterio.",
            "\\end{flushleft}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lineas)


def generar_tabla_latex_por_funcion(funcid, filas_funcion):
    filas_funcion = sorted(filas_funcion, key=lambda r: r["rank_funcion"])
    lineas = [
        f"% AUTO-GENERATED: tabla ranking CEC f{funcid}",
        "% Do not edit manually.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Ranking de configuraciones para $f_{{{funcid}}}$ (CEC2017, dimensi\\'on 10, poblaci\\'on 50).}}",
        f"\\label{{tab:cec_ranking_f{funcid}}}",
        "\\begin{tabular}{S[table-format=1.0] l S[table-format=6.3]}",
        "\\toprule",
        "\\multicolumn{1}{c}{\\textbf{Pos.}} & "
        "\\textbf{Configuraci\\'on} & "
        "\\multicolumn{1}{c}{\\textbf{Mediana fitness}} \\\\",
        "\\midrule",
    ]

    for row in filas_funcion:
        config_corta = f"{row['algoritmo_tex']}{SUFIJO_ESTRATEGIA_CORTO[row['estrategia']]}"
        lineas.append(
            " & ".join([
                str(int(row["rank_funcion"])),
                config_corta,
                formatear_error_siunitx(row["mediana_funcion"]),
            ])
            + " \\\\"
        )

    lineas.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2pt}",
        "\\begin{flushleft}\\small",
        "\\textbf{Nota.} Ordenaci\\'on por mediana del fitness final; empate por media, mejor valor e identificador de configuraci\\'on.",
        "\\end{flushleft}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lineas)


def calcular_ranking_por_funcion_fitness(filas):
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
            por_config[clave].append(float(fila["fitness"]))

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
                    "media_funcion": statistics.fmean(valores),
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


def construir_ranking_global(filas, filas_ranking):
    metadatos = {}
    for fila in filas:
        metadatos[fila["configuracion_id"]] = fila

    acumulado = defaultdict(lambda: {"suma_ranks": 0.0, "n": 0, "victorias": 0})
    for fila in filas_ranking:
        ref = acumulado[fila["configuracion_id"]]
        ref["suma_ranks"] += float(fila["rank_funcion"])
        ref["n"] += 1
        ref["victorias"] += int(fila["victoria_funcion"])

    tabla = []
    for configuracion_id, ref in acumulado.items():
        meta = metadatos[configuracion_id]
        tabla.append(
            {
                "configuracion_id": configuracion_id,
                "configuracion_corta": f"{meta['algoritmo_tex']}{SUFIJO_ESTRATEGIA_CORTO[meta['estrategia']]}",
                "algoritmo": meta["algoritmo"],
                "orden_estrategia": int(meta["orden_estrategia"]),
                "ranking_medio": ref["suma_ranks"] / max(ref["n"], 1),
                "victorias": ref["victorias"],
            }
        )
    return tabla


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    filas = []
    for orden_estrategia, (estrategia_id, estrategia_tex, root) in enumerate(CONFIG_DEFAULT):
        filas.extend(cargar_runs_experimento(estrategia_id, estrategia_tex, root, orden_estrategia))

    validar_datos(filas)
    ranking_funcion = calcular_ranking_por_funcion_fitness(filas)
    tabla_final = construir_ranking_global(filas, ranking_funcion)

    tabla_final.sort(
        key=lambda row: (
            float(row["ranking_medio"]),
            -int(row["victorias"]),
            ORDEN_ALGORITMOS.index(row["algoritmo"]),
            row["orden_estrategia"],
        )
    )

    tabla_ranking = []
    for pos_final, row in enumerate(tabla_final, start=1):
        tabla_ranking.append(
            {
                "pos_final": pos_final,
                "configuracion_id": row["configuracion_id"],
                "configuracion_corta": row["configuracion_corta"],
                "ranking_medio": row["ranking_medio"],
                "victorias": row["victorias"],
            }
        )

    tabla_csv = outdir / "cec_ranking_global.csv"
    tabla_tex = outdir / "cec_ranking_global.tex"
    escribir_csv(tabla_csv, tabla_ranking)
    tabla_tex.write_text(generar_tabla_latex(tabla_ranking), encoding="utf-8")
    print(f"Tabla CSV: {tabla_csv}")
    print(f"Tabla LaTeX: {tabla_tex}")

    por_funcion = defaultdict(list)
    for fila in ranking_funcion:
        por_funcion[int(fila["cec_funcid"])].append(fila)

    for funcid, filas_f in sorted(por_funcion.items()):
        tex = outdir / f"cec_ranking_f{funcid}.tex"
        tex.write_text(generar_tabla_latex_por_funcion(funcid, filas_f), encoding="utf-8")
        print(f"Tabla LaTeX f{funcid}: {tex}")


if __name__ == "__main__":
    main()
