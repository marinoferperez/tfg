import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

"""
Genera tablas comparativas para experimentos de integración online.

El script compara varias carpetas de resultados pareadas y produce tablas de
ranking medio y error medio por función usando, por defecto, el error CEC.
"""

EXPERIMENTOS_DEFAULT = (
    ("p=0.00", Path("results/cec/online_final_d10_tam50_p00")),
    ("p=0.50", Path("results/cec/online_final_d10_tam50_p50")),
)
ALGORITMOS_DEFAULT = ("age", "de", "shade")
NOMBRE_ALGORITMO = {
    "age": "AGE",
    "de": "DE",
    "shade": "SHADE",
}

def parse_args():
    """
    Lee los argumentos de línea de comandos.

    Permite indicar los experimentos a comparar, los algoritmos incluidos, la
    métrica de comparación y el directorio donde se guardarán las tablas.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Genera tablas de comparacion online por ranking medio y error medio "
            "por funcion a partir de los runs.csv de varios experimentos."
        )
    )
    parser.add_argument(
        "--experimento",
        nargs=2,
        action="append",
        metavar=("ETIQUETA", "DIRECTORIO"),
        help=(
            "Experimento a comparar. Puede repetirse. Ejemplo: "
            "--experimento 'p=0.00' results/cec/online_final_d10_tam50_p00"
        ),
    )
    parser.add_argument(
        "--algoritmos",
        nargs="+",
        default=list(ALGORITMOS_DEFAULT),
        help="Algoritmos a incluir.",
    )
    parser.add_argument(
        "--metrica",
        default="cec_error",
        choices=("cec_error", "fitness"),
        help="Metrica usada para calcular medias y rankings. Menor es mejor.",
    )
    parser.add_argument(
        "--outdir",
        default="memoria/tablas/online",
        help="Directorio de salida para las tablas CSV y LaTeX.",
    )
    return parser.parse_args()


def escribir_csv(path, filas):
    """
    Escribe una lista de diccionarios en formato CSV.

    path: ruta del fichero CSV de salida.
    filas: lista de filas ya construidas, con las mismas claves.
    """
    if not filas:
        raise ValueError(f"No hay filas para escribir en {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)


def cargar_experimento(etiqueta, root, algoritmos, metrica):
    """
    Carga los runs.csv de un experimento online.

    etiqueta: nombre corto de la versión comparada, por ejemplo p=0.50.
    root: directorio raíz del experimento, con subcarpetas f*/runs.csv.
    algoritmos: algoritmos que se quieren incluir en la comparación.
    metrica: columna usada como valor de comparación, normalmente cec_error.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio de resultados: {root}")

    runs_paths = sorted(
        root.glob("f*/runs.csv"),
        key=lambda path: int(path.parent.name[1:]) if path.parent.name[1:].isdigit() else path.parent.name,
    )
    if not runs_paths:
        raise RuntimeError(f"No se encontraron f*/runs.csv en {root}")

    filas = []
    algoritmos = {alg.lower() for alg in algoritmos}
    columnas_requeridas = {"algoritmo", "cec_funcid", "semilla", metrica}

    for runs_path in runs_paths:
        with runs_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            faltantes = columnas_requeridas - set(reader.fieldnames or [])
            if faltantes:
                raise ValueError(f"Faltan columnas {sorted(faltantes)} en {runs_path}")

            for row in reader:
                algoritmo = row["algoritmo"].strip().lower()
                if algoritmo not in algoritmos:
                    continue

                valor = row[metrica].strip()
                if valor == "":
                    raise ValueError(f"Valor vacio para {metrica} en {runs_path}")

                filas.append(
                    {
                        "experimento": etiqueta,
                        "algoritmo": algoritmo,
                        "cec_funcid": int(row["cec_funcid"]),
                        "semilla": int(row["semilla"]),
                        "valor": float(valor),
                    }
                )

    if not filas:
        raise RuntimeError(f"No hay filas validas para {etiqueta} en {root}")
    return filas


def validar_comparabilidad(filas):
    """
    Comprueba que todos los experimentos sean pareables (mismas semillas, funciones...).

    filas: ejecuciones cargadas. Cada experimento debe tener las mismas claves
    algoritmo-función-semilla para que la comparación sea consistente.
    """
    claves_por_experimento = defaultdict(set)
    duplicados = set()
    vistos = set()

    for row in filas:
        clave = (row["experimento"], row["algoritmo"], row["cec_funcid"], row["semilla"])
        if clave in vistos:
            duplicados.add(clave)
        vistos.add(clave)
        claves_por_experimento[row["experimento"]].add((row["algoritmo"], row["cec_funcid"], row["semilla"]))

    if duplicados:
        ejemplo = sorted(duplicados)[0]
        raise ValueError(f"Hay ejecuciones duplicadas para la clave {ejemplo}")

    referencia_exp, referencia_claves = next(iter(claves_por_experimento.items()))
    for experimento, claves in claves_por_experimento.items():
        if claves != referencia_claves:
            faltan = sorted(referencia_claves - claves)[:5]
            sobran = sorted(claves - referencia_claves)[:5]
            raise ValueError(
                "Los experimentos no son pareables. "
                f"Referencia: {referencia_exp}; problema en {experimento}. "
                f"Faltan: {faltan}; sobran: {sobran}"
            )


def calcular_medias_por_funcion(filas):
    """
    Calcula el valor medio por algoritmo, función y experimento.

    filas: ejecuciones individuales ya cargadas desde los runs.csv.
    Devuelve un diccionario indexado por algoritmo, función y experimento.
    """
    valores = defaultdict(list)
    for row in filas:
        clave = (row["algoritmo"], row["cec_funcid"], row["experimento"])
        valores[clave].append(row["valor"])

    medias = {}
    for clave, xs in valores.items():
        medias[clave] = sum(xs) / len(xs)
    return medias


def rankear_valores(valores_por_experimento, tolerancia=1e-12):
    """
    Asigna rankings a varias versiones para una misma función.

    valores_por_experimento: diccionario experimento -> valor medio.
    tolerancia: margen para considerar empates numéricos exactos.
    Devuelve los rankings y el reparto de victorias de la función.
    """
    ordenados = sorted(valores_por_experimento.items(), key=lambda item: (item[1], item[0]))
    ranks = {}
    victorias = defaultdict(float)
    posicion = 1
    i = 0

    while i < len(ordenados):
        j = i + 1
        while j < len(ordenados) and math.isclose(
            ordenados[i][1],
            ordenados[j][1],
            rel_tol=tolerancia,
            abs_tol=tolerancia,
        ):
            j += 1

        rank = (posicion + posicion + (j - i) - 1) / 2.0
        for k in range(i, j):
            ranks[ordenados[k][0]] = rank

        if i == 0:
            peso = 1.0 / (j - i)
            for k in range(i, j):
                victorias[ordenados[k][0]] += peso

        posicion += j - i
        i = j

    return ranks, victorias


def calcular_rankings(medias, algoritmos, experimentos):
    """
    Calcula el ranking medio y las funciones ganadas por versión.

    medias: valores medios por algoritmo, función y experimento.
    algoritmos: lista de algoritmos incluidos en la tabla.
    experimentos: etiquetas de las versiones comparadas.
    """
    suma_ranks = defaultdict(float)
    victorias = defaultdict(float)
    n_funciones = defaultdict(int)

    funciones_por_algoritmo = defaultdict(set)
    for algoritmo, funcid, _ in medias:
        funciones_por_algoritmo[algoritmo].add(funcid)

    for algoritmo in algoritmos:
        for funcid in sorted(funciones_por_algoritmo[algoritmo]):
            valores = {
                experimento: medias[(algoritmo, funcid, experimento)]
                for experimento in experimentos
                if (algoritmo, funcid, experimento) in medias
            }
            if len(valores) != len(experimentos):
                raise ValueError(f"Faltan valores para {algoritmo}, f{funcid}: {valores.keys()}")

            ranks, victorias_funcion = rankear_valores(valores)
            for experimento in experimentos:
                suma_ranks[(algoritmo, experimento)] += ranks[experimento]
                victorias[(algoritmo, experimento)] += victorias_funcion[experimento]
                n_funciones[(algoritmo, experimento)] += 1

    filas = []
    for algoritmo in algoritmos:
        for experimento in experimentos:
            n = n_funciones[(algoritmo, experimento)]
            filas.append(
                {
                    "algoritmo": NOMBRE_ALGORITMO.get(algoritmo, algoritmo.upper()),
                    "experimento": experimento,
                    "ranking_medio": suma_ranks[(algoritmo, experimento)] / n,
                    "funciones_ganadas": victorias[(algoritmo, experimento)],
                    "n_funciones": n,
                }
            )
    return filas


def formatear_csv(valor):
    """
    Formatea valores numéricos para las tablas CSV.

    valor: dato que se quiere escribir. Los flotantes se redondean con una
    precisión compacta y el resto se devuelve sin cambios.
    """
    if isinstance(valor, float):
        return f"{valor:.12g}"
    return valor


def formatear_latex_numero(valor):
    """
    Formatea un número para LaTeX usando \\num{}.

    valor: número que se mostrará en la tabla. Se usa notación científica para
    valores muy pequeños o muy grandes.
    """
    valor = float(valor)
    if valor == 0.0:
        return r"\num{0}"
    if abs(valor) < 1e-3 or abs(valor) >= 1e4:
        return rf"\num{{{valor:.3e}}}"
    return rf"\num{{{valor:.3f}}}"


def formatear_latex_victorias(valor):
    """
    Formatea el recuento de funciones ganadas.

    valor: número de victorias. Puede ser decimal cuando hay empates repartidos
    entre varias versiones.
    """
    valor = float(valor)
    if valor.is_integer():
        return str(int(valor))
    return f"{valor:.1f}"


def tabla_latex_ranking(filas):
    """
    Construye la tabla LaTeX de ranking medio.

    filas: resultados agregados con algoritmo, experimento, ranking medio y
    número de funciones ganadas.
    """
    lineas = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Ranking medio de las variantes \textit{online} "
            r"comparadas por metaheurística.}"
        ),
        r"\label{tab:online_ranking_medio}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        (
            r"\textbf{Metaheurística} & \textbf{Versión} & "
            r"\textbf{Ranking medio} & \textbf{Funciones ganadas} \\"
        ),
        r"\midrule",
    ]

    for row in filas:
        lineas.append(
            " & ".join(
                [
                    row["algoritmo"],
                    row["experimento"],
                    f"{float(row['ranking_medio']):.3f}",
                    formatear_latex_victorias(row["funciones_ganadas"]),
                ]
            )
            + r" \\"
        )

    lineas.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{flushleft}\small",
            (
                r"\textbf{Nota.} El ranking se calcula por función y "
                r"metaheurística a partir del error medio final; un menor valor "
                r"indica mejor comportamiento. Los empates exactos en una función "
                r"se reparten entre las variantes empatadas."
            ),
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lineas)


def construir_tabla_funciones(algoritmo, medias, experimentos):
    """
    Construye la tabla de errores medios por función para un algoritmo.

    algoritmo: algoritmo para el que se genera la tabla.
    medias: valores medios por algoritmo, función y experimento.
    experimentos: etiquetas de las versiones que serán columnas de la tabla.
    """
    funciones = sorted({funcid for alg, funcid, _ in medias if alg == algoritmo})
    filas = []
    victorias = defaultdict(float)

    for funcid in funciones:
        valores = {experimento: medias[(algoritmo, funcid, experimento)] for experimento in experimentos}
        _, victorias_funcion = rankear_valores(valores)
        for experimento, victoria in victorias_funcion.items():
            victorias[experimento] += victoria

        row = {"funcion": f"f{funcid}"}
        for experimento in experimentos:
            row[experimento] = valores[experimento]
        filas.append(row)

    resumen = {"funcion": "Funciones ganadas"}
    for experimento in experimentos:
        resumen[experimento] = victorias[experimento]
    filas.append(resumen)
    return filas


def tabla_latex_funciones(algoritmo, filas, experimentos):
    """
    Construye la tabla LaTeX de errores medios por función.

    algoritmo: algoritmo asociado a la tabla.
    filas: filas generadas por construir_tabla_funciones.
    experimentos: etiquetas de las versiones comparadas.
    """
    algoritmo_tex = NOMBRE_ALGORITMO.get(algoritmo, algoritmo.upper())
    formato_columnas = "l" + "r" * len(experimentos)
    cabecera = " & ".join([r"\textbf{Función}"] + [rf"\textbf{{{exp}}}" for exp in experimentos])

    lineas = [
        r"\begin{table}[H]",
        r"\centering",
        (
            rf"\caption{{Error medio por función para {algoritmo_tex} en la "
            rf"comparación \textit{{online}}.}}"
        ),
        rf"\label{{tab:online_medias_funcion_{algoritmo}}}",
        r"\adjustbox{max width=\textwidth}{%",
        rf"\begin{{tabular}}{{{formato_columnas}}}",
        r"\toprule",
        cabecera + r" \\",
        r"\midrule",
    ]

    for row in filas[:-1]:
        valores = [float(row[exp]) for exp in experimentos]
        minimo = min(valores)
        celdas = [row["funcion"]]
        for exp, valor in zip(experimentos, valores):
            celda = formatear_latex_numero(valor)
            if math.isclose(valor, minimo, rel_tol=1e-12, abs_tol=1e-12):
                celda = rf"\textbf{{{celda}}}"
            celdas.append(celda)
        lineas.append(" & ".join(celdas) + r" \\")

    resumen = filas[-1]
    lineas.extend([r"\midrule"])
    lineas.append(
        " & ".join(
            [r"\textbf{Funciones ganadas}"]
            + [rf"\textbf{{{formatear_latex_victorias(resumen[exp])}}}" for exp in experimentos]
        )
        + r" \\"
    )

    lineas.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\vspace{2pt}",
            r"\begin{flushleft}\small",
            (
                r"\textbf{Nota.} En cada fila se destaca el menor error medio. "
                r"Los empates exactos se reparten en el recuento de funciones ganadas."
            ),
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lineas)


def normalizar_experimentos(args):
    """
    Obtiene la lista de experimentos a comparar.

    args: argumentos parseados. Si no se especifican experimentos por CLI, se
    usa la comparación final por defecto entre p=0.00 y p=0.50.
    """
    if args.experimento:
        return [(etiqueta, Path(root)) for etiqueta, root in args.experimento]
    return list(EXPERIMENTOS_DEFAULT)


def main():
    """
    Ejecuta el flujo completo de generación de tablas.

    Carga experimentos, valida que sean comparables, calcula medias/rankings y
    escribe las salidas CSV y LaTeX.
    """
    args = parse_args()
    experimentos = normalizar_experimentos(args)
    etiquetas = [etiqueta for etiqueta, _ in experimentos]
    algoritmos = [alg.lower() for alg in args.algoritmos]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    filas = []
    for etiqueta, root in experimentos:
        filas.extend(cargar_experimento(etiqueta, root, algoritmos, args.metrica))

    validar_comparabilidad(filas)
    medias = calcular_medias_por_funcion(filas)

    ranking = calcular_rankings(medias, algoritmos, etiquetas)
    ranking_csv = [
        {clave: formatear_csv(valor) for clave, valor in row.items()}
        for row in ranking
    ]
    escribir_csv(outdir / "online_ranking_medio.csv", ranking_csv)
    (outdir / "online_ranking_medio.tex").write_text(tabla_latex_ranking(ranking), encoding="utf-8")

    for algoritmo in algoritmos:
        tabla = construir_tabla_funciones(algoritmo, medias, etiquetas)
        tabla_csv = [
            {clave: formatear_csv(valor) for clave, valor in row.items()}
            for row in tabla
        ]
        escribir_csv(outdir / f"online_medias_funcion_{algoritmo}.csv", tabla_csv)
        (outdir / f"online_medias_funcion_{algoritmo}.tex").write_text(
            tabla_latex_funciones(algoritmo, tabla, etiquetas),
            encoding="utf-8",
        )

    print(f"Tablas generadas en: {outdir}")
    print(f"Experimentos: {', '.join(etiquetas)}")
    print(f"Metrica: {args.metrica}")


if __name__ == "__main__":
    main()
