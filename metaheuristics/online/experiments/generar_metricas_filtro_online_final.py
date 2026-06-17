import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "memoria" / "tablas" / "online"

EXPERIMENTOS = {
    "age": {
        "nombre": "AGE",
        "base": ROOT / "results" / "cec" / "online_final_base_age",
        "rbf": ROOT / "results" / "cec" / "online_final_surrogate_age",
    },
    "de": {
        "nombre": "DE",
        "base": ROOT / "results" / "cec" / "online_final_base_de",
        "rbf": ROOT / "results" / "cec" / "online_final_surrogate_de",
    },
    "shade": {
        "nombre": "SHADE",
        "base": ROOT / "results" / "cec" / "online_final_base_shade",
        "rbf": ROOT / "results" / "cec" / "online_final_surrogate_shade",
    },
}


def _float(row, key):
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return 0.0
    return float(value)


def cargar_runs(root, algoritmo):
    rows = {}
    for path in sorted(root.glob("f*/runs.csv"), key=lambda p: int(p.parent.name[1:])):
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["algoritmo"].strip().lower() != algoritmo:
                    continue
                key = (int(row["cec_funcid"]), int(row["semilla"]))
                if key in rows:
                    raise ValueError(f"Ejecucion duplicada en {root}: {key}")
                rows[key] = row
    if not rows:
        raise RuntimeError(f"No se encontraron ejecuciones para {algoritmo} en {root}")
    return rows


def claves_comunes(base, rbf, algoritmo):
    keys = sorted(set(base) & set(rbf))
    por_funcion = defaultdict(int)
    for funcid, _seed in keys:
        por_funcion[funcid] += 1

    funciones = sorted(por_funcion)
    if funciones != list(range(1, 31)):
        raise ValueError(f"{algoritmo}: funciones incompletas tras filtrar semillas comunes: {funciones}")

    incompletas = {funcid: n for funcid, n in por_funcion.items() if n != 51}
    if incompletas:
        raise ValueError(f"{algoritmo}: se esperaban 51 semillas comunes por funcion: {incompletas}")

    return keys


def resumir(algoritmo_nombre, version, rows):
    n = len(rows)
    total_generados = sum(_float(row, "candidatos_generados") for row in rows)
    total_consultados = sum(_float(row, "candidatos_con_subrogado") for row in rows)
    total_rechazados = sum(_float(row, "candidatos_rechazados") for row in rows)
    tiempo_total = sum(_float(row, "tiempo_s") for row in rows) / n
    tiempo_modelo = sum(_float(row, "tiempo_online_total") for row in rows) / n
    entrenamientos = sum(_float(row, "entrenamientos_rbf") for row in rows) / n

    return {
        "Metaheuristica": algoritmo_nombre,
        "Version": version,
        "Consulta (%)": 100.0 * total_consultados / total_generados if total_generados else 0.0,
        "Rechazo (%)": 100.0 * total_rechazados / total_consultados if total_consultados else 0.0,
        "Eval. evitadas (%)": 100.0 * total_rechazados / total_generados if total_generados else 0.0,
        "Reentren.": entrenamientos,
        "Tiempo total (s)": tiempo_total,
        "Tiempo modelo (s)": tiempo_modelo,
        "n_runs": n,
    }


def escribir_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def escribir_latex(path, rows):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Métricas internas de la integración \textit{online}.}",
        r"\label{tab:online_metricas_filtro}",
        r"\adjustbox{max width=\textwidth}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Metaheurística} & \textbf{Versión} & "
            r"\textbf{Consulta (\%)} & \textbf{Rechazo (\%)} & "
            r"\textbf{Eval. evitadas (\%)} & "
            r"\textbf{Reentren.} & \textbf{Tiempo total (s)} & "
            r"\textbf{Tiempo modelo (s)} \\"
        ),
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["Metaheuristica"],
                    row["Version"],
                    f"{row['Consulta (%)']:.2f}",
                    f"{row['Rechazo (%)']:.2f}",
                    f"{row['Eval. evitadas (%)']:.2f}",
                    f"{row['Reentren.']:.2f}",
                    f"{row['Tiempo total (s)']:.2f}",
                    f"{row['Tiempo modelo (s)']:.2f}",
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\vspace{2pt}",
            r"\begin{flushleft}\small",
            (
                r"\textbf{Nota.} La consulta indica el porcentaje de candidatos "
                r"generados que son preevaluados por el subrogado. El rechazo se "
                r"calcula sobre los candidatos consultados. Las evaluaciones "
                r"evitadas corresponden al porcentaje de candidatos generados que "
                r"son rechazados por el filtro antes de evaluarse con la función "
                r"objetivo real. El tiempo del modelo agrupa entrenamiento y "
                r"predicción. Las medias se calculan sobre las 30 funciones y las "
                r"51 semillas comunes por función."
            ),
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = []
    for algoritmo, config in EXPERIMENTOS.items():
        base = cargar_runs(config["base"], algoritmo)
        rbf = cargar_runs(config["rbf"], algoritmo)
        keys = claves_comunes(base, rbf, algoritmo)
        base_rows = [base[key] for key in keys]
        rbf_rows = [rbf[key] for key in keys]

        rows.append(resumir(config["nombre"], "Base", base_rows))
        rows.append(resumir(config["nombre"], "RBF", rbf_rows))
        print(f"{config['nombre']}: {len(keys)} ejecuciones comunes ({len(keys) // 30} semillas por funcion)")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    escribir_csv(OUTDIR / "online_metricas_filtro.csv", rows)
    escribir_latex(OUTDIR / "online_metricas_filtro.tex", rows)
    print(f"Tabla escrita en {OUTDIR / 'online_metricas_filtro.tex'}")


if __name__ == "__main__":
    main()
