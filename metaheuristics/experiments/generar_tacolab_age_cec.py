import argparse
import csv
import statistics
from bisect import bisect_right
from pathlib import Path

import xlwt


MILESTONES = (1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
FUNCIONES_EVALUADAS = (1, 3, 5, 10, 17, 29)
DIMENSION = 10
MAX_EVALS = 100000

CONFIGS_AGE = (
    ("age", "AGE", Path("results/cec/cec2017_d10_tam50")),
    ("age_5", "AGE-5", Path("results/cec/cec2017_d10_tam50_reinicio_005")),
    ("age_10", "AGE-10", Path("results/cec/cec2017_d10_tam50_reinicio_010")),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera ficheros XLS con formato Tacolab para las configuraciones AGE en CEC2017."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="informes/benchmarks/generated/tacolab",
        help="Directorio de salida para los XLS.",
    )
    return parser.parse_args()


def cec_error(funcid, fitness):
    return max(0.0, float(fitness) - float(funcid) * 100.0)


def cargar_trayectoria_error(path_csv, funcid):
    evaluaciones = []
    errores = []
    with Path(path_csv).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            evals = int(float(row["evaluaciones"]))
            fitness = float(row["min/mejor_hasta_ahora"])
            evaluaciones.append(evals)
            errores.append(cec_error(funcid, fitness))
    if not evaluaciones:
        raise ValueError(f"No hay filas en {path_csv}")
    return evaluaciones, errores


def error_en_milestone(evaluaciones, errores, milestone):
    objetivo = int(round(MAX_EVALS * float(milestone) / 100.0))
    idx = bisect_right(evaluaciones, objetivo) - 1
    if idx < 0:
        idx = 0
    return float(errores[idx])


def rutas_resultados_age(root, funcid):
    base = Path(root) / f"f{funcid}" / "metricas_runs" / "cec2017" / "age"
    patron = f"*/resultados_age_cec2017_f{funcid}_d10.csv"
    rutas = sorted(base.glob(patron))
    if not rutas:
        raise FileNotFoundError(f"No se encontraron resultados AGE para f{funcid} en {base}")
    return rutas


def resumen_configuracion(root):
    resumen = {milestone: {} for milestone in MILESTONES}
    for funcid in FUNCIONES_EVALUADAS:
        valores_por_milestone = {milestone: [] for milestone in MILESTONES}
        for ruta in rutas_resultados_age(root, funcid):
            evaluaciones, errores = cargar_trayectoria_error(ruta, funcid)
            for milestone in MILESTONES:
                valores_por_milestone[milestone].append(
                    error_en_milestone(evaluaciones, errores, milestone)
                )

        for milestone in MILESTONES:
            resumen[milestone][funcid] = statistics.median(valores_por_milestone[milestone])
    return resumen


def escribir_xls(path, alg_label, resumen):
    columnas = ["milestone"] + [f"F{i:02d}" for i in range(1, 31)] + ["dimension", "alg"]

    wb = xlwt.Workbook()
    ws = wb.add_sheet("Sheet1")

    header_style = xlwt.easyxf("font: bold on; align: horiz center")
    number_style = xlwt.easyxf("align: horiz right", num_format_str="0.000000")
    int_style = xlwt.easyxf("align: horiz right", num_format_str="0")
    text_style = xlwt.easyxf("align: horiz left")

    for col, name in enumerate(columnas):
        ws.write(0, col, name, header_style)

    for row_idx, milestone in enumerate(MILESTONES, start=1):
        ws.write(row_idx, 0, int(milestone), int_style)
        for funcid in range(1, 31):
            col_idx = funcid
            if funcid in resumen[milestone]:
                ws.write(row_idx, col_idx, float(resumen[milestone][funcid]), number_style)
            else:
                ws.write(row_idx, col_idx, "")
        ws.write(row_idx, 31, DIMENSION, int_style)
        ws.write(row_idx, 32, alg_label, text_style)

    ws.col(0).width = 3000
    for col in range(1, 31):
        ws.col(col).width = 3000
    ws.col(31).width = 3000
    ws.col(32).width = 4000

    wb.save(str(path))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for config_id, alg_label, root in CONFIGS_AGE:
        resumen = resumen_configuracion(root)
        path = outdir / f"tacolab_cec2017_d10_{config_id}.xls"
        escribir_xls(path, alg_label, resumen)
        print(f"XLS Tacolab generado: {path}")


if __name__ == "__main__":
    main()
