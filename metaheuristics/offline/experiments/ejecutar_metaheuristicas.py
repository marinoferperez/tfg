import argparse
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

"""
Ejecuta metaheuristicas offline sobre CEC2017.

El script lanza AGE, DE y SHADE puros, guarda los resultados por funcion y
opcionalmente registra metricas detalladas para construir datasets offline.
"""

# Permite ejecutar desde la raiz del repositorio sin instalar el paquete.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocesado_de_datos.utils.path_utils import escribir_csv_dicts, escribir_json

ALGORITMOS_OFFLINE = ("age", "de", "shade")


def normalizar_algoritmo(valor):
    """
    Normaliza el identificador de algoritmo indicado por CLI.

    valor: cadena introducida por el usuario. Acepta 'todos' como alias de 'all'.
    """
    return "all" if str(valor).lower() == "todos" else str(valor).lower()


def validar_tam_poblacion(valor):
    """
    Valida el tamano de poblacion recibido por CLI.

    valor: tamano de poblacion solicitado. Debe ser un entero >= 4.
    """
    valor = int(valor)
    if valor < 4:
        raise argparse.ArgumentTypeError("debe ser un entero >= 4")
    return valor


def parse_args():
    """
    Lee los argumentos de linea de comandos.

    Permite seleccionar funciones CEC, semillas, algoritmos, reinicio elitista
    y opciones de registro de metricas/datasets.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta AGE, DE y SHADE offline sobre CEC2017 y guarda resultados "
            "por funcion, semilla y algoritmo."
        )
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help=(
            "Lista explicita de semillas a ejecutar. Acepta valores separados "
            "por espacios o comas. Si se indica, tiene prioridad sobre "
            "--seed-start y --n-seeds."
        ),
    )

    parser.add_argument(
        "--tam-poblacion",
        type=validar_tam_poblacion,
        default=50,
        help="Tamano de poblacion utilizado por las metaheuristicas. Por defecto 50.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Presupuesto maximo de evaluaciones. Por defecto: 10000 * dimension.",
    )
    parser.add_argument(
        "--reinicio",
        "--reinicio-elitista",
        dest="reinicio",
        action="store_true",
        help=(
            "Activa el reinicio elitista en CEC2017. El criterio actual "
            "preserva el mejor individuo y reinicia cuando el segundo mejor "
            "fitness permanece estancado durante la paciencia configurada."
        ),
    )
    parser.add_argument(
        "--reinicio-ratio",
        "--reinicio-elitista-ratio-paciencia",
        dest="reinicio_ratio",
        type=float,
        default=None,
        help=(
            "Fraccion de max_evals que debe transcurrir sin mejora real del "
            "segundo mejor fitness antes de permitir el reinicio. Debe "
            "indicarse cuando se activa --reinicio."
        ),
    )
    parser.add_argument(
        "--cec-funcid",
        nargs="+",
        required=True,
        help="Funciones CEC a ejecutar: lista (ej. --cec-funcid 1 2 3), CSV (ej. --cec-funcid 1,2,3) o 'all'.",
    )
    parser.add_argument("--cec-dim", type=int, choices=[2, 5, 10, 30, 50], default=10)
    parser.add_argument(
        "--algoritmo",
        type=normalizar_algoritmo,
        default="all",
        choices=[*ALGORITMOS_OFFLINE, "all"],
        help="Algoritmo a ejecutar",
    )
    parser.add_argument(
        "--sin-metricas",
        action="store_true",
        help="Si se indica, no registra métricas DEAP ni genera metricas_runs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Si se indica, muestra informacion de progreso por terminal.",
    )
    parser.add_argument(
        "--generar-dataset-completo",
        action="store_true",
        help=(
            "Si se indica, concatena las runs por funcion/algoritmo y guarda "
            "dataset_completo.h5 en benchmarking/offline/dataset_preprocesado."
        ),
    )
    parser.add_argument(
        "--solo-dataset-completo",
        action="store_true",
        help=(
            "Si se indica, no ejecuta benchmarks; solo genera dataset_completo "
            "a partir de metricas_runs ya existentes."
        ),
    )

    parser.add_argument("--outdir", type=str, default="results/resultados_benchmark_mhs")
    return parser.parse_args()


def normalizar_ratio_paciencia_reinicio(valor):
    """
    Valida la paciencia del reinicio elitista.

    valor: fraccion de max_evals sin mejora antes de permitir un reinicio.
    """
    if valor is None:
        return None
    valor = float(valor)
    if not np.isfinite(valor):
        raise ValueError("--reinicio-ratio debe ser finito.")
    if valor <= 0.0 or valor >= 1.0:
        raise ValueError(
            "--reinicio-ratio debe estar en el intervalo abierto (0, 1)."
        )
    return valor


def sufijo_ratio_paciencia_reinicio(valor):
    """
    Construye el sufijo usado en ficheros cuando el reinicio esta activo.

    valor: ratio de paciencia ya validado.
    """
    txt = f"{float(valor):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"_pat{txt}"


def gestiona_funcids_cec(args):
    """
    Resuelve las funciones CEC solicitadas.

    args: namespace de argparse. Acepta listas, valores CSV o 'all'.
    """
    tokens = []
    for parte in args.cec_funcid:
        for tk in str(parte).split(","):
            tk = tk.strip().lower()
            if tk:
                tokens.append(tk)

    if len(tokens) == 0:
        raise ValueError("--cec-funcid no puede estar vacio.")

    if "all" in tokens:
        if len(tokens) > 1:
            raise ValueError("Si usas --cec-funcid all, no combines con otros valores.")
        return list(range(1, 31))

    vistos = set()
    funcids = []
    for tk in tokens:
        try:
            fid = int(tk)
        except ValueError as exc:
            raise ValueError(f"Valor de --cec-funcid invalido: '{tk}'. Usa enteros en [1, 30] o 'all'.") from exc
        if not 1 <= fid <= 30:
            raise ValueError(f"funcid={fid} fuera de rango. Debe estar en [1, 30].")
        if fid not in vistos:
            vistos.add(fid)
            funcids.append(fid)
    return funcids


def gestiona_semillas(args):
    """
    Resuelve las semillas a ejecutar.

    args: namespace de argparse. --seeds tiene prioridad sobre seed-start/n-seeds.
    """
    if args.seeds is None:
        semillas = [int(args.seed_start) + i for i in range(int(args.n_seeds))]
        if len(semillas) == 0:
            raise ValueError("n-seeds debe ser >= 1")
        return semillas

    tokens = []
    for parte in args.seeds:
        for tk in str(parte).split(","):
            tk = tk.strip()
            if tk:
                tokens.append(tk)

    if not tokens:
        raise ValueError("--seeds no puede estar vacio.")

    semillas = []
    vistas = set()
    for tk in tokens:
        try:
            seed = int(tk)
        except ValueError as exc:
            raise ValueError(f"Valor de --seeds invalido: '{tk}'. Usa enteros positivos.") from exc
        if seed < 1:
            raise ValueError(f"Semilla invalida: {seed}. Debe ser >= 1.")
        if seed not in vistas:
            vistas.add(seed)
            semillas.append(seed)
    return semillas


def clave_fila_run(fila):
    """
    Construye la clave unica de una ejecucion.

    fila: fila de runs.csv con problema, algoritmo, adaptacion, funcion y semilla.
    """
    return (
        str(fila.get("problema", "")),
        str(fila.get("algoritmo", "")),
        str(fila.get("adaptacion", "")),
        str(fila.get("cec_funcid", "")),
        str(fila.get("qap_instancia", "")),
        int(fila.get("semilla", 0)),
    )


def leer_csv_dicts(path):
    """
    Lee un CSV como lista de diccionarios.

    path: ruta del CSV. Si no existe, devuelve una lista vacia.
    """
    import csv

    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fusionar_runs_existentes(path, filas_nuevas):
    """
    Fusiona runs existentes con runs nuevas evitando duplicados por clave.

    path: ruta del runs.csv existente.
    filas_nuevas: ejecuciones generadas en la llamada actual.
    """
    path = Path(path)
    existentes = leer_csv_dicts(path)
    if not existentes:
        return list(filas_nuevas)

    fusionadas = {clave_fila_run(fila): fila for fila in existentes}
    for fila in filas_nuevas:
        fusionadas[clave_fila_run(fila)] = fila
    return sorted(
        fusionadas.values(),
        key=lambda fila: (
            str(fila.get("problema", "")),
            str(fila.get("algoritmo", "")),
            int(float(fila.get("cec_funcid") or 0)),
            str(fila.get("qap_instancia", "")),
            int(float(fila.get("semilla") or 0)),
        ),
    )


def guardar_bloque_resultados(outdir, filas_runs, config_json, incluir_columnas_contexto=True):
    """
    Guarda runs, resumen CSV y resumen JSON de un bloque experimental.

    outdir: directorio del bloque, normalmente results/cec/<experimento>/fX.
    filas_runs: ejecuciones individuales del bloque.
    config_json: configuracion asociada al bloque.
    incluir_columnas_contexto: conserva columnas vacias de contexto si procede.
    """
    filas_runs = fusionar_runs_existentes(outdir / "runs.csv", filas_runs)
    filas_resumen = construir_resumen(
        filas_runs,
        incluir_columnas_contexto=incluir_columnas_contexto,
    )
    guardar_csv(outdir / "runs.csv", filas_runs)
    guardar_csv(outdir / "resumen_promedios.csv", filas_resumen)
    guardar_json(
        outdir / "resumen_promedios.json",
        {
            "config": config_json,
            "resumen": filas_resumen,
        },
    )
    return filas_resumen

def guardar_csv(path, filas):
    """
    Escribe un CSV manteniendo el orden de columnas de la primera fila.

    path: ruta de salida.
    filas: lista de diccionarios a escribir.
    """
    campos = list(filas[0].keys()) if filas else None
    escribir_csv_dicts(path, filas, fieldnames=campos)

def guardar_json(path, payload):
    """
    Escribe un JSON de resultados.

    path: ruta de salida.
    payload: diccionario serializable.
    """
    escribir_json(path, payload)

def mostrar(args, *valores, **kwargs):
    """
    Imprime mensajes solo si --verbose esta activo.

    args: namespace de argparse.
    valores: argumentos posicionales enviados a print.
    kwargs: argumentos nombrados enviados a print.
    """
    if args.verbose:
        print(*valores, **kwargs)

def resumen_grupo(filas):
    """
    Calcula medias de fitness y tiempo para un grupo de runs.

    filas: ejecuciones del mismo algoritmo/adaptacion/contexto.
    """
    fitness = np.asarray([float(f["fitness"]) for f in filas], dtype=float)
    tiempos = np.asarray([float(f["tiempo_s"]) for f in filas], dtype=float)

    return {
        "n_runs": int(len(filas)),
        "fitness_promedio": float(np.mean(fitness)),
        "tiempo_promedio_s": float(np.mean(tiempos)),
    }


def construir_kwargs_metaheuristica(args, seed, reinicio_activo, ratio_paciencia_reinicio):
    """
    Construye los parametros comunes de AGE, DE y SHADE.

    args: namespace de argparse.
    seed: semilla de la ejecucion.
    reinicio_activo: indica si se activa el reinicio elitista.
    ratio_paciencia_reinicio: paciencia del reinicio ya validada.
    """
    kwargs = {}
    if args.tam_poblacion is not None:
        kwargs["tam_poblacion"] = int(args.tam_poblacion)
    if args.max_evals is not None:
        kwargs["max_evals"] = int(args.max_evals)
    if reinicio_activo:
        kwargs["reinicio"] = True
        kwargs["reinicio_ratio"] = float(ratio_paciencia_reinicio)
    return kwargs


def crear_metaheuristica_offline(algoritmo, kwargs):
    """
    Instancia el adaptador CEC2017 correspondiente.

    algoritmo: identificador normalizado, uno de age, de o shade.
    kwargs: parametros de construccion de la metaheuristica.
    """
    if algoritmo == "age":
        from metaheuristics.offline.adapted.age_cec2017 import GeneticStationaryCEC2017

        return GeneticStationaryCEC2017(**kwargs), "age"
    if algoritmo == "de":
        from metaheuristics.offline.adapted.de_cec2017 import DifferentialEvolutionCEC2017

        return DifferentialEvolutionCEC2017(**kwargs), "de"
    if algoritmo == "shade":
        from metaheuristics.offline.adapted.shade_cec2017 import SHADECEC2017

        return SHADECEC2017(**kwargs), "shade"
    raise ValueError(f"Algoritmo offline no soportado: {algoritmo}")


def ejecutar_cec(args, semillas, outdir_metricas, algoritmos, funcid):
    """
    Ejecuta las metaheuristicas offline para una funcion CEC concreta.

    args: namespace de argparse.
    semillas: semillas a ejecutar.
    outdir_metricas: directorio base para metricas detalladas.
    algoritmos: algoritmos seleccionados.
    funcid: identificador de funcion CEC2017.
    """
    filas = []
    total_runs = len(algoritmos) * len(semillas)
    run_idx = 0
    registrar_metricas = not args.sin_metricas
    ratio_paciencia_reinicio = normalizar_ratio_paciencia_reinicio(args.reinicio_ratio)
    reinicio_activo = bool(args.reinicio)
    sufijo_reinicio = ""
    if reinicio_activo:
        sufijo_reinicio = "_reinicio" + sufijo_ratio_paciencia_reinicio(ratio_paciencia_reinicio)

    for algoritmo in algoritmos:
        for seed in semillas:
            run_idx += 1
            mostrar(
                args,
                f"[CEC2017 F{int(funcid)} {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed}...",
                flush=True,
            )

            kwargs = construir_kwargs_metaheuristica(
                args,
                seed=seed,
                reinicio_activo=reinicio_activo,
                ratio_paciencia_reinicio=ratio_paciencia_reinicio,
            )
            mtheuristica, algname = crear_metaheuristica_offline(algoritmo, kwargs)

            # Se mide el tiempo de pared de la ejecucion completa.
            t0 = time.perf_counter()
            resultado = mtheuristica.optimize(
                funcid = int(funcid),
                dim = args.cec_dim,
                seed = seed,
                algname = algname,
                registrar_metricas = registrar_metricas,
                ruta_metricas = str((outdir_metricas / "cec2017" / algoritmo).resolve()) if registrar_metricas else None,
                run_id = (
                    f"{algoritmo}_cec2017_f{int(funcid)}_d{int(args.cec_dim)}_s{int(seed)}"
                    f"{sufijo_reinicio}"
                ),
                cec_workdir = str(outdir_metricas.resolve().parent),
            )
            dt = time.perf_counter() - t0

            filas.append({
                "problema": "cec2017",
                "algoritmo": algoritmo,
                "adaptacion": f"{algoritmo}_cec2017",
                "cec_funcid": int(funcid),
                "qap_instancia": "",
                "semilla": int(seed),
                "fitness": float(resultado["mejor_fitness"]),
                "cec_error": float(resultado["mejor_error"]),
                "tiempo_s": float(dt),  
                "ruta_metricas": str(resultado.get("ruta_metricas", "")),
                "reinicio": bool(reinicio_activo),
                "reinicio_ratio": (
                    float(ratio_paciencia_reinicio) if reinicio_activo else ""
                ),
                "n_reinicios": int(len(resultado.get("reinicios", []))),
            })

            mostrar(
                args,
                f"[CEC2017 F{int(funcid)} {run_idx}/{total_runs}] {algoritmo.upper()} seed={seed} "
                f"fitness={float(resultado['mejor_fitness']):.6f} "
                f"error={float(resultado['mejor_error']):.6f} "
                f"tiempo={dt:.4f}s",
                flush=True,
            )
    return filas

def construir_resumen(filas_runs, desglosar_contexto=False, incluir_columnas_contexto=True):
    """
    Agrega las ejecuciones individuales en resumenes medios.

    filas_runs: filas individuales de runs.csv.
    desglosar_contexto: si True, separa por funcion/instancia.
    incluir_columnas_contexto: conserva columnas vacias cuando no se desglosa.
    """
    grupos = defaultdict(list)
    for fila in filas_runs:
        if desglosar_contexto:
            clave = (
                fila["problema"],
                fila["algoritmo"],
                fila["adaptacion"],
                fila.get("cec_funcid", ""),
                fila.get("qap_instancia", ""),
            )
        else:
            clave = (fila["problema"], fila["algoritmo"], fila["adaptacion"])
        grupos[clave].append(fila)

    filas_resumen = []
    for clave, filas in sorted(grupos.items()):
        if desglosar_contexto:
            problema, algoritmo, adaptacion, cec_funcid, qap_instancia = clave
        else:
            problema, algoritmo, adaptacion = clave
            cec_funcid, qap_instancia = "", ""

        resumen = resumen_grupo(filas)
        fila_resumen = {"algoritmo": algoritmo}
        if desglosar_contexto:
            fila_resumen["cec_funcid"] = cec_funcid
        elif incluir_columnas_contexto:
            fila_resumen["cec_funcid"] = cec_funcid
            fila_resumen["qap_instancia"] = qap_instancia
        fila_resumen.update(resumen)
        filas_resumen.append(fila_resumen)
    return filas_resumen


def main():
    """Punto de entrada del script offline."""
    args = parse_args()
    from preprocesado_de_datos.exportar_dataset_completo import ejecutar_exportacion_dataset_completo
    args.reinicio_ratio = normalizar_ratio_paciencia_reinicio(
        args.reinicio_ratio
    )
    args.reinicio = bool(args.reinicio)
    if args.max_evals is None:
        args.max_evals = 10000 * int(args.cec_dim)
    if args.reinicio and args.reinicio_ratio is None:
        raise ValueError("--reinicio requiere indicar --reinicio-ratio.")
    if not args.reinicio and args.reinicio_ratio is not None:
        raise ValueError("--reinicio-ratio solo puede utilizarse junto con --reinicio.")

    outdir_raiz = Path(args.outdir).resolve()
    nombre_experimento = outdir_raiz.name if outdir_raiz.name else "experimento"
    outdir_cec = (ROOT / "results" / "cec" / nombre_experimento).resolve()

    semillas = gestiona_semillas(args)

    if args.algoritmo in ("all", "todos"):
        algoritmos = ALGORITMOS_OFFLINE
    else:
        algoritmos = (args.algoritmo,)

    funcids_cec = gestiona_funcids_cec(args)
    funcids_txt = ",".join(str(int(f)) for f in funcids_cec)

    mostrar(args, "Configuracion:", flush=True)
    if args.seeds is None:
        mostrar(args, f"  semillas={semillas[0]}..{semillas[-1]} (n={len(semillas)})", flush=True)
    else:
        mostrar(args, f"  semillas={','.join(str(s) for s in semillas)} (n={len(semillas)})", flush=True)
    mostrar(
        args,
        "  tam_poblacion="
        + (
            str(int(args.tam_poblacion))
            if args.tam_poblacion is not None
            else "default_original_algoritmo"
        ),
        flush=True,
    )
    mostrar(
        args,
        "  max_evals="
        + (
            str(int(args.max_evals))
            if args.max_evals is not None
            else "default_original_algoritmo"
        ),
        flush=True,
    )
    mostrar(args, f"  cec: funcid={funcids_txt}, dim={args.cec_dim}", flush=True)
    mostrar(args, f"  algoritmo={args.algoritmo}", flush=True)
    mostrar(args, f"  registrar_metricas={not args.sin_metricas} (DEAP)", flush=True)
    mostrar(args, f"  reinicio={bool(args.reinicio)}", flush=True)
    mostrar(
        args,
        "  reinicio_ratio="
        + (
            str(args.reinicio_ratio)
            if args.reinicio
            else "no_aplica"
        ),
        flush=True,
    )
    mostrar(args, f"  generar_dataset_completo={bool(args.generar_dataset_completo)}", flush=True)
    mostrar(args, f"  solo_dataset_completo={bool(args.solo_dataset_completo)}", flush=True)
    mostrar(args, f"  outdir_solicitado={outdir_raiz}", flush=True)
    mostrar(args, f"  outdir_cec_fijo={outdir_cec}", flush=True)

    if args.generar_dataset_completo and args.sin_metricas and not args.solo_dataset_completo:
        raise ValueError(
            "--generar-dataset-completo requiere metricas_runs disponibles. "
            "No es compatible con --sin-metricas durante la misma ejecucion."
        )

    if args.solo_dataset_completo:
        resultados_exportacion = []
        for funcid in funcids_cec:
            resultados_exportacion.extend(
                ejecutar_exportacion_dataset_completo(
                    outdir_cec,
                    algoritmo=args.algoritmo,
                    funcion=f"f{int(funcid)}",
                )
            )

        if not resultados_exportacion:
            raise RuntimeError("No se generó ningun dataset_completo.")

        mostrar(args, "\nDatasets completos generados:", flush=True)
        for resultado in resultados_exportacion:
            bloque = resultado["funcion"]
            mostrar(
                args,
                f"  {resultado['algoritmo']}/{bloque}: "
                f"runs={resultado['n_runs_entrada']} muestras={resultado['n_muestras']}",
                flush=True,
            )
            mostrar(args, f"    {resultado['dataset_path']}", flush=True)
        return

    filas_runs_cec = []
    total_funcids = len(funcids_cec)
    for idx, funcid in enumerate(funcids_cec, start=1):
        mostrar(
            args,
            f"\nCEC2017: F{int(funcid)} ({idx}/{total_funcids})",
            flush=True,
        )
        outdir_funcid = outdir_cec / f"f{int(funcid)}"
        outdir_metricas_funcid = outdir_funcid / "metricas_runs"
        filas_funcid = ejecutar_cec(args, semillas, outdir_metricas_funcid, algoritmos, int(funcid))
        filas_runs_cec.extend(filas_funcid)

        config_funcid = {
            "seed_start": int(args.seed_start) if args.seeds is None else None,
            "n_seeds": int(args.n_seeds) if args.seeds is None else len(semillas),
            "seeds": [int(s) for s in semillas],
            "tam_poblacion": int(args.tam_poblacion) if args.tam_poblacion is not None else None,
            "max_evals": int(args.max_evals) if args.max_evals is not None else None,
            "funcid": int(funcid),
            "cec_dim": int(args.cec_dim),
            "algoritmo": str(args.algoritmo),
            "problema": "cec2017",
            "registrar_metricas": bool(not args.sin_metricas),
            "reinicio": bool(args.reinicio),
            "reinicio_ratio": (
                float(args.reinicio_ratio)
                if args.reinicio
                else None
            ),
        }
        guardar_bloque_resultados(
            outdir_funcid,
            filas_funcid,
            config_funcid,
            incluir_columnas_contexto=False,
        )

        if args.generar_dataset_completo:
            resultados_dataset = ejecutar_exportacion_dataset_completo(
                outdir_cec,
                algoritmo=args.algoritmo,
                funcion=f"f{int(funcid)}",
            )
            for resultado in resultados_dataset:
                mostrar(
                    args,
                    f"  dataset_completo {resultado['algoritmo']}/f{int(funcid)} "
                    f"muestras={resultado['n_muestras']} -> {resultado['dataset_path']}",
                    flush=True,
                )

    if len(filas_runs_cec) == 0:
        raise RuntimeError("No se ejecuto ninguna combinacion. Revisa --algoritmo.")

    if filas_runs_cec:
        config_cec = {
            "seed_start": int(args.seed_start) if args.seeds is None else None,
            "n_seeds": int(args.n_seeds) if args.seeds is None else len(semillas),
            "seeds": [int(s) for s in semillas],
            "tam_poblacion": int(args.tam_poblacion) if args.tam_poblacion is not None else None,
            "max_evals": int(args.max_evals) if args.max_evals is not None else None,
            "cec_funcid_raw": list(args.cec_funcid),
            "funcids_resueltas": [int(f) for f in funcids_cec],
            "cec_dim": int(args.cec_dim),
            "algoritmo": str(args.algoritmo),
            "problema": "cec2017",
            "registrar_metricas": bool(not args.sin_metricas),
            "reinicio": bool(args.reinicio),
            "reinicio_ratio": (
                float(args.reinicio_ratio)
                if args.reinicio
                else None
            ),
        }
        filas_resumen_contexto = construir_resumen(filas_runs_cec, desglosar_contexto=True)
        guardar_csv(outdir_cec / "resumen_funciones_promedio.csv", filas_resumen_contexto)
        guardar_json(
            outdir_cec / "resumen_funciones_promedio.json",
            {
                "config": config_cec,
                "resumen": filas_resumen_contexto,
            },
        )

        mostrar(args, "\nPromedios CEC por funcion y algoritmo/adaptacion:", flush=True)
        for fila in filas_resumen_contexto:
            funcid_txt = f"f{int(fila['cec_funcid'])}" if fila.get("cec_funcid") not in ("", None) else "f?"
            mostrar(
                args,
                f"  {funcid_txt} / {fila['algoritmo']}: "
                f"fitness_promedio={fila['fitness_promedio']:.6f}, "
                f"tiempo_promedio_s={fila['tiempo_promedio_s']:.6f}",
                flush=True,
            )
        mostrar(args, f"\nResultados CEC guardados en: {outdir_cec}", flush=True)


if __name__ == "__main__":
    main()
