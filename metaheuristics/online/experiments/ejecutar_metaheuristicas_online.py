import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaheuristics.experiments.ejecutar_metaheuristicas import (
    VENTANA_DIAGNOSTICA_REINICIO,
    construir_resumen,
    gestiona_funcids_cec,
    gestiona_semillas,
    guardar_bloque_resultados,
    guardar_csv,
    guardar_json,
    normalizar_algoritmo,
    normalizar_ratio_paciencia_reinicio,
    sufijo_ratio_paciencia_reinicio,
    validar_tam_poblacion,
)
from metaheuristics.online.adapted.age_cec2017_online import GeneticStationaryCEC2017Online
from metaheuristics.online.adapted.de_cec2017_online import DifferentialEvolutionCEC2017Online
from metaheuristics.online.adapted.shade_cec2017_online import SHADECEC2017Online
from metaheuristics.online.surrogate_controller import ConfiguracionSubrogadoOnline

from preprocesado_de_datos.utils.path_utils import leer_json


DEFAULT_RBF_PARAMS_JSON = (
    ROOT
    / "surrogate_models"
    / "configs"
    / "rbf_multiquadric_eps1_smoothing1e-3_neighbors50.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta metaheuristicas online con filtro subrogado sobre CEC2017. "
            "Actualmente soporta AGE, DE y SHADE online."
        )
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help=(
            "Lista explicita de semillas. Acepta valores separados por espacios "
            "o comas. Si se indica, tiene prioridad sobre --seed-start y --n-seeds."
        ),
    )
    parser.add_argument(
        "--tam-poblacion",
        type=validar_tam_poblacion,
        default=50,
        help="Tamano de poblacion utilizado por las metaheuristicas.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Presupuesto maximo de evaluaciones reales. Por defecto: 10000 * dimension.",
    )
    parser.add_argument(
        "--reinicio",
        "--reinicio-elitista",
        dest="reinicio_elitista",
        action="store_true",
        help="Activa el reinicio elitista.",
    )
    parser.add_argument(
        "--reinicio-ratio",
        "--reinicio-elitista-ratio-paciencia",
        dest="reinicio_elitista_ratio_paciencia",
        type=float,
        default=None,
        help="Fraccion de max_evals sin mejora antes de permitir reinicio.",
    )
    parser.add_argument(
        "--cec-funcid",
        nargs="+",
        required=True,
        help="Funciones CEC: lista, CSV o 'all'.",
    )
    parser.add_argument("--cec-dim", type=int, choices=[2, 5, 10, 30, 50], default=10)
    parser.add_argument(
        "--algoritmo",
        nargs="+",
        default=["age"],
        help=(
            "Metaheuristica online a ejecutar. Acepta age, de, shade, all, "
            "listas separadas por espacios o listas separadas por comas."
        ),
    )
    parser.add_argument(
        "--sin-metricas",
        action="store_true",
        help="Si se indica, no registra metricas DEAP ni datasets por run.",
    )
    parser.add_argument(
        "--sin-dataset",
        action="store_true",
        help=(
            "Si se registran metricas, evita guardar datasets de muestras "
            "evaluadas por run. Mantiene metricas, resumenes y reinicios."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outdir", type=str, default="results/cec/cec2017_d10_tam50_online")

    parser.add_argument("--surrogate-model", default="rbf")
    parser.add_argument(
        "--modelo-params-json",
        "--surrogate-params-json",
        dest="modelo_params_json",
        default=None,
        help=(
            "JSON con hiperparametros del modelo subrogado. Si no se indica y "
            "el modelo es rbf, se usa la configuracion RBF ajustada."
        ),
    )
    parser.add_argument(
        "--surrogate-prob",
        "--probabilidad-subrogado",
        dest="probabilidad_subrogado",
        type=float,
        default=0.50,
        help="Probabilidad p de aplicar el filtro subrogado a un candidato.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.20)
    parser.add_argument("--window-ratio", type=float, default=0.20)
    parser.add_argument(
        "--cooldown-reinicio-evals",
        type=int,
        default=0,
        help=(
            "Evaluaciones reales de enfriamiento tras cada reinicio. "
            "Usa 0 para desactivarlo."
        ),
    )
    parser.add_argument(
        "--guardar-decisiones-subrogado",
        action="store_true",
        help=(
            "Guarda decisiones_subrogado.csv con fitness_pred, fitness_ref y "
            "margen_pred_ref para analizar el filtro."
        ),
    )
    
    parser.add_argument(
        "--retrain-ratio",
        type=float,
        default=0.25,
        help=(
            "Fraccion de la ventana de entrenamiento que debe renovarse antes "
            "de reentrenar el subrogado. Por ejemplo, 0.25 con window_ratio=0.20 "
            "y max_evals=10000 implica reentrenar cada 500 evaluaciones reales."
        ),
    )
    return parser.parse_args()


def mostrar(args, *valores, **kwargs):
    if args.verbose:
        print(*valores, **kwargs)


def gestiona_algoritmos_online(valores):
    """
    Normaliza la seleccion de algoritmos online.

    Permite:
        --algoritmo age
        --algoritmo de shade
        --algoritmo age,de
        --algoritmo all
    """
    tokens = []
    for valor in valores or ["age"]:
        for token in str(valor).split(","):
            token = normalizar_algoritmo(token.strip())
            if token:
                tokens.append(token)

    if not tokens:
        return ["age"]

    if "all" in tokens:
        return ["age", "de", "shade"]

    permitidos = {"age", "de", "shade"}
    algoritmos = []
    for token in tokens:
        if token not in permitidos:
            raise ValueError(
                f"Algoritmo online no soportado: {token}. "
                "Usa age, de, shade o all."
            )
        if token not in algoritmos:
            algoritmos.append(token)

    return algoritmos


def validar_args(args):
    args.reinicio_elitista_ratio_paciencia = normalizar_ratio_paciencia_reinicio(
        args.reinicio_elitista_ratio_paciencia
    )
    args.reinicio_elitista = bool(args.reinicio_elitista)

    if args.max_evals is None:
        args.max_evals = 10000 * int(args.cec_dim)
    if int(args.max_evals) <= 0:
        raise ValueError("--max-evals debe ser positivo.")

    if args.reinicio_elitista and args.reinicio_elitista_ratio_paciencia is None:
        raise ValueError("--reinicio requiere indicar --reinicio-ratio.")
    if not args.reinicio_elitista and args.reinicio_elitista_ratio_paciencia is not None:
        raise ValueError("--reinicio-ratio solo puede utilizarse junto con --reinicio.")

    args.algoritmos = gestiona_algoritmos_online(args.algoritmo)


def cargar_modelo_params(modelo_nombre, params_json):
    modelo_nombre = str(modelo_nombre).lower()
    if params_json is not None:
        return dict(leer_json(params_json))
    if modelo_nombre == "rbf":
        return dict(leer_json(DEFAULT_RBF_PARAMS_JSON))
    return {}


def etiqueta_probabilidad(probabilidad):
    return f"p{int(round(float(probabilidad) * 100)):03d}"


def etiqueta_ratio(valor):
    return f"rt{int(round(float(valor) * 100)):03d}"


def construir_config_subrogado(args, seed):
    return ConfiguracionSubrogadoOnline(
        modelo_nombre=str(args.surrogate_model).lower(),
        modelo_params=cargar_modelo_params(args.surrogate_model, args.modelo_params_json),
        cooldown_reinicio_evals=int(args.cooldown_reinicio_evals),
        warmup_ratio=float(args.warmup_ratio),
        window_ratio=float(args.window_ratio),
        probabilidad_subrogado=float(args.probabilidad_subrogado),
        max_evals=int(args.max_evals),
        minimizacion=True,
        seed=int(seed),
        retrain_ratio=float(args.retrain_ratio),
    )


def construir_age_online(args, seed):
    kwargs = {
        "seed": int(seed),
        "tam_poblacion": int(args.tam_poblacion),
        "max_evals": int(args.max_evals),
    }
    if args.reinicio_elitista:
        kwargs["reinicio_elitista"] = True
        kwargs["reinicio_elitista_ratio_paciencia"] = float(
            args.reinicio_elitista_ratio_paciencia
        )

    return GeneticStationaryCEC2017Online(
        surrogate_config=construir_config_subrogado(args, seed),
        **kwargs,
    )


def construir_de_online(args, seed):
    kwargs = {
        "seed": int(seed),
        "tam_poblacion": int(args.tam_poblacion),
        "max_evals": int(args.max_evals),
    }
    if args.reinicio_elitista:
        kwargs["reinicio_elitista"] = True
        kwargs["reinicio_elitista_ratio_paciencia"] = float(
            args.reinicio_elitista_ratio_paciencia
        )

    return DifferentialEvolutionCEC2017Online(
        surrogate_config=construir_config_subrogado(args, seed),
        **kwargs,
    )


def construir_shade_online(args, seed):
    kwargs = {
        "seed": int(seed),
        "tam_poblacion": int(args.tam_poblacion),
        "max_evals": int(args.max_evals),
    }

    if args.reinicio_elitista:
        kwargs["reinicio_elitista"] = True
        kwargs["reinicio_elitista_ratio_paciencia"] = float(
            args.reinicio_elitista_ratio_paciencia
        )

    return SHADECEC2017Online(
        surrogate_config=construir_config_subrogado(args, seed),
        **kwargs,
    )
    
def construir_metaheuristica_online(args, seed, algoritmo):
    if algoritmo == "age":
        return construir_age_online(args, seed)
    if algoritmo == "de":
        return construir_de_online(args, seed)
    if algoritmo == "shade":
        return construir_shade_online(args, seed)
    raise NotImplementedError(f"Algoritmo online no soportado: {algoritmo}")


def fila_run(args, algoritmo, funcid, seed, resultado, tiempo_s, ruta_metricas, adaptacion):
    resumen_online = dict(resultado.get("resumen_online", {}))
    return {
        "problema": "cec2017",
        "algoritmo": str(algoritmo),
        "adaptacion": adaptacion,
        "cec_funcid": int(funcid),
        "semilla": int(seed),
        "fitness": float(resultado["mejor_fitness"]),
        "cec_error": float(resultado["mejor_error"]),
        "tiempo_s": float(tiempo_s),
        "ruta_metricas": str(ruta_metricas or ""),
        "ruta_resumen_online": str(resultado.get("ruta_resumen_online", "")),
        "reinicio_elitista": bool(args.reinicio_elitista),
        "reinicio_elitista_ratio_estabilidad_diversidad": "",
        "reinicio_elitista_ratio_paciencia": (
            float(args.reinicio_elitista_ratio_paciencia)
            if args.reinicio_elitista
            else ""
        ),
        "reinicio_elitista_ventana_evaluaciones": (
            int(VENTANA_DIAGNOSTICA_REINICIO) if args.reinicio_elitista else ""
        ),
        "n_reinicios_elitistas": int(len(resultado.get("reinicios_elitistas", []))),
        "modelo_subrogado": str(resumen_online.get("modelo_nombre", args.surrogate_model)),
        "probabilidad_subrogado": float(args.probabilidad_subrogado),
        "warmup_ratio": float(args.warmup_ratio),
        "window_ratio": float(args.window_ratio),
        "cooldown_reinicio_evals": int(args.cooldown_reinicio_evals),
        "evals_reales": int(resumen_online.get("evals_reales", 0)),
        "candidatos_generados": int(resumen_online.get("candidatos_generados", 0)),
        "candidatos_con_subrogado": int(resumen_online.get("candidatos_con_subrogado", 0)),
        "candidatos_evaluados_reales": int(
            resumen_online.get("candidatos_evaluados_reales", 0)
        ),
        "candidatos_aceptados_por_subrogado": int(
            resumen_online.get("candidatos_aceptados_por_subrogado", 0)
        ),
        "candidatos_rechazados": int(resumen_online.get("candidatos_rechazados", 0)),
        "porcentaje_rechazo": float(resumen_online.get("porcentaje_rechazo", 0.0)),
        "entrenamientos_rbf": int(resumen_online.get("entrenamientos_rbf", 0)),
        "tiempo_entrenamiento_total": float(
            resumen_online.get("tiempo_entrenamiento_total", 0.0)
        ),
        "tiempo_prediccion_total": float(
            resumen_online.get("tiempo_prediccion_total", 0.0)
        ),
        "tiempo_online_total": float(resumen_online.get("tiempo_online_total", 0.0)),
        "retrain_ratio": float(resumen_online.get("retrain_ratio", args.retrain_ratio)),
        "retrain_interval_efectivo": int(
            resumen_online.get("retrain_interval_efectivo", 0)
        ),
        "ruta_decisiones_subrogado": resultado.get("ruta_decisiones_subrogado_csv", ""),
    }


def ejecutar_cec_online(args, algoritmo, semillas, outdir_metricas, funcid):
    filas = []
    registrar_metricas = not args.sin_metricas
    ratio_paciencia_reinicio = args.reinicio_elitista_ratio_paciencia
    reinicio_activo = bool(args.reinicio_elitista)
    sufijo_reinicio = (
        "_reinicio" + sufijo_ratio_paciencia_reinicio(ratio_paciencia_reinicio)
        if reinicio_activo
        else ""
    )
    prob_label = etiqueta_probabilidad(args.probabilidad_subrogado)
    retrain_label = etiqueta_ratio(args.retrain_ratio)
    prefijo_adaptacion = {"age": "ao", "de": "do", "shade": "so"}[algoritmo]
    adaptacion = (
        f"{prefijo_adaptacion}_{prob_label}_c{int(args.cooldown_reinicio_evals)}_{retrain_label}"
    )

    for run_idx, seed in enumerate(semillas, start=1):
        mostrar(
            args,
            f"[ONLINE CEC2017 F{int(funcid)} {run_idx}/{len(semillas)}] "
            f"{algoritmo.upper()} seed={seed} p={float(args.probabilidad_subrogado):.2f}...",
            flush=True,
        )

        metaheuristica = construir_metaheuristica_online(args, seed, algoritmo)
        run_id = (
            f"{algoritmo}_online_cec2017_f{int(funcid)}_d{int(args.cec_dim)}_s{int(seed)}"
            f"_{prob_label}_cd{int(args.cooldown_reinicio_evals)}_{retrain_label}"
            f"{sufijo_reinicio}"
        )

        t0 = time.perf_counter()
        resultado = metaheuristica.optimize(
            funcid=int(funcid),
            dim=int(args.cec_dim),
            seed=int(seed),
            algname=adaptacion,
            registrar_metricas=registrar_metricas,
            ruta_metricas=(
                str((outdir_metricas / "cec2017" / algoritmo).resolve())
                if registrar_metricas or args.guardar_decisiones_subrogado
                else None
            ),
            run_id=run_id,
            cec_workdir=str(outdir_metricas.resolve().parent),
            guardar_decisiones_subrogado=bool(args.guardar_decisiones_subrogado),
            guardar_dataset=not bool(args.sin_dataset),
        )
        tiempo_s = time.perf_counter() - t0

        filas.append(
            fila_run(
                args=args,
                algoritmo=algoritmo,
                funcid=funcid,
                seed=seed,
                resultado=resultado,
                tiempo_s=tiempo_s,
                ruta_metricas=resultado.get("ruta_metricas", ""),
                adaptacion=adaptacion,
            )
        )

        mostrar(
            args,
            f"[ONLINE CEC2017 F{int(funcid)}] {algoritmo.upper()} seed={seed} "
            f"fitness={float(resultado['mejor_fitness']):.6f} "
            f"error={float(resultado['mejor_error']):.6f} "
            f"tiempo={tiempo_s:.4f}s",
            flush=True,
        )

    return filas


def config_base(args, semillas, funcids):
    return {
        "seed_start": int(args.seed_start) if args.seeds is None else None,
        "n_seeds": int(args.n_seeds) if args.seeds is None else len(semillas),
        "seeds": [int(s) for s in semillas],
        "tam_poblacion": int(args.tam_poblacion),
        "max_evals": int(args.max_evals),
        "cec_funcid_raw": list(args.cec_funcid),
        "funcids_resueltas": [int(f) for f in funcids],
        "cec_dim": int(args.cec_dim),
        "algoritmos": [str(a) for a in args.algoritmos],
        "problema": "cec2017",
        "registrar_metricas": bool(not args.sin_metricas),
        "guardar_dataset": bool(not args.sin_dataset),
        "reinicio_elitista": bool(args.reinicio_elitista),
        "reinicio_elitista_ratio_estabilidad_diversidad": None,
        "reinicio_elitista_ratio_paciencia": (
            float(args.reinicio_elitista_ratio_paciencia)
            if args.reinicio_elitista
            else None
        ),
        "reinicio_elitista_ventana_evaluaciones": (
            int(VENTANA_DIAGNOSTICA_REINICIO) if args.reinicio_elitista else None
        ),
        "surrogate_model": str(args.surrogate_model).lower(),
        "modelo_params_json": (
            str(Path(args.modelo_params_json).resolve())
            if args.modelo_params_json is not None
            else str(DEFAULT_RBF_PARAMS_JSON)
        ),
        "probabilidad_subrogado": float(args.probabilidad_subrogado),
        "warmup_ratio": float(args.warmup_ratio),
        "window_ratio": float(args.window_ratio),
        "cooldown_reinicio_evals": int(args.cooldown_reinicio_evals),
        "retrain_ratio": float(args.retrain_ratio),
        "guardar_decisiones_subrogado": bool(args.guardar_decisiones_subrogado),
    }


def main():
    args = parse_args()
    validar_args(args)

    semillas = gestiona_semillas(args)
    funcids_cec = gestiona_funcids_cec(args)

    outdir_raiz = Path(args.outdir).resolve()
    nombre_experimento = outdir_raiz.name if outdir_raiz.name else "online"
    outdir_cec = (ROOT / "results" / "cec" / nombre_experimento).resolve()

    mostrar(args, "Configuracion online:", flush=True)
    mostrar(args, f"  algoritmos={','.join(str(a) for a in args.algoritmos)}", flush=True)
    mostrar(args, f"  funciones={','.join(str(f) for f in funcids_cec)}", flush=True)
    mostrar(args, f"  semillas={','.join(str(s) for s in semillas)}", flush=True)
    mostrar(args, f"  max_evals={int(args.max_evals)}", flush=True)
    mostrar(args, f"  p={float(args.probabilidad_subrogado)}", flush=True)
    mostrar(args, f"  warmup_ratio={float(args.warmup_ratio)}", flush=True)
    mostrar(args, f"  window_ratio={float(args.window_ratio)}", flush=True)
    mostrar(args, f"  cooldown_reinicio_evals={int(args.cooldown_reinicio_evals)}", flush=True)
    mostrar(args, f"  retrain_ratio={float(args.retrain_ratio)}", flush=True)
    mostrar(args, f"  guardar_dataset={not bool(args.sin_dataset)}", flush=True)
    mostrar(args, f"  guardar_decisiones_subrogado={bool(args.guardar_decisiones_subrogado)}", flush=True)
    mostrar(args, f"  outdir={outdir_cec}", flush=True)

    filas_runs_cec = []
    for idx, funcid in enumerate(funcids_cec, start=1):
        mostrar(args, f"\nCEC2017 online: F{int(funcid)} ({idx}/{len(funcids_cec)})")
        outdir_funcid = outdir_cec / f"f{int(funcid)}"
        outdir_metricas_funcid = outdir_funcid / "metricas_runs"

        filas_funcid = []
        for alg_idx, algoritmo in enumerate(args.algoritmos, start=1):
            mostrar(
                args,
                f"  Algoritmo online: {algoritmo.upper()} ({alg_idx}/{len(args.algoritmos)})",
                flush=True,
            )
            filas_funcid.extend(
                ejecutar_cec_online(
                    args,
                    algoritmo,
                    semillas,
                    outdir_metricas_funcid,
                    int(funcid),
                )
            )
        filas_runs_cec.extend(filas_funcid)

        config_funcid = config_base(args, semillas, [int(funcid)])
        config_funcid["funcid"] = int(funcid)
        guardar_bloque_resultados(
            outdir_funcid,
            filas_funcid,
            config_funcid,
            incluir_columnas_contexto=False,
        )

    if not filas_runs_cec:
        raise RuntimeError("No se ejecuto ninguna combinacion online.")

    config_cec = config_base(args, semillas, funcids_cec)
    filas_resumen_contexto = construir_resumen(filas_runs_cec, desglosar_contexto=True)
    guardar_csv(outdir_cec / "resumen_funciones_promedio.csv", filas_resumen_contexto)
    guardar_json(
        outdir_cec / "resumen_funciones_promedio.json",
        {
            "config": config_cec,
            "resumen": filas_resumen_contexto,
        },
    )

    mostrar(args, f"\nResultados online guardados en: {outdir_cec}", flush=True)


if __name__ == "__main__":
    main()
