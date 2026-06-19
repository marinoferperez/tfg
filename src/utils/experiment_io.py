"""
Utilidades de entrada/salida compartidas entre scripts de experimentacion.

Agrupa las funciones de gestion de semillas, funciones CEC, construccion de
resumenes y escritura de resultados que son comunes a los scripts offline y
online. De este modo, los scripts de ejecucion no necesitan importarse entre si.
"""

import numpy as np
from collections import defaultdict
from pathlib import Path

from src.utils.file_io import escribir_csv_dicts, leer_csv_dicts


# ---------------------------------------------------------------------------
# Normalizacion y validacion de argumentos CLI
# ---------------------------------------------------------------------------

def validar_tam_poblacion(valor):
    """
    Valida el tamano de poblacion recibido por CLI.

    valor: tamano de poblacion solicitado. Debe ser un entero >= 4.
    """
    import argparse
    valor = int(valor)
    if valor < 4:
        raise argparse.ArgumentTypeError("debe ser un entero >= 4")
    return valor


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
    Construye el sufijo usado en nombres de fichero cuando el reinicio esta activo.

    valor: ratio de paciencia ya validado.
    """
    txt = f"{float(valor):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"_pat{txt}"


# ---------------------------------------------------------------------------
# Gestion de argumentos de experimento
# ---------------------------------------------------------------------------

def gestiona_funcids_cec(args):
    """
    Resuelve las funciones CEC solicitadas desde la linea de comandos.

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
            raise ValueError(
                f"Valor de --cec-funcid invalido: '{tk}'. Usa enteros en [1, 30] o 'all'."
            ) from exc
        if not 1 <= fid <= 30:
            raise ValueError(f"funcid={fid} fuera de rango. Debe estar en [1, 30].")
        if fid not in vistos:
            vistos.add(fid)
            funcids.append(fid)
    return funcids


def gestiona_semillas(args):
    """
    Resuelve las semillas a ejecutar desde la linea de comandos.

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
            raise ValueError(
                f"Valor de --seeds invalido: '{tk}'. Usa enteros positivos."
            ) from exc
        if seed < 1:
            raise ValueError(f"Semilla invalida: {seed}. Debe ser >= 1.")
        if seed not in vistas:
            vistas.add(seed)
            semillas.append(seed)
    return semillas


# ---------------------------------------------------------------------------
# Fusion y construccion de resumenes
# ---------------------------------------------------------------------------

COLUMNAS_RUNS_DESCARTADAS = {
    "problema",
    "adaptacion",
    "qap_instancia",
    "ruta_metricas",
    "ruta_resumen_online",
    "ruta_decisiones_subrogado",
}


def normalizar_fila_run_csv(fila):
    """
    Elimina columnas de trazabilidad que no forman parte del CSV final de runs.
    """
    return {
        clave: valor
        for clave, valor in dict(fila).items()
        if clave not in COLUMNAS_RUNS_DESCARTADAS
    }


def clave_fila_run(fila):
    """
    Construye la clave unica de una ejecucion para detectar duplicados.

    fila: diccionario con los campos principales de la ejecucion.
    """
    return (
        str(fila.get("algoritmo", "")),
        str(fila.get("cec_funcid", "")),
        int(fila.get("semilla", 0)),
        str(fila.get("reinicio", "")),
        str(fila.get("reinicio_ratio", "")),
        str(fila.get("modelo_subrogado", "")),
        str(fila.get("probabilidad_subrogado", "")),
        str(fila.get("warmup_ratio", "")),
        str(fila.get("window_ratio", "")),
        str(fila.get("cooldown_reinicio_evals", "")),
        str(fila.get("retrain_ratio", "")),
        str(fila.get("retrain_interval_efectivo", "")),
    )


def fusionar_runs_existentes(path, filas_nuevas):
    """
    Fusiona runs existentes con runs nuevas evitando duplicados por clave.

    path: ruta del runs.csv existente.
    filas_nuevas: ejecuciones generadas en la llamada actual.
    """
    path = Path(path)
    existentes = [normalizar_fila_run_csv(fila) for fila in leer_csv_dicts(path)]
    filas_nuevas = [normalizar_fila_run_csv(fila) for fila in filas_nuevas]
    if not existentes:
        return list(filas_nuevas)

    fusionadas = {clave_fila_run(fila): fila for fila in existentes}
    for fila in filas_nuevas:
        fusionadas[clave_fila_run(fila)] = fila
    return sorted(
        fusionadas.values(),
        key=lambda fila: (
            str(fila.get("algoritmo", "")),
            int(float(fila.get("cec_funcid") or 0)),
            int(float(fila.get("semilla") or 0)),
            str(fila.get("modelo_subrogado", "")),
            str(fila.get("probabilidad_subrogado", "")),
            str(fila.get("cooldown_reinicio_evals", "")),
            str(fila.get("retrain_ratio", "")),
        ),
    )


def resumen_grupo(filas):
    """
    Calcula medias de fitness y tiempo para un grupo de runs del mismo algoritmo.

    filas: ejecuciones del mismo algoritmo y contexto.
    """
    fitness = np.asarray([float(f["fitness"]) for f in filas], dtype=float)
    tiempos = np.asarray([float(f["tiempo_s"]) for f in filas], dtype=float)
    return {
        "n_runs": int(len(filas)),
        "fitness_media": float(np.mean(fitness)),
        "tiempo_media_s": float(np.mean(tiempos)),
    }


def construir_resumen(filas_runs, desglosar_contexto=False, incluir_columnas_contexto=True):
    """
    Agrega las ejecuciones individuales en resumenes medios por algoritmo.

    filas_runs: filas individuales de runs.csv.
    desglosar_contexto: si True, separa el resumen por función CEC.
    incluir_columnas_contexto: conserva columnas vacias cuando no se desglosa.
    """
    grupos = defaultdict(list)
    for fila in filas_runs:
        if desglosar_contexto:
            clave = (
                fila["algoritmo"],
                fila.get("cec_funcid", ""),
            )
        else:
            clave = (fila["algoritmo"],)
        grupos[clave].append(fila)

    filas_resumen = []
    for clave, filas in sorted(grupos.items()):
        if desglosar_contexto:
            algoritmo, cec_funcid = clave
        else:
            algoritmo = clave[0]
            cec_funcid = ""

        resumen = resumen_grupo(filas)
        fila_resumen = {"algoritmo": algoritmo}
        if desglosar_contexto:
            fila_resumen["cec_funcid"] = cec_funcid
        elif incluir_columnas_contexto:
            fila_resumen["cec_funcid"] = cec_funcid
        fila_resumen.update(resumen)
        filas_resumen.append(fila_resumen)
    return filas_resumen


def guardar_bloque_resultados(outdir, filas_runs, config_json, incluir_columnas_contexto=True):
    """
    Guarda runs.csv de un bloque experimental.

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
    escribir_csv_dicts(outdir / "runs.csv", filas_runs)
    return filas_resumen


# ---------------------------------------------------------------------------
# I/O de artefactos de ejecución
# ---------------------------------------------------------------------------

def guardar_reinicios_elitistas_csv(ruta_base, eventos):
    """
    Guarda los eventos de reinicio elitista en un CSV dentro de ruta_base.

    ruta_base: directorio donde se escribe reinicios_elitistas.csv.
    eventos: lista de dicts generados por ControlReinicioElitista.

    Retorna la ruta del CSV generado, o None si no hay eventos.
    """
    eventos = list(eventos or [])
    if not eventos:
        return None

    ruta_csv = Path(ruta_base) / "reinicios_elitistas.csv"
    eventos_normalizados = []
    for evento in eventos:
        fila = dict(evento)
        if "evals_antes_reinicio" not in fila:
            fila["evals_antes_reinicio"] = fila.pop("evaluaciones_antes_reinicio", None)
        if "evals_despues_reinicio" not in fila:
            fila["evals_despues_reinicio"] = fila.pop("evaluaciones_despues_reinicio", None)
        eventos_normalizados.append(fila)

    fieldnames = [
        "generacion", "evals_antes_reinicio", "evals_despues_reinicio",
        "mejor_fitness", "segundo_mejor_fitness", "evals_desde_mejora_mejor",
        "evals_desde_mejora_segundo", "paciencia_evals", "criterio_mejor_estancado",
        "criterio_segundo_estancado", "criterio_fitness_estancado", "criterio_reinicio",
        "reinicio", "indice_individuo_preservado", "fitness_preservado",
    ]
    escribir_csv_dicts(ruta_csv, eventos_normalizados, fieldnames=fieldnames)
    return str(ruta_csv)


def guardar_decisiones_subrogado_csv(ruta_base, decisiones):
    """
    Guarda el historial de decisiones del subrogado en un CSV dentro de ruta_base.

    ruta_base: directorio donde se escribe decisiones_subrogado.csv.
    decisiones: lista de dicts de EstadisticasSubrogado.decisiones_subrogado.

    Retorna la ruta del CSV generado, o None si no hay decisiones.
    """
    decisiones = list(decisiones or [])
    if not decisiones:
        return None

    ruta_csv = Path(ruta_base) / "decisiones_subrogado.csv"
    fieldnames = [
        "evals_reales", "generacion", "reinicios", "evals_desde_reinicio",
        "fitness_pred", "fitness_ref", "margen_pred_ref", "debe_evaluar", "motivo",
    ]
    escribir_csv_dicts(ruta_csv, decisiones, fieldnames=fieldnames)
    return str(ruta_csv)


def mostrar(args, *valores, **kwargs):
    """Imprime mensajes solo si args.verbose esta activo."""
    if args.verbose:
        print(*valores, **kwargs)
