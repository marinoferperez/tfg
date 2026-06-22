"""Construccion de casos de entrenamiento/validacion por bloques temporales del 20%."""

import hashlib
import numpy as np

N_BLOQUES = 5
PCT_POR_BLOQUE = 100 // N_BLOQUES
VAL_RATIO_TRAIN = 0.25
KEY_N_BLOQUES_UTILES = "_n_bloques_utiles"
KEY_CONVERGENCIA_DETALLE = "_convergencia_detalle"
TOL_MEJORA_BLOQUE_ABS = 1e-8
TOL_MEJORA_BLOQUE_REL = 1e-4


def detectar_ultimo_bloque_informativo(fitness, n_bloques=N_BLOQUES, *, tol_abs=TOL_MEJORA_BLOQUE_ABS, tol_rel=TOL_MEJORA_BLOQUE_REL, devolver_detalle=False):
    """
    Detecta el ultimo bloque con mejora practica en el fitness.

    fitness: array con la evolucion temporal del fitness.
    n_bloques: numero de bloques en que se divide la secuencia.
    tol_abs: tolerancia absoluta minima de mejora para considerar un bloque informativo.
    tol_rel: tolerancia relativa de mejora respecto al mejor fitness al inicio del bloque.
    devolver_detalle: si True, devuelve tambien la lista de dicts con el detalle por bloque.
    """
    fitness = np.asarray(fitness, dtype=float)
    n = len(fitness)
    if n == 0:
        if devolver_detalle:
            return 0, []
        return 0

    mejor_hasta_ahora = np.minimum.accumulate(fitness)
    ventanas = np.array_split(np.arange(n), n_bloques)
    ultimo_bloque = 0
    detalle = []

    for bloque_id, ventana in enumerate(ventanas, 1):
        if len(ventana) == 0:
            continue
        mejor_inicio = float(mejor_hasta_ahora[int(ventana[0])])
        mejor_fin = float(mejor_hasta_ahora[int(ventana[-1])])
        mejora = float(mejor_inicio - mejor_fin)
        tolerancia = float(max(tol_abs, tol_rel * max(1.0, abs(mejor_inicio))))
        informativo = bool(mejora > tolerancia)
        if informativo:
            ultimo_bloque = int(bloque_id)
        detalle.append({"bloque": int(bloque_id), "mejor_inicio": mejor_inicio, "mejor_fin": mejor_fin, "mejora": mejora, "tolerancia": tolerancia, "informativo": informativo})

    if devolver_detalle:
        return ultimo_bloque, detalle
    return ultimo_bloque


def truncar_por_convergencia(dataset, n_bloques=N_BLOQUES):
    """
    Recorta el dataset al ultimo bloque con mejora practica.

    dataset: dict con arrays de evaluaciones (eval_id, fitness, x, seed…).
    n_bloques: numero de bloques en que se divide la secuencia.
    """
    fitness = np.asarray(dataset["fitness"], dtype=float)
    n = len(fitness)

    ultimo_bloque, detalle = detectar_ultimo_bloque_informativo(fitness, n_bloques, devolver_detalle=True)
    if ultimo_bloque < 1:
        ventanas = np.array_split(np.arange(n), n_bloques)
        idx_corte = int(ventanas[0][-1]) + 1 if ventanas and len(ventanas[0]) else 0
        dataset_truncado = {k: np.asarray(v)[:idx_corte] for k, v in dataset.items() if not str(k).startswith("_")}
        dataset_truncado[KEY_N_BLOQUES_UTILES] = 1
        dataset_truncado[KEY_CONVERGENCIA_DETALLE] = detalle
        return dataset_truncado, 0, idx_corte / n if n else 0.0

    ventanas = np.array_split(np.arange(n), n_bloques)
    idx_corte = int(ventanas[ultimo_bloque - 1][-1]) + 1

    if idx_corte >= n:
        dataset[KEY_CONVERGENCIA_DETALLE] = detalle
        return dataset, ultimo_bloque, 1.0

    dataset_truncado = {k: np.asarray(v)[:idx_corte] for k, v in dataset.items() if not str(k).startswith("_")}
    dataset_truncado[KEY_N_BLOQUES_UTILES] = int(ultimo_bloque)
    dataset_truncado[KEY_CONVERGENCIA_DETALLE] = detalle
    return dataset_truncado, ultimo_bloque, idx_corte / n


def _rango_porcentual(bloque_id):
    """
    Calcula el rango porcentual de un bloque dado su identificador.

    bloque_id: indice del bloque (empieza en 1).
    """
    inicio = 1 if bloque_id == 1 else ((bloque_id - 1) * PCT_POR_BLOQUE) + 1
    fin = bloque_id * PCT_POR_BLOQUE
    return int(inicio), int(fin)


def _nombre_bloque(pct_ini, pct_fin):
    """
    Devuelve la etiqueta textual de un bloque.

    pct_ini: porcentaje de inicio del bloque.
    pct_fin: porcentaje de fin del bloque.
    """
    return f"{int(pct_ini)}-{int(pct_fin)}%"


def construir_bloques_por_eval_id(dataset):
    """
    Divide el dataset en bloques de igual tamaño segun el orden temporal de eval_id.

    dataset: dict con arrays de evaluaciones. Debe contener 'eval_id' y opcionalmente KEY_N_BLOQUES_UTILES.
    """
    eval_id = np.asarray(dataset["eval_id"])
    n_bloques_utiles = int(dataset.get(KEY_N_BLOQUES_UTILES, N_BLOQUES))
    if n_bloques_utiles < 1:
        raise ValueError("El numero de bloques utiles debe ser >= 1.")

    indices_orden_temporal = np.arange(eval_id.shape[0], dtype=np.int64)
    trozos = np.array_split(indices_orden_temporal, n_bloques_utiles)

    bloques = []
    for bloque_id, idx in enumerate(trozos, 1):
        pct_ini, pct_fin = _rango_porcentual(bloque_id)
        bloques.append({
            "bloque_id": int(bloque_id),
            "pct_ini": int(pct_ini),
            "pct_fin": int(pct_fin),
            "etiqueta_bloque": _nombre_bloque(pct_ini, pct_fin),
            "indices": np.asarray(idx, dtype=np.int64),
        })

    return bloques


def calcular_n_muestras_validacion_fija(bloques):
    """
    Calcula el numero fijo de muestras de validacion a partir del primer bloque.

    bloques: lista de dicts devuelta por construir_bloques_por_eval_id.
    """
    if not bloques:
        raise ValueError("No hay bloques para calcular el tamaño fijo de validación.")
    
    n_train_base = int(np.asarray(bloques[0]["indices"], dtype=np.int64).size)
    n_muestras_val = int(np.floor(n_train_base * float(VAL_RATIO_TRAIN)))
    
    if n_muestras_val < 1:
        raise ValueError("El tamaño fijo de validación debe ser >= 1.")
    
    return n_muestras_val


def muestrear_validacion_siguiente(validacion_idx, n_muestras_val, random_state, seed, bloque_id):
    """
    Muestrea aleatoriamente n_muestras_val indices del bloque de validacion.

    validacion_idx: indices del bloque de validacion disponibles.
    n_muestras_val: numero de muestras a seleccionar.
    random_state: semilla base del experimento.
    seed: semilla del dataset actual.
    bloque_id: identificador del bloque actual (se usa como parte de la semilla).
    """
    validacion_idx = np.asarray(validacion_idx, dtype=np.int64)

    if validacion_idx.size < n_muestras_val:
        raise ValueError(f"No hay suficientes muestras en el bloque de validación: disponibles={validacion_idx.size}, requeridas={n_muestras_val}")

    rng = np.random.default_rng([int(random_state), int(seed), int(bloque_id)])
    seleccion = rng.choice(validacion_idx, size=n_muestras_val, replace=False)
    return np.sort(np.asarray(seleccion, dtype=np.int64))


def obtener_bloque_validacion_siguiente(bloques, idx_bloque):
    """
    Devuelve el bloque inmediatamente posterior al bloque de entrenamiento actual.

    bloques: lista completa de bloques del dataset.
    idx_bloque: indice del bloque de entrenamiento actual en la lista.
    """
    return list(bloques[idx_bloque + 1:])[:1]


def _construir_caso_temporal(*, seed, bloque_entrenamiento, train_pct_ini, train_pct_fin, etiqueta_bloque, bloques_entrenamiento, bloque_validacion, train_idx, val_idx):
    """
    Construye el dict que describe un caso de entrenamiento/validacion.

    seed: semilla del dataset.
    bloque_entrenamiento: identificador del bloque de entrenamiento actual.
    train_pct_ini: porcentaje de inicio del rango de entrenamiento.
    train_pct_fin: porcentaje de fin del rango de entrenamiento.
    etiqueta_bloque: etiqueta textual del rango de entrenamiento.
    bloques_entrenamiento: lista de identificadores de bloques usados en entrenamiento.
    bloque_validacion: identificador del bloque de validacion.
    train_idx: indices de posicion en el dataset para entrenamiento.
    val_idx: indices de posicion en el dataset para validacion.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    return {
        "seed": int(seed),
        "bloque_entrenamiento": int(bloque_entrenamiento),
        "train_pct_ini": int(train_pct_ini),
        "train_pct_fin": int(train_pct_fin),
        "etiqueta_bloque": str(etiqueta_bloque),
        "bloques_entrenamiento": [int(item) for item in bloques_entrenamiento],
        "bloque_validacion": int(bloque_validacion),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_idx_hash": hash_indices(train_idx),
        "val_idx_hash": hash_indices(val_idx),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
    }


def hash_indices(indices):
    """
    Calcula un hash SHA-256 truncado de un array de indices.

    indices: array de indices a hashear.
    """
    indices = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(indices.tobytes()).hexdigest()[:16]


def construir_casos_no_acumulativos(dataset, random_state=42):
    """
    Construye casos de entrenamiento donde cada bloque se usa de forma independiente.

    dataset: dict con arrays de evaluaciones del dataset.
    random_state: semilla base para el muestreo de validacion.
    """
    seed = int(np.asarray(dataset["seed"])[0])
    bloques = construir_bloques_por_eval_id(dataset)
    n_muestras_val_fija = calcular_n_muestras_validacion_fija(bloques)
    casos = []

    for idx_bloque, bloque in enumerate(bloques[:-1]):
        train_idx = np.asarray(bloque["indices"], dtype=np.int64)
        if train_idx.size == 0:
            continue

        bloques_validacion = obtener_bloque_validacion_siguiente(bloques, idx_bloque)
        validacion_idx = np.concatenate([np.asarray(item["indices"], dtype=np.int64) for item in bloques_validacion])
        if validacion_idx.size == 0:
            continue

        val_idx = muestrear_validacion_siguiente(validacion_idx, n_muestras_val_fija, random_state=random_state, seed=seed, bloque_id=bloque["bloque_id"])

        casos.append(_construir_caso_temporal(
            seed=seed,
            bloque_entrenamiento=bloque["bloque_id"],
            train_pct_ini=bloque["pct_ini"],
            train_pct_fin=bloque["pct_fin"],
            etiqueta_bloque=bloque["etiqueta_bloque"],
            bloques_entrenamiento=[bloque["bloque_id"]],
            bloque_validacion=bloques_validacion[0]["bloque_id"],
            train_idx=train_idx,
            val_idx=val_idx,
        ))

    return casos


def construir_casos_acumulativos(dataset, random_state=42):
    """
    Construye casos de entrenamiento donde cada bloque acumula todos los anteriores.

    dataset: dict con arrays de evaluaciones del dataset.
    random_state: semilla base para el muestreo de validacion.
    """
    seed = int(np.asarray(dataset["seed"])[0])
    bloques = construir_bloques_por_eval_id(dataset)
    n_muestras_val_fija = calcular_n_muestras_validacion_fija(bloques)
    casos = []

    for idx_bloque, bloque_last in enumerate(bloques[:-1]):
        bloques_entrenamiento = bloques[: idx_bloque + 1]
        train_idx = np.concatenate([np.asarray(item["indices"], dtype=np.int64) for item in bloques_entrenamiento])
        if train_idx.size == 0:
            continue

        bloques_validacion = obtener_bloque_validacion_siguiente(bloques, idx_bloque)
        validacion_idx = np.concatenate([np.asarray(item["indices"], dtype=np.int64) for item in bloques_validacion])
        if validacion_idx.size == 0:
            continue

        val_idx = muestrear_validacion_siguiente(validacion_idx, n_muestras_val_fija, random_state=random_state, seed=seed, bloque_id=bloque_last["bloque_id"])

        casos.append(_construir_caso_temporal(
            seed=seed,
            bloque_entrenamiento=bloque_last["bloque_id"],
            train_pct_ini=bloques_entrenamiento[0]["pct_ini"],
            train_pct_fin=bloque_last["pct_fin"],
            etiqueta_bloque=_nombre_bloque(bloques_entrenamiento[0]["pct_ini"], bloque_last["pct_fin"]),
            bloques_entrenamiento=[item["bloque_id"] for item in bloques_entrenamiento],
            bloque_validacion=bloques_validacion[0]["bloque_id"],
            train_idx=train_idx,
            val_idx=val_idx,
        ))

    return casos
