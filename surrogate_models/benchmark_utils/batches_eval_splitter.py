import hashlib

import numpy as np

N_BATCHES = 5
PCT_POR_BATCH = 100 // N_BATCHES
VAL_RATIO_TRAIN = 0.25
KEY_N_BATCHES_UTILES = "_n_batches_utiles"
KEY_CONVERGENCIA_DETALLE = "_convergencia_detalle"
TOL_MEJORA_BATCH_ABS = 1e-8
TOL_MEJORA_BATCH_REL = 1e-4


def detectar_ultimo_batch_informativo(
    fitness,
    n_batches=N_BATCHES,
    *,
    tol_abs=TOL_MEJORA_BATCH_ABS,
    tol_rel=TOL_MEJORA_BATCH_REL,
    devolver_detalle=False,
):
    fitness = np.asarray(fitness, dtype=float)
    n = len(fitness)
    if n == 0:
        if devolver_detalle:
            return 0, []
        return 0

    best_so_far = np.minimum.accumulate(fitness)
    ventanas = np.array_split(np.arange(n), n_batches)
    ultimo_batch = 0
    detalle = []

    for batch_id, ventana in enumerate(ventanas, 1):
        if len(ventana) == 0:
            continue
        best_inicio = float(best_so_far[int(ventana[0])])
        best_fin = float(best_so_far[int(ventana[-1])])
        mejora = float(best_inicio - best_fin)
        tolerancia = float(max(tol_abs, tol_rel * max(1.0, abs(best_inicio))))
        informativo = bool(mejora > tolerancia)
        if informativo:
            ultimo_batch = int(batch_id)
        detalle.append(
            {
                "batch": int(batch_id),
                "best_inicio": best_inicio,
                "best_fin": best_fin,
                "mejora": mejora,
                "tolerancia": tolerancia,
                "informativo": informativo,
            }
        )

    if devolver_detalle:
        return ultimo_batch, detalle
    return ultimo_batch


def truncar_por_convergencia(dataset, n_batches=N_BATCHES):
    fitness = np.asarray(dataset["fitness"], dtype=float)
    n = len(fitness)

    ultimo_batch, detalle = detectar_ultimo_batch_informativo(
        fitness,
        n_batches,
        devolver_detalle=True,
    )
    if ultimo_batch < 1:
        ventanas = np.array_split(np.arange(n), n_batches)
        idx_corte = int(ventanas[0][-1]) + 1 if ventanas and len(ventanas[0]) else 0
        dataset_truncado = {
            k: np.asarray(v)[:idx_corte]
            for k, v in dataset.items()
            if not str(k).startswith("_")
        }
        dataset_truncado[KEY_N_BATCHES_UTILES] = 1
        dataset_truncado[KEY_CONVERGENCIA_DETALLE] = detalle
        return dataset_truncado, 0, idx_corte / n if n else 0.0

    ventanas = np.array_split(np.arange(n), n_batches)
    idx_corte = int(ventanas[ultimo_batch - 1][-1]) + 1  # extremo exclusivo

    if idx_corte >= n:
        dataset[KEY_CONVERGENCIA_DETALLE] = detalle
        return dataset, ultimo_batch, 1.0

    dataset_truncado = {
        k: np.asarray(v)[:idx_corte]
        for k, v in dataset.items()
        if not str(k).startswith("_")
    }
    dataset_truncado[KEY_N_BATCHES_UTILES] = int(ultimo_batch)
    dataset_truncado[KEY_CONVERGENCIA_DETALLE] = detalle
    return dataset_truncado, ultimo_batch, idx_corte / n
VALIDATION_SCOPE_CHOICES = {"all", "next"}

def _rango_porcentual(batch_id):
    inicio = 1 if batch_id == 1 else ((batch_id - 1) * PCT_POR_BATCH) + 1
    fin = batch_id * PCT_POR_BATCH
    return int(inicio), int(fin)


def _nombre_batch(pct_ini, pct_fin):
    return f"{int(pct_ini)}-{int(pct_fin)}%"


def construir_batches_por_eval_id(dataset):
    eval_id = np.asarray(dataset["eval_id"])
    n_batches_utiles = int(dataset.get(KEY_N_BATCHES_UTILES, N_BATCHES))
    if n_batches_utiles < 1:
        raise ValueError("El numero de batches utiles debe ser >= 1.")

    # Los datasets por seed ya vienen en orden temporal de evaluacion.
    indices_orden_temporal = np.arange(eval_id.shape[0], dtype=np.int64)
    trozos = np.array_split(indices_orden_temporal, n_batches_utiles)

    batches = []
    for batch_id, idx in enumerate(trozos, 1):
        pct_ini, pct_fin = _rango_porcentual(batch_id)
        batches.append(
            {
                "batch_id": int(batch_id),
                "pct_ini": int(pct_ini),
                "pct_fin": int(pct_fin),
                "batch_label": _nombre_batch(pct_ini, pct_fin),
                "indices": np.asarray(idx, dtype=np.int64),
                "eval_id_min": int(eval_id[idx].min()),
                "eval_id_max": int(eval_id[idx].max()),
            }
        )

    return batches


def calcular_n_muestras_validacion_fija(batches):
    if not batches:
        raise ValueError("No hay batches para calcular el tamaño fijo de validación.")
    n_train_base = int(np.asarray(batches[0]["indices"], dtype=np.int64).size)
    n_muestras_val = int(np.floor(n_train_base * float(VAL_RATIO_TRAIN)))
    if n_muestras_val < 1:
        raise ValueError("El tamaño fijo de validación debe ser >= 1.")
    return n_muestras_val


def muestrear_validacion_futura(futuros_idx, n_muestras_val, random_state, seed, batch_id):
    futuros_idx = np.asarray(futuros_idx, dtype=np.int64)

    if futuros_idx.size < n_muestras_val:
        raise ValueError(
            f"No hay suficientes muestras futuras para validación: "
            f"future={futuros_idx.size}, required={n_muestras_val}"
        )

    rng = np.random.default_rng([int(random_state), int(seed), int(batch_id)])
    seleccion = rng.choice(futuros_idx, size=n_muestras_val, replace=False)
    return np.sort(np.asarray(seleccion, dtype=np.int64))


def obtener_batches_futuros(batches, idx_batch, validation_scope="all"):
    scope = str(validation_scope).strip().lower()
    if scope not in VALIDATION_SCOPE_CHOICES:
        raise ValueError(
            f"validation_scope no valido: {validation_scope}. "
            f"Valores permitidos: {sorted(VALIDATION_SCOPE_CHOICES)}"
        )

    futuros = list(batches[idx_batch + 1 :])
    if scope == "next":
        return futuros[:1]
    return futuros


def _construir_caso_temporal(
    *,
    seed,
    batch_train,
    batch_train_last,
    train_pct_ini,
    train_pct_fin,
    batch_label,
    batches_train,
    batches_futuros,
    train_idx,
    val_idx,
    eval_id,
):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    return {
        "seed": int(seed),
        "batch_train": int(batch_train),
        "batch_train_last": int(batch_train_last),
        "train_pct_ini": int(train_pct_ini),
        "train_pct_fin": int(train_pct_fin),
        "batch_label": str(batch_label),
        "batches_train": [int(item) for item in batches_train],
        "batches_futuros": [int(item) for item in batches_futuros],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_idx_hash": hash_indices(train_idx),
        "val_idx_hash": hash_indices(val_idx),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "train_eval_id_min": int(eval_id[train_idx].min()),
        "train_eval_id_max": int(eval_id[train_idx].max()),
        "val_eval_id_min": int(eval_id[val_idx].min()),
        "val_eval_id_max": int(eval_id[val_idx].max()),
    }


def hash_indices(indices):
    indices = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(indices.tobytes()).hexdigest()[:16]


def construir_casos_no_acumulativos(dataset, random_state=42, validation_scope="all"):
    seed = int(np.asarray(dataset["seed"])[0])
    eval_id = np.asarray(dataset["eval_id"])
    batches = construir_batches_por_eval_id(dataset)
    n_muestras_val_fija = calcular_n_muestras_validacion_fija(batches)
    casos = []

    for idx_batch, batch in enumerate(batches[:-1]):
        train_idx = np.asarray(batch["indices"], dtype=np.int64)
        if train_idx.size == 0:
            continue

        batches_futuros = obtener_batches_futuros(
            batches,
            idx_batch,
            validation_scope=validation_scope,
        )
        futuros_idx = np.concatenate([np.asarray(item["indices"], dtype=np.int64) for item in batches_futuros])
        if futuros_idx.size == 0:
            continue

        val_idx = muestrear_validacion_futura(
            futuros_idx,
            n_muestras_val_fija,
            random_state=random_state,
            seed=seed,
            batch_id=batch["batch_id"],
        )

        casos.append(
            _construir_caso_temporal(
                seed=seed,
                batch_train=batch["batch_id"],
                batch_train_last=batch["batch_id"],
                train_pct_ini=batch["pct_ini"],
                train_pct_fin=batch["pct_fin"],
                batch_label=batch["batch_label"],
                batches_train=[batch["batch_id"]],
                batches_futuros=[item["batch_id"] for item in batches_futuros],
                train_idx=train_idx,
                val_idx=val_idx,
                eval_id=eval_id,
            )
        )

    return casos


def construir_casos_acumulativos(dataset, random_state=42, validation_scope="all"):
    seed = int(np.asarray(dataset["seed"])[0])
    eval_id = np.asarray(dataset["eval_id"])
    batches = construir_batches_por_eval_id(dataset)
    n_muestras_val_fija = calcular_n_muestras_validacion_fija(batches)
    casos = []

    for idx_batch, batch_last in enumerate(batches[:-1]):
        batches_train = batches[: idx_batch + 1]
        train_idx = np.concatenate(
            [np.asarray(item["indices"], dtype=np.int64) for item in batches_train]
        )
        if train_idx.size == 0:
            continue

        batches_futuros = obtener_batches_futuros(
            batches,
            idx_batch,
            validation_scope=validation_scope,
        )
        futuros_idx = np.concatenate(
            [np.asarray(item["indices"], dtype=np.int64) for item in batches_futuros]
        )
        if futuros_idx.size == 0:
            continue

        val_idx = muestrear_validacion_futura(
            futuros_idx,
            n_muestras_val_fija,
            random_state=random_state,
            seed=seed,
            batch_id=batch_last["batch_id"],
        )

        casos.append(
            _construir_caso_temporal(
                seed=seed,
                batch_train=batch_last["batch_id"],
                batch_train_last=batch_last["batch_id"],
                train_pct_ini=batches_train[0]["pct_ini"],
                train_pct_fin=batch_last["pct_fin"],
                batch_label=_nombre_batch(batches_train[0]["pct_ini"], batch_last["pct_fin"]),
                batches_train=[item["batch_id"] for item in batches_train],
                batches_futuros=[item["batch_id"] for item in batches_futuros],
                train_idx=train_idx,
                val_idx=val_idx,
                eval_id=eval_id,
            )
        )

    return casos
