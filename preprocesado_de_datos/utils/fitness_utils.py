"""
fitness_utils.py – Utilidades relacionadas con el fitness para el preprocesado
de datasets.

"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Asignación de intervalos/bins al rango de fitness
# ---------------------------------------------------------------------------


def construir_fases_relativas_por_seed(eval_id, seed_arr):
    eval_id = np.asarray(eval_id, dtype=np.int64).reshape(-1)
    seed_arr = np.asarray(seed_arr, dtype=np.int32).reshape(-1)
    if eval_id.size != seed_arr.size:
        raise ValueError("'eval_id' y 'seed' deben tener la misma longitud.")

    fases = np.full(eval_id.shape, -1, dtype=np.int32)
    limites = (
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
    )

    for seed in np.unique(seed_arr):
        mascara_seed = seed_arr == seed
        evals_seed = eval_id[mascara_seed].astype(float)
        if evals_seed.size == 0:
            continue

        eval_total = max(float(np.max(evals_seed)), 1.0)
        progreso = np.clip(evals_seed / eval_total, 0.0, 1.0)

        fases_seed = np.full(evals_seed.shape, -1, dtype=np.int32)
        for idx, (inicio, fin) in enumerate(limites):
            if idx < len(limites) - 1:
                mascara_fase = (progreso >= inicio) & (progreso < fin)
            else:
                mascara_fase = (progreso >= inicio) & (progreso <= fin)
            fases_seed[mascara_fase] = idx

        fases[mascara_seed] = fases_seed

    if np.any(fases < 0):
        raise ValueError("No se pudo asignar una fase valida a todas las muestras.")
    return fases

# asignar_bins_fitness discretiza el rango [fitness_min, fitness_max] en n_bins
# intervalos uniformes y asigna a cada muestra el ID del bin en el que cae.
#
# devuelve:
#   bin_ids: array de enteros (0..n_bins-1), uno por muestra
#   edges:   array de n_bins+1 bordes que delimitan los intervalos
#
# ejemplo con n_bins=5 y fitness entre 100 y 600:
#   edges  = [100, 200, 300, 400, 500, 600]
#   bin 0  = [100, 200)
#   bin 1  = [200, 300)
#   bin 2  = [300, 400)
#   bin 3  = [400, 500)
#   bin 4  = [500, 600]   ← el último incluye el extremo derecho

def asignar_bins_fitness(fitness, n_bins, tipo_bins="cuantiles"):
    fitness = np.asarray(fitness, dtype=float)
    fmin = float(np.min(fitness))
    fmax = float(np.max(fitness))

    # caso especial: todos los fitness son iguales → un solo bin
    if np.isclose(fmin, fmax):
        edges = np.array([fmin, fmax + 1e-12], dtype=float)
        bin_ids = np.zeros(len(fitness), dtype=np.int32)
        return bin_ids, edges

    if tipo_bins == "uniformes":
        # crear n_bins+1 bordes equidistantes entre fmin y fmax
        edges = np.linspace(fmin, fmax, n_bins + 1, dtype=float)
    elif tipo_bins == "cuantiles":
        # crear bordes basados en percentiles para que cada bin recoja la misma cantidad de poblacion
        edges = np.percentile(fitness, np.linspace(0, 100, n_bins + 1))
        # Asegurarte de que el maximo percentil alcance (puede haber problemas de float)
        edges[-1] = max(edges[-1], fmax + 1e-12)

        # si la distribucion tiene muchos valores repetidos (fitness identicos), los percentiles pueden colapsar.
        # entonces, se eliminan bordes duplicados que generen bins vacios
        
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([fmin, fmax + 1e-12])
    else:
        raise ValueError(f"tipo_bins desconocido: {tipo_bins}")

    # np.digitize asigna a cada valor el indice del intervalo en el que cae
    # se le pasan solo los bordes interiores (edges[1:-1]) para que devuelva
    # valores entre 0 y n_bins-1
    bin_ids = np.digitize(fitness, edges[1:-1], right=False).astype(np.int32)
    return bin_ids, edges


# ---------------------------------------------------------------------------
# 2. Contabilizar muestras por bin
# ---------------------------------------------------------------------------

# contabilizar_muestras_por_bin cuenta cuantas muestras hay en cada bin.
# recibe el array de bin_ids y el numero total de bins.
# devuelve un array de tamaño n_bins donde la posicion i indica cuantas
# muestras pertenecen al bin i.

def contabilizar_muestras_por_bin(bin_ids, n_bins):
    return np.bincount(bin_ids, minlength=n_bins)


# ---------------------------------------------------------------------------
# 3. Cálculo de cuotas por seed (estratificación)
# ---------------------------------------------------------------------------

# calcular_muestras_por_seed determina cuantas muestras tomar de cada seed
# dentro de un bin para que la seleccion sea proporcional y estratificada.
#
# la idea es sencilla:
#   - si un bin tiene 3 seeds con 100, 200 y 700 muestras respectivamente
#     y el tope_por_bin es 500, se reparten 500 muestras proporcionalmente:
#     seed_1 → 50, seed_2 → 100, seed_3 → 350
#   - se garantiza al menos 1 muestra por seed presente en el bin
#   - si sobran o faltan muestras por redondeo, se ajustan

def repartir_cuotas_proporcionales(counts, tope):
    counts = np.asarray(counts, dtype=int)
    if counts.size == 0:
        return np.empty((0,), dtype=int)

    # proporcion de cada categoria sobre el total
    proporciones = counts / counts.sum()

    # muestras iniciales redondeando hacia abajo
    muestras = np.floor(proporciones * tope).astype(int)

    # asegurar al menos 1 muestra por categoria presente
    muestras = np.where((counts > 0) & (muestras == 0), 1, muestras)

    total = int(muestras.sum())

    # ajustar si nos pasamos del tope (quitar de la categoria con mas cuota)
    while total > tope:
        i = int(np.argmax(muestras))
        if muestras[i] > 1:
            muestras[i] -= 1
            total -= 1
        else:
            break

    # ajustar si nos quedamos cortos (añadir a la categoria con mas margen)
    while total < tope:
        margen = counts - muestras
        i = int(np.argmax(margen))
        if margen[i] > 0:
            muestras[i] += 1
            total += 1
        else:
            break

    return muestras


def calcular_muestras_por_seed(seeds_del_bin, tope):
    unique_seeds, counts = np.unique(seeds_del_bin, return_counts=True)
    muestras = repartir_cuotas_proporcionales(counts, tope)
    return unique_seeds, muestras


# ---------------------------------------------------------------------------
# 4. Submuestreo estratificado por seed
# ---------------------------------------------------------------------------

# submuestreo_estratificado_por_seed selecciona 'tope' muestras de un bin
# de forma proporcional a la cantidad de muestras de cada seed.
# asi se evita que el balanceo favorezca unas seeds sobre otras.
#
# parametros:
#   idx_bin:   indices (del dataset global) de las muestras del bin
#   seed_arr:  array de seeds del dataset completo
#   tope:      numero maximo de muestras a seleccionar
#   rng:       generador aleatorio de numpy
#
# devuelve:
#   array de indices seleccionados

def submuestreo_estratificado_por_seed(idx_bin, seed_arr, tope, rng):
    seeds_bin = seed_arr[idx_bin]
    unique_seeds, muestras = calcular_muestras_por_seed(seeds_bin, tope)

    idx_sel = []
    for seed, muestra in zip(unique_seeds, muestras):
        # indices dentro del bin que pertenecen a esta seed
        idx_seed = idx_bin[seeds_bin == seed]

        if muestra >= idx_seed.size:
            # la seed tiene menos muestras que su cuota → se cogen todas
            idx_sel.append(idx_seed)
        else:
            # seleccion aleatoria sin reemplazo
            idx_sel.append(rng.choice(idx_seed, size=muestra, replace=False))

    return np.concatenate(idx_sel)


def submuestreo_estratificado_por_fase_y_seed(idx_bin, seed_arr, fase_arr, tope, rng):
    fases_bin = fase_arr[idx_bin]
    unique_fases, counts_fases = np.unique(fases_bin, return_counts=True)
    cuotas_fase = repartir_cuotas_proporcionales(counts_fases, tope)

    idx_sel = []
    for fase, cuota_fase in zip(unique_fases, cuotas_fase):
        idx_fase = idx_bin[fases_bin == fase]
        if cuota_fase >= idx_fase.size:
            idx_sel.append(idx_fase)
        else:
            idx_sel.append(submuestreo_estratificado_por_seed(idx_fase, seed_arr, int(cuota_fase), rng))

    return np.concatenate(idx_sel)


# ---------------------------------------------------------------------------
# 5. Balanceo a la baja (upsampling) estratificado por seed
# ---------------------------------------------------------------------------

# balanceo_a_la_baja es la funcion principal que orquesta todo el pipeline:
#
#   paso 1: asignar cada muestra a un bin de fitness
#   paso 2: contabilizar cuantas muestras hay por bin
#   paso 3: determinar el tope por bin = min(bin_menos_poblado, max_por_bin)
#   paso 4: para cada bin que supere el tope, submuestrear estratificado
#   paso 5: devolver los indices seleccionados
#
# parametros:
#   fitness:      array con el fitness de cada muestra
#   seed_arr:     array con la seed de cada muestra
#   n_bins:       numero de intervalos para discretizar el fitness
#   max_por_bin:  tope maximo de muestras por bin (evita perder demasiada
#                 info si el bin mas pequeño es muy pequeño)
#   random_state: semilla para reproducibilidad del muestreo aleatorio
#
# devuelve:
#   indices:  array de indices seleccionados del dataset original
#   bin_ids:  array con el bin asignado a cada muestra (del dataset completo)
#   edges:    bordes de los intervalos

def balanceo_a_la_baja(
    fitness,
    seed_arr,
    n_bins=10,
    max_por_bin=3000,
    tipo_bins="uniformes",
    random_state=42,
    fase_arr=None,
    estratificar_por_fase=False,
):
    rng = np.random.default_rng(random_state)

    if estratificar_por_fase:
        if fase_arr is None:
            raise ValueError("Si estratificar_por_fase=True, fase_arr no puede ser None.")
        fase_arr = np.asarray(fase_arr, dtype=np.int32)
        if len(fase_arr) != len(fitness):
            raise ValueError("fase_arr debe tener la misma longitud que fitness.")

    # paso 1: asignar bins
    bin_ids, edges = asignar_bins_fitness(fitness, n_bins, tipo_bins=tipo_bins)

    # paso 2: contabilizar
    conteo = contabilizar_muestras_por_bin(bin_ids, n_bins)

    # paso 3: aplicar max_por_bin bin a bin
    seleccion = []

    for b in range(n_bins):
        idx_bin = np.flatnonzero(bin_ids == b)

        if idx_bin.size == 0:
            continue

        # el numero real de muestras a extraer de este bin:
        muestras_bin = min(max_por_bin, idx_bin.size)

        # si nos podemos quedar con todas, evitamos el sobrecoste del muestreo estratificado
        if muestras_bin == idx_bin.size:
            seleccion.append(idx_bin)
        else:
            # paso 4: submuestreo estratificado
            if estratificar_por_fase:
                seleccion.append(
                    submuestreo_estratificado_por_fase_y_seed(idx_bin, seed_arr, fase_arr, muestras_bin, rng)
                )
            else:
                seleccion.append(submuestreo_estratificado_por_seed(idx_bin, seed_arr, muestras_bin, rng))

    if not seleccion:
        raise ValueError("No se seleccionó ninguna muestra")

    # paso 5: juntar todos los indices y mezclar
    indices = np.concatenate(seleccion)
    rng.shuffle(indices)

    return indices.astype(np.int64), bin_ids, edges
