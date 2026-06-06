import numpy as np

def split_por_run_aleatorio(dataset, random_state=42, train_ratio=0.7, max_seeds=None):
    seeds = np.asarray(dataset["seed"], dtype=int)
    seeds_unicas = sorted(int(s) for s in np.unique(seeds))
    if max_seeds is not None:
        if max_seeds < 1:
            raise ValueError("max_seeds debe ser >= 1.")
        seeds_unicas = seeds_unicas[:max_seeds]
    rng = np.random.default_rng(random_state)

    splits = []
    for fold_idx, seed in enumerate(seeds_unicas):
        idx_seed = np.flatnonzero(seeds == seed)
        if idx_seed.size < 2:
            continue

        idx_seed = rng.permutation(idx_seed)
        corte = int(idx_seed.size * train_ratio)
        corte = max(1, min(corte, idx_seed.size - 1))

        train_idx = np.sort(idx_seed[:corte])
        test_idx = np.sort(idx_seed[corte:])
        splits.append(
            {
                "seed": int(seed),
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )

    if not splits:
        raise ValueError("No se pudo construir ningun split intrarun aleatorio valido.")

    return splits
