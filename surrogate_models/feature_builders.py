# MUY IMPORTANTE
# NO DEBE ESTAR MEZCLADO CON EL MODELO.
# Para el flujo continuo (CEC2017) solo se construyen features con x y, en su
# caso, con diversidad euclídea.

import numpy as np

def _obtener_diversidad(dataset):
    if "div_dist_euclidea" in dataset:
        return np.asarray(dataset["div_dist_euclidea"], dtype=float).reshape(-1, 1)
    raise ValueError("El dataset no contiene una metrica de diversidad compatible.")

def construir_features(dataset, feature_mode):
    if feature_mode == "x":
        return np.asarray(dataset["x"], dtype=float)

    if feature_mode == "x_div":
        x = np.asarray(dataset["x"], dtype=float)
        div = _obtener_diversidad(dataset)
        return np.hstack([x, div])

    raise ValueError(f"Feature mode no valido: {feature_mode}")
