# #### `rbf_factory.py`

# Debe construir siempre el mismo RBF seleccionado:

# ```text
# kernel = multiquadric
# epsilon = 1.0
# smoothing = 1e-3
# neighbors = 50
# degree = -1
# ```

# Así evitamos instanciar el modelo a mano en cada algoritmo.