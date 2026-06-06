import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURACIÓN DE ESTILO ACADÉMICO (MINIMALISTA)
# =====================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "axes.edgecolor": "#444444",   # Suavizado de bordes
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444"
})

# =====================================================================
# DEFINICIÓN DE PAISAJE CON ÓPTIMO LOCAL Y GLOBAL (GAUSSIANAS)
# =====================================================================
x = np.linspace(0, 7.5, 1000)

def f(x):
    # Genera dos valles limpios y una penalización cuadrática suave en los extremos
    return -2 * np.exp(-(x - 2)**2) - 4 * np.exp(-(x - 5)**2) + 0.05 * x**2

y = f(x)

# Coordenadas exactas aproximadas de los fondos de los valles
x_local_min = 2.0
x_global_min = 5.0

# =====================================================================
# INDIVIDUOS / AGENTES SIMULADOS
# =====================================================================
# Exploración: Muestreo disperso por el espacio
x_explor = np.array([0.6, 1.6, 2.4, 3.5, 6.4, 7.1])
y_explor = f(x_explor)

# Explotación: Convergiendo en masa hacia el óptimo global
x_explot = np.array([4.5, 4.7, 4.9, 5.1, 5.3, 5.5])
y_explot = f(x_explot)

# =====================================================================
# CONSTRUCCIÓN DE LA FIGURA
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))

# Dibujar superficie de la función objetivo
ax.plot(x, y, color="#4b5563", lw=2.5, label="Función objetivo $f(\\mathbf{x})$", zorder=1)

# Dibujar las poblaciones
ax.scatter(x_explor, y_explor, color="#e66101", edgecolor="#b34000", 
           s=60, marker="o", zorder=3, label="Exploración (Muestreo global)")
ax.scatter(x_explot, y_explot, color="#5e3c99", edgecolor="#3d2173", 
           s=60, marker="^", zorder=3, label="Explotación (Refinamiento local)")

# =====================================================================
# CARTELAS DE TEXTO Y ANOTACIONES DE OPTIMALIDAD
# =====================================================================
ax.annotate("Óptimo Local\n(Atractor)", xy=(x_local_min, f(x_local_min)), xytext=(1.2, f(x_local_min) + 1.8),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2),
            ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#f3f4f6", ec="#d1d5db", lw=0.8))

ax.annotate("Óptimo Global\n(Solución ideal)", xy=(x_global_min, f(x_global_min)), xytext=(5.0, f(x_global_min) + 1.8),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2),
            ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#f3f4f6", ec="#d1d5db", lw=0.8))

# =====================================================================
# DINÁMICA DE LAS FLECHAS (ORIENTACIÓN CORRECTA)
# =====================================================================
# Flechas de Explotación: apuntan estrictamente HACIA ABAJO en las laderas del óptimo global
ax.annotate("", xy=(4.8, f(4.8)), xytext=(4.2, f(4.2)),
            arrowprops=dict(arrowstyle="->", color="#5e3c99", lw=2))
ax.annotate("", xy=(5.2, f(5.2)), xytext=(5.8, f(5.8)),
            arrowprops=dict(arrowstyle="->", color="#5e3c99", lw=2))

# Flecha de Exploración: muestra un salto dinámico escapando del óptimo local hacia zonas abiertas
ax.annotate("Salto global\n(Escape)", xy=(4.0, f(4.0) + 0.5), xytext=(1.8, f(1.8) + 0.5),
            arrowprops=dict(arrowstyle="->", color="#e66101", lw=1.5, ls="--", connectionstyle="arc3,rad=-0.2"),
            ha='center', color="#b34000", fontsize=9)

# =====================================================================
# ESTILIZACIÓN FINAL
# =====================================================================
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Espacio de soluciones ($S$)")
ax.set_ylabel("Valor de la función objetivo ($f(\\mathbf{x})$)")

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=True,
    facecolor="white",
    edgecolor="#e5e7eb",
    framealpha=0.95,
)

fig.tight_layout(rect=(0, 0, 1, 0.86))

# Guardado doble (PNG para previsualizar, PDF vectorial sin pérdida para incluir en LaTeX)
plt.savefig("./dinamicas_exploracion_explotacion.png", bbox_inches="tight", dpi=300)
plt.savefig("./dinamicas_exploracion_explotacion.pdf", bbox_inches="tight", dpi=300)

print("¡Gráfico conceptual generado con éxito!")
