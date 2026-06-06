hola. soy un estudiante de ingeniería informática en la ugr, universidad de granada. me encuentro en el ultimo cuatrimestre del ultimo año de carrera, y por tanto, tengo que realizar el trabajo de fin de grado (tfg). en concreto, estoy especializado en la rama de ciencias de la computacion e inteligencia artificial, donde se ha estudiado temario relacionado con el aprendizaje profundo y su tecnicas (mls), metaheurísticas, predicciones, etc...

por tanto, el titulo escogido para el tfg es: "Utilización de Técnicas de Machine Learning para la Hibridación de Metaheurísticas"; y la descripción del proyecto presentada se corresponde con:
"Las metaheurísticas son técnicas de optimización utilizadas especialmente para la resolución de problemas reales gracias a su capacidad de trabajar eficientemente con grandes espacios de búsqueda, aunque en ocasiones con un coste computacional elevado. En los últimos años, el Aprendizaje Automático, y en concreto, técnicas como el Deep Learning, continúan demostrando su efectividad como herramientas de predicción y aproximación con grandes datasets. Por ello, su combinación puede ser un enfoque de gran interés para la mejora de dichos algoritmos de optimización.

Este Trabajo de Fin de Grado plantea la hibridación de una metaheurística con técnicas de Aprendizaje Automático, con el objetivo de valorar las mejoras sobre la eficiencia y calidad de las soluciones obtenidas mediante modelos híbridos. Para ello, se utilizarán modelos de aprendizaje automático aplicados a niveles distintos de la metaheurística:

- Para aproximar la función de fitness dentro de la heurística y reducir así su coste computacional, como técnicas de evaluación subrogadas. Los modelos usarán para aprender las evaluaciones anteriores durante el proceso de optimización.
- Para optimizar soluciones mediante redes neuronales. Estas redes aprenderán a partir de las mejoras obtenidos por componentes como la búsqueda local, y podrán de forma alternativa reemplazarla. De esta manera, se reducirá el alto coste en evaluaciones que implica el uso de una búsqueda local dentro de las metaheurísticas.

La combinación de ambas estrategias permitirá el diseño de modelos híbridos con un menor coste en evaluaciones, que podrá redundar en unos mejores resultados. Además, también nos permitirá estudiar las ventajas y limitaciones de las distintas técnicas de Aprendizaje Automático para la construcción de modelos híbridos, comparándolos entre sí, y con respecto a otras técnicas subrogadas más clásicas".

hoy mismo tuve la primera reunion con el tutor para asentar las bases del proyecto asi como un planning para poder llevarlo a cabo. por tanto, necesito que te comportes como si fueses un programador de problemas y logica altamente prestigioso, fundamentado y con un conocimiento completo a cerca de las metaheurísticas y el machine learning. ademas, y para que lo tengas mas claro, el tutor me ha guiado en las herramientas que van a ser utilizadas:

- python (como lenguaje de programacion a utilizar)
- librerias:
  - scikitlearn (técnicas de machine learning, donde, entre las recomendadas por el profesor, se encuentran las bayesianas y las neural networks, donde la complejidad de estas ultimas no va a ser muy grande ya que van a ser entrenadas de manera online durante el proceso iterativo del algoritmo utilizado como metaheuristica a partir de las buenas y malas soluciones obtenidas)
  - pyade (https://github.com/xKuZz/pyade), que será utilizada ya que cuenta directamente con la implementación de algoritmos DE
  - en un principio, solo se han mencionado estas.

en cuanto a los papers que pueden ser utiles revisar y extraer información de ayuda, aunque algo genéricos se encuentran:

- sobre la parte de uso de redes neuronales y técnicas de machine learning como surrogadas para un componente evolutivo:
  - [1] Y. Jin, «Surrogate-assisted evolutionary computation: Recent
    advances and future challenges», Swarm and Evolutionary Computation,
    vol. 1, n.º 2, pp. 61-70, jun. 2011, doi: 10.1016/j.swevo.2011.05.001.
  - [2] T . Janus, A. Lüubbers, y S. Engell, «Neural Networks for
    Surrogate-assisted Evolutionary optimization of Chemical Processes», en
    2020 IEEE Congress on Evolutionary Computation (CEC), jul. 2020, pp.
    1-8. doi: 10.1109/CEC48606.2020.9185781.

- sobre la parte Búsqueda Local hay menos bibliografía porque es más
  diferente, pero una referencia podría ser: - [3] M. S. A. Sakhri, A. Goëffon, O. Goudet, F. Saubion, y C. Touhami,
  «Discovering new robust local search algorithms with neuro-evolution»,
  12 de marzo de 2025, arXiv: arXiv:2501.04747. doi:
  10.48550/arXiv.2501.04747.

una vez aclarado el contenido. el tutor ha hecho incapie en comenzar por la primera parte del proyecto y dejar, por ahora, la busqueda local y sus mejoras a un lado. esto se debe a que, si realmente se obtienen resultados bastantes buenos (ya que existe la posibilidad de que no sea así al tratarse de un tfg de investigación) y un gran abanico de pruebas y demostraciones, este temario podría extenderse más para obviar la parte de la búsqueda local, aunque, por ahora, vamos a ceñirnos a lo dicho en la reunión, que será indicado ahora.

dicho esto, asienta toda esta información y posteriormente continuaré indicandote las pautas que vamos a seguir para la primera parte.

---

okei. una vez entendido, comencemos con la primera parte del proyecto, y mas importante ahora mismo. esta se corresponde con:
....
se utilizarán modelos de aprendizaje automático aplicados a niveles distintos de la metaheurística:

- Para aproximar la función de fitness dentro de la heurística y reducir así su coste computacional, como técnicas de evaluación subrogadas. Los modelos usarán para aprender las evaluaciones anteriores durante el proceso de optimización.
  ....

los algoritmos que se van a utilizar van a ser DOS CONCRETAMENTE:

- AGE (algoritmo genético estacionario)
- DE (differential evolution)

para cada uno de ellos, queremos evaluar, en primer lugar, su rendimiento como METAHEURÍSTICA AL 100%, es decir, sin técnicas surrogadas de por medio, sobre un problema continuo (CEC2017 Benchmark: https://github.com/dmolina/cec2017real) y un problema combinatorio, el cual deberías ayudarme a escoger. en este último caso, el tutor no ha logrado caer en ningun problema combinatorio que realmente sea costoso ya que uno de los objetivos de este proyecto es lograr reducir el coste computacional que supondría resolver un problema costoso si no se utilizasen predicciones y aproximaciones de fitness. por tanto, debes ayudarme a encontrar un problema combinatorio que sea costoso computacionalmente, pero a la vez, algo conocido, para que sea facil la adaptación de las dos metaheurísticas al problema. en concreto, el ha indicado como posibles aquellos problemas de grafos que son bastante costosos (creo haber escuchado que NO FUESE DETERMINISTA ya que sino sería un "lio", pero no estoy seguro).

ademas, me ha indicado que en la implementación de las metaheurísticas, aunque para una de ellas (DE) ya tenemos acceso mediante la libreria PYADE, estas deberan contar con una función "eval" que será la encargada de realizar todo el procedimiento y donde posteriormente, estas serán modifcadas para trabajar con el porcentaje de iteraciones y determinar cuando aplicar las surrogadas, etc...

dicho esto, comencemos.








---------

voy a comenzar con la primerea funcion, f1. para ella, ya he generado las disitntas runs y una serie de plots, entre ellos, los plots de preprocesado (histogramas). te los adjunto junto al veredicto obtenido por el siguiente script, que a partir de valores numericos, estima que valor de max_por_bin seria el mas recomendado:

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FASES = (
    (0.0, 0.2, "0-20"),
    (0.2, 0.4, "20-40"),
    (0.4, 0.6, "40-60"),
    (0.6, 0.8, "60-80"),
    (0.8, 1.0, "80-100"),
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analiza visualmente un experimento a partir de los datasets por run y propone una orientacion para max_por_bin."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Lista de rutas a dataset_*.npz de un mismo algoritmo/problema.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Directorio raiz de un experimento benchmark. Si se indica, busca automaticamente los dataset_*.npz del algoritmo pedido.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta opcional para guardar el resumen en JSON.",
    )
    parser.add_argument(
        "--algoritmo",
        default="desconocido",
        choices=["age", "de", "desconocido"],
        help="Algoritmo analizado. Si usas --experiment-dir, sirve tambien para localizar los datasets.",
    )
    parser.add_argument(
        "--funcion",
        default=None,
        help=(
            "Funcion CEC a analizar cuando se usa --experiment-dir. "
            "Acepta formatos como 'f1' o '1'. Si no se indica, se incluyen todas las funciones."
        ),
    )
    return parser.parse_args()

def normalizar_funcion(funcion):
    if funcion is None:
        return None
    txt = str(funcion).strip().lower()
    if not txt:
        return None
    if txt.startswith("f"):
        txt = txt[1:]
    if not txt.isdigit():
        raise ValueError("--funcion debe ser un entero o un identificador tipo f1.")
    return f"f{int(txt)}"

def resolver_inputs(inputs, experiment_dir, algoritmo, funcion=None):
    if inputs:
        return inputs

    if not experiment_dir:
        raise ValueError("Debes indicar --inputs o --experiment-dir.")
    if algoritmo not in {"age", "de"}:
        raise ValueError("Si usas --experiment-dir, debes indicar --algoritmo age o de.")

    base = Path(experiment_dir)
    if not base.exists():
        raise FileNotFoundError(f"No existe el directorio de experimento: {base}")

    funcion_norm = normalizar_funcion(funcion)

    patrones = [
        f"metricas_runs/qap/{algoritmo}/*/dataset_{algoritmo}_qap_*.npz",
    ]
    if funcion_norm is None:
        patrones.append(f"f*/metricas_runs/cec2017/{algoritmo}/*/dataset_{algoritmo}_cec2017_*.npz")
    else:
        patrones.append(
            f"{funcion_norm}/metricas_runs/cec2017/{algoritmo}/*/dataset_{algoritmo}_cec2017_*.npz"
        )

    rutas = []
    for patron in patrones:
        rutas.extend(sorted(base.glob(patron)))

    if len(rutas) == 0:
        raise FileNotFoundError(
            f"No se encontraron dataset_*.npz para algoritmo={algoritmo} dentro de {base}"
        )
    return [str(r) for r in rutas]

def asignar_fases_relativas(eval_ids: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    fases = np.full(eval_ids.shape, -1, dtype=np.int32)
    for seed in np.unique(seeds):
        mascara_seed = seeds == seed
        evals_seed = eval_ids[mascara_seed].astype(float)
        if evals_seed.size == 0:
            continue
        total = max(float(np.max(evals_seed)), 1.0)
        progreso = np.clip(evals_seed / total, 0.0, 1.0)
        fases_seed = np.full(evals_seed.shape, -1, dtype=np.int32)
        for idx, (inicio, fin, _etq) in enumerate(FASES):
            if idx < len(FASES) - 1:
                mascara_fase = (progreso >= inicio) & (progreso < fin)
            else:
                mascara_fase = (progreso >= inicio) & (progreso <= fin)
            fases_seed[mascara_fase] = idx
        fases[mascara_seed] = fases_seed
    if np.any(fases < 0):
        raise ValueError("No se pudo asignar fase a todas las muestras.")
    return fases

def cargar_experimento(rutas_npz: list[str]):
    fitness_runs = []
    evals_runs = []
    seeds_runs = []
    for ruta in sorted(Path(p) for p in rutas_npz):
        with np.load(ruta, allow_pickle=True) as data:
            if "fitness" not in data or "eval_id" not in data:
                raise ValueError(f"{ruta} no contiene 'fitness' y 'eval_id'.")
            fitness = np.asarray(data["fitness"], dtype=float).reshape(-1)
            eval_id = np.asarray(data["eval_id"], dtype=int).reshape(-1)
        if fitness.size != eval_id.size:
            raise ValueError(f"{ruta} tiene longitudes inconsistentes entre fitness y eval_id.")
        mascara = np.isfinite(fitness)
        fitness = fitness[mascara]
        eval_id = eval_id[mascara]
        seed = int(ruta.parent.name.rsplit("_s", 1)[1])
        fitness_runs.append(fitness)
        evals_runs.append(eval_id)
        seeds_runs.append(np.full(fitness.shape, seed, dtype=np.int32))
    fitness = np.concatenate(fitness_runs)
    evals = np.concatenate(evals_runs)
    seeds = np.concatenate(seeds_runs)
    fases = asignar_fases_relativas(evals, seeds)
    return fitness, evals, seeds, fases

def hist_counts(valores: np.ndarray, edges: np.ndarray):
    counts, _ = np.histogram(valores, bins=edges)
    total = max(int(counts.sum()), 1)
    return counts, counts / total

def analizar(rutas_npz: list[str], algoritmo: str):
    fitness, evals, seeds, fases = cargar_experimento(rutas_npz)

    p005, p995 = np.percentile(fitness, [0.5, 99.5])
    vis = fitness[(fitness >= p005) & (fitness <= p995)]
    n_bins_hist = 60
    edges = np.linspace(p005, p995, n_bins_hist + 1)
    counts_global, shares_global = hist_counts(vis, edges)

    global_stats = {
        "visual_range_p0.5_p99.5": [float(p005), float(p995)],
        "max_bin_share": float(np.max(shares_global)),
        "top3_bins_share": float(np.sort(shares_global)[-3:].sum()),
        "occupied_bins_ratio": float(np.mean(counts_global > 0)),
        "width_p95_p5": float(np.percentile(fitness, 95) - np.percentile(fitness, 5)),
        "tail_low_p5_share": float(np.mean(fitness < np.percentile(fitness, 5))),
        "tail_high_p95_share": float(np.mean(fitness > np.percentile(fitness, 95))),
    }

    phase_stats = []
    for idx, (_inicio, _fin, etiqueta) in enumerate(FASES):
        arr = fitness[fases == idx]
        arr_vis = arr[(arr >= p005) & (arr <= p995)]
        counts_fase, shares_fase = hist_counts(arr_vis, edges)
        phase_stats.append(
            {
                "phase": etiqueta,
                "n": int(arr.size),
                "width_p95_p5": float(np.percentile(arr, 95) - np.percentile(arr, 5)),
                "max_bin_share": float(np.max(shares_fase)) if arr_vis.size > 0 else 0.0,
                "occupied_bins_ratio": float(np.mean(counts_fase > 0)) if arr_vis.size > 0 else 0.0,
            }
        )

    width_ratio = phase_stats[-1]["width_p95_p5"] / max(phase_stats[0]["width_p95_p5"], 1e-12)
    peak_ratio = phase_stats[-1]["max_bin_share"] / max(phase_stats[0]["max_bin_share"], 1e-12)

    booleans = {
        "pico_global_alto": global_stats["max_bin_share"] > 0.08,
        "concentracion_global_fuerte_top3": global_stats["top3_bins_share"] > 0.20,
        "fases_finales_colapsadas": width_ratio < 0.35,
        "pico_tardio_mas_agudo": peak_ratio > 1.6,
        "distribucion_global_extensa": global_stats["occupied_bins_ratio"] > 0.65,
    }

    if booleans["pico_global_alto"] and booleans["fases_finales_colapsadas"]:
        recomendacion = "bajar mucho desde 3000"
        sugerido = 1000
        comentario = (
            "La masa del experimento esta fuertemente concentrada y las fases finales colapsan sobre una zona estrecha. "
            "Mantener 3000 probablemente conserva demasiada redundancia. Tiene sentido probar un recorte agresivo."
        )
    elif booleans["concentracion_global_fuerte_top3"] or (
        booleans["fases_finales_colapsadas"] and booleans["pico_tardio_mas_agudo"]
    ):
        recomendacion = "bajar moderadamente desde 3000"
        sugerido = 1500
        comentario = (
            "Hay señales claras de concentracion o de colapso progresivo en fases finales, pero no un dominio tan extremo de la masa global. "
            "Conviene recortar, aunque no de forma tan agresiva."
        )
    else:
        recomendacion = "mantener cerca de 3000"
        sugerido = 3000
        comentario = (
            "La distribucion global no esta dominada por unos pocos picos y la estructura por fases no sugiere una sobresaturacion extrema. "
            "No hay evidencia fuerte para bajar de forma agresiva el max_por_bin."
        )

    return {
        "algoritmo": algoritmo,
        "n_runs": int(len(np.unique(seeds))),
        "n_samples": int(fitness.size),
        "global": global_stats,
        "phases": phase_stats,
        "comparativa_fases": {
            "width_ratio_80_100_vs_0_20": float(width_ratio),
            "peak_ratio_80_100_vs_0_20": float(peak_ratio),
        },
        "booleans": booleans,
        "recomendacion_visual": recomendacion,
        "max_por_bin_sugerido": int(sugerido),
        "comentario_salida": comentario,
    }

def main():
    args = parse_args()
    rutas_npz = resolver_inputs(args.inputs, args.experiment_dir, args.algoritmo, args.funcion)
    resumen = analizar(rutas_npz, args.algoritmo)
    resumen["n_inputs"] = int(len(rutas_npz))
    resumen["source"] = {
        "experiment_dir": str(args.experiment_dir) if args.experiment_dir else None,
        "funcion": normalizar_funcion(args.funcion),
        "inputs_resueltos": rutas_npz,
    }
    if args.out:
        ruta_out = Path(args.out)
        ruta_out.parent.mkdir(parents=True, exist_ok=True)
        ruta_out.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

con esto, te adjunto los jsons obtenidos para age y de:

{
  "algoritmo": "age",
  "n_runs": 51,
  "n_samples": 1530000,
  "global": {
    "visual_range_p0.5_p99.5": [
      118.3154061883163,
      31041557928.01257
    ],
    "max_bin_share": 0.7507018525212601,
    "top3_bins_share": 0.7917668213732563,
    "occupied_bins_ratio": 1.0,
    "width_p95_p5": 15103003074.505672,
    "tail_low_p5_share": 0.049981045751633986,
    "tail_high_p95_share": 0.05
  },
  "phases": [
    {
      "phase": "0-20",
      "n": 305949,
      "width_p95_p5": 27024049465.694557,
      "max_bin_share": 0.08740223735245509,
      "occupied_bins_ratio": 1.0
    },
    {
      "phase": "20-40",
      "n": 306000,
      "width_p95_p5": 5609996639.651901,
      "max_bin_share": 0.6665294117647059,
      "occupied_bins_ratio": 0.45
    },
    {
      "phase": "40-60",
      "n": 306000,
      "width_p95_p5": 1324749.0950498586,
      "max_bin_share": 0.9880261437908496,
      "occupied_bins_ratio": 0.08333333333333333
    },
    {
      "phase": "60-80",
      "n": 306000,
      "width_p95_p5": 561351.4612008936,
      "max_bin_share": 1.0,
      "occupied_bins_ratio": 0.016666666666666666
    },
    {
      "phase": "80-100",
      "n": 306051,
      "width_p95_p5": 557547.3423486252,
      "max_bin_share": 1.0,
      "occupied_bins_ratio": 0.016666666666666666
    }
  ],
  "comparativa_fases": {
    "width_ratio_80_100_vs_0_20": 2.063152463720875e-05,
    "peak_ratio_80_100_vs_0_20": 11.441354710033753
  },
  "booleans": {
    "pico_global_alto": true,
    "concentracion_global_fuerte_top3": true,
    "fases_finales_colapsadas": true,
    "pico_tardio_mas_agudo": true,
    "distribucion_global_extensa": true
  },
  "recomendacion_visual": "bajar mucho desde 3000",
  "max_por_bin_sugerido": 1000,
  "comentario_salida": "La masa del experimento esta fuertemente concentrada y las fases finales colapsan sobre una zona estrecha. Mantener 3000 probablemente conserva demasiada redundancia. Tiene sentido probar un recorte agresivo.",
  "n_inputs": 51,
  "source": {
    "experiment_dir": "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10",
    "funcion": "f1",
    "inputs_resueltos": [
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s1/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s10/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s11/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s12/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s13/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s14/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s15/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s16/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s17/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s18/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s19/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s2/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s20/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s21/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s22/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s23/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s24/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s25/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s26/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s27/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s28/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s29/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s3/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s30/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s31/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s32/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s33/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s34/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s35/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s36/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s37/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s38/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s39/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s4/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s40/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s41/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s42/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s43/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s44/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s45/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s46/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s47/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s48/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s49/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s5/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s50/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s51/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s6/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s7/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s8/dataset_age_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/age/age_cec2017_f1_d10_s9/dataset_age_cec2017_f1_d10.npz"
    ]
  }
}

{
  "algoritmo": "de",
  "n_runs": 51,
  "n_samples": 1530000,
  "global": {
    "visual_range_p0.5_p99.5": [
      62189918.96538371,
      13173406351.108377
    ],
    "max_bin_share": 0.13078353663971098,
    "top3_bins_share": 0.3434738949857695,
    "occupied_bins_ratio": 1.0,
    "width_p95_p5": 6113701028.539389,
    "tail_low_p5_share": 0.05,
    "tail_high_p95_share": 0.05
  },
  "phases": [
    {
      "phase": "0-20",
      "n": 305949,
      "width_p95_p5": 8721140546.006104,
      "max_bin_share": 0.12649388700599062,
      "occupied_bins_ratio": 1.0
    },
    {
      "phase": "20-40",
      "n": 306000,
      "width_p95_p5": 6113696943.314229,
      "max_bin_share": 0.13725490196078433,
      "occupied_bins_ratio": 0.3333333333333333
    },
    {
      "phase": "40-60",
      "n": 306000,
      "width_p95_p5": 6113697768.1611,
      "max_bin_share": 0.13725490196078433,
      "occupied_bins_ratio": 0.3333333333333333
    },
    {
      "phase": "60-80",
      "n": 306000,
      "width_p95_p5": 6113697775.006431,
      "max_bin_share": 0.13260612432645552,
      "occupied_bins_ratio": 0.3333333333333333
    },
    {
      "phase": "80-100",
      "n": 306051,
      "width_p95_p5": 6113697775.012516,
      "max_bin_share": 0.12,
      "occupied_bins_ratio": 0.3333333333333333
    }
  ],
  "comparativa_fases": {
    "width_ratio_80_100_vs_0_20": 0.7010204390998284,
    "peak_ratio_80_100_vs_0_20": 0.9486624440145232
  },
  "booleans": {
    "pico_global_alto": true,
    "concentracion_global_fuerte_top3": true,
    "fases_finales_colapsadas": false,
    "pico_tardio_mas_agudo": false,
    "distribucion_global_extensa": true
  },
  "recomendacion_visual": "bajar moderadamente desde 3000",
  "max_por_bin_sugerido": 1500,
  "comentario_salida": "Hay señales claras de concentracion o de colapso progresivo en fases finales, pero no un dominio tan extremo de la masa global. Conviene recortar, aunque no de forma tan agresiva.",
  "n_inputs": 51,
  "source": {
    "experiment_dir": "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10",
    "funcion": "f1",
    "inputs_resueltos": [
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s1/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s10/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s11/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s12/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s13/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s14/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s15/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s16/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s17/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s18/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s19/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s2/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s20/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s21/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s22/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s23/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s24/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s25/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s26/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s27/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s28/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s29/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s3/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s30/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s31/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s32/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s33/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s34/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s35/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s36/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s37/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s38/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s39/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s4/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s40/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s41/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s42/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s43/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s44/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s45/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s46/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s47/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s48/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s49/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s5/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s50/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s51/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s6/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s7/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s8/dataset_de_cec2017_f1_d10.npz",
      "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/metricas_runs/cec2017/de/de_cec2017_f1_d10_s9/dataset_de_cec2017_f1_d10.npz"
    ]
  }
}

quiero que como experto en investigacion de mhs y ml como redactor de papers, generes un conclusion lo mas cientifica posible que justifique el valor de max_Por_bins y n_bins para f1 segun los resultaos obtenidos tanto en los histogramas adjuntados como en los dos jsons



------

n_bins es 10 en ambos casos porque debe ser igual o porque para ambos es una cifra correcta? aunque tu decision a partir de los graficos visuales y los valores numericos coincide con los resultados de esta ultima, realmente opinas asi o simplemente has justificado le porque de los resultados del script¿

------

te muestro como han quedado los json asociados a age y de tras balancear a la baja + estratificaion pr seed + max_por_bin/n_bins como dijiste.

{
  "preprocesado": "balanceo_a_la_baja_estratificado",
  "criterio_seleccion": {
    "variable_balanceo": "fitness",
    "tipo_bins": "uniformes",
    "unidad_estratificacion": "seed",
    "estratificar_por_fase": false,
    "politica": "submuestreo_a_la_baja"
  },
  "n_runs_entrada": 51,
  "experimento_id": "experimentos_tam_10_age_cec2017_f1_d10",
  "parametros": {
    "n_bins": 10,
    "max_por_bin": 1000,
    "tipo_bins": "uniformes",
    "random_state": 42,
    "estratificar_por_fase": false
  },
  "muestras": {
    "original": 1530000,
    "balanceado": 3587,
    "retencion_pct": 0.2344
  },
  "fitness": {
    "original": {
      "min": 114.20291567824498,
      "max": 182752980151.37402,
      "media": 2237571650.466367,
      "mediana": 126158.38170760543,
      "desv_tipica": 5807780401.362751,
      "percentiles": {
        "p1": 156.38031447932576,
        "p5": 354.2308360147048,
        "p25": 6170.8857118736,
        "p50": 126158.38170760543,
        "p75": 554839710.6434386,
        "p95": 15103003428.73651,
        "p99": 27143250646.453648
      }
    },
    "balanceado": {
      "min": 114.20291567824498,
      "max": 182752980151.37402,
      "media": 31681170946.130222,
      "mediana": 27838919835.834343,
      "desv_tipica": 28286761488.3114,
      "percentiles": {
        "p1": 355.00511956841524,
        "p5": 5226.907241366057,
        "p25": 5668158974.914925,
        "p50": 27838919835.834343,
        "p75": 40747763234.459854,
        "p95": 87612103573.43163,
        "p99": 125838803392.18791
      }
    },
    "edges": [
      114.20291567824498,
      18275298117.920025,
      36550596121.63714,
      54825894125.35425,
      73101192129.07135,
      91376490132.78845,
      109651788136.50557,
      127927086140.22269,
      146202384143.9398,
      164477682147.6569,
      182752980151.37402
    ]
  },
  "distribucion_bins": {
    "antes": [
      1475892,
      50859,
      2662,
      276,
      153,
      82,
      45,
      18,
      9,
      4
    ],
    "despues": [
      1000,
      1000,
      1000,
      276,
      153,
      82,
      45,
      18,
      9,
      4
    ]
  },
  "distribucion_seeds": {
    "antes": {
      "1": 30000,
      "2": 30000,
      "3": 30000,
      "4": 30000,
      "5": 30000,
      "6": 30000,
      "7": 30000,
      "8": 30000,
      "9": 30000,
      "10": 30000,
      "11": 30000,
      "12": 30000,
      "13": 30000,
      "14": 30000,
      "15": 30000,
      "16": 30000,
      "17": 30000,
      "18": 30000,
      "19": 30000,
      "20": 30000,
      "21": 30000,
      "22": 30000,
      "23": 30000,
      "24": 30000,
      "25": 30000,
      "26": 30000,
      "27": 30000,
      "28": 30000,
      "29": 30000,
      "30": 30000,
      "31": 30000,
      "32": 30000,
      "33": 30000,
      "34": 30000,
      "35": 30000,
      "36": 30000,
      "37": 30000,
      "38": 30000,
      "39": 30000,
      "40": 30000,
      "41": 30000,
      "42": 30000,
      "43": 30000,
      "44": 30000,
      "45": 30000,
      "46": 30000,
      "47": 30000,
      "48": 30000,
      "49": 30000,
      "50": 30000,
      "51": 30000
    },
    "despues": {
      "1": 146,
      "2": 50,
      "3": 26,
      "4": 42,
      "5": 53,
      "6": 38,
      "7": 32,
      "8": 37,
      "9": 81,
      "10": 31,
      "11": 52,
      "12": 107,
      "13": 33,
      "14": 36,
      "15": 94,
      "16": 70,
      "17": 34,
      "18": 89,
      "19": 39,
      "20": 33,
      "21": 38,
      "22": 36,
      "23": 51,
      "24": 85,
      "25": 66,
      "26": 66,
      "27": 420,
      "28": 81,
      "29": 37,
      "30": 34,
      "31": 58,
      "32": 43,
      "33": 35,
      "34": 31,
      "35": 73,
      "36": 37,
      "37": 75,
      "38": 36,
      "39": 37,
      "40": 29,
      "41": 81,
      "42": 562,
      "43": 31,
      "44": 62,
      "45": 108,
      "46": 44,
      "47": 31,
      "48": 46,
      "49": 64,
      "50": 35,
      "51": 32
    }
  },
  "n_seeds_unicas": 51,
  "shapes": {
    "div_dist_euclidea": [
      3587
    ],
    "fitness": [
      3587
    ],
    "eval_id": [
      3587
    ],
    "generacion": [
      3587
    ],
    "x": [
      3587,
      10
    ],
    "seed": [
      3587
    ]
  },
  "claves": [
    "div_dist_euclidea",
    "eval_id",
    "fitness",
    "generacion",
    "seed",
    "x"
  ],
  "artefactos": {
    "dataset_npz": "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/preprocesado/age/dataset_balanceado_age.npz",
    "dataset_parquet": "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/preprocesado/age/dataset_balanceado_age.parquet"
  }
}

{
  "preprocesado": "balanceo_a_la_baja_estratificado",
  "criterio_seleccion": {
    "variable_balanceo": "fitness",
    "tipo_bins": "uniformes",
    "unidad_estratificacion": "seed",
    "estratificar_por_fase": false,
    "politica": "submuestreo_a_la_baja"
  },
  "n_runs_entrada": 51,
  "experimento_id": "experimentos_tam_10_de_cec2017_f1_d10",
  "parametros": {
    "n_bins": 10,
    "max_por_bin": 1500,
    "tipo_bins": "uniformes",
    "random_state": 42,
    "estratificar_por_fase": false
  },
  "muestras": {
    "original": 1530000,
    "balanceado": 5173,
    "retencion_pct": 0.3381
  },
  "fitness": {
    "original": {
      "min": 62189918.965383515,
      "max": 228895801291.0713,
      "media": 2140164179.7854235,
      "mediana": 1259596884.6016884,
      "desv_tipica": 3984655219.391694,
      "percentiles": {
        "p1": 62189918.97367215,
        "p5": 97918160.10559525,
        "p25": 516863731.5890687,
        "p50": 1259596884.6016884,
        "p75": 2388749979.0448127,
        "p95": 6211619188.644984,
        "p99": 13009178171.637394
      }
    },
    "balanceado": {
      "min": 62189918.965383515,
      "max": 228895801291.0713,
      "media": 41597575764.00257,
      "mediana": 36937629476.235756,
      "desv_tipica": 35661067657.96993,
      "percentiles": {
        "p1": 93049906.16713127,
        "p5": 462369025.6078205,
        "p25": 3776303078.4380946,
        "p50": 36937629476.235756,
        "p75": 61924358375.196396,
        "p95": 107485383480.35008,
        "p99": 146184666424.5049
      }
    },
    "edges": [
      62189918.965383515,
      22945551056.175972,
      45828912193.386566,
      68712273330.59716,
      91595634467.80774,
      114478995605.01833,
      137362356742.22893,
      160245717879.4395,
      183129079016.65012,
      206012440153.86072,
      228895801291.0713
    ]
  },
  "distribucion_bins": {
    "antes": [
      1525274,
      2553,
      1133,
      550,
      288,
      127,
      47,
      20,
      7,
      1
    ],
    "despues": [
      1500,
      1500,
      1133,
      550,
      288,
      127,
      47,
      20,
      7,
      1
    ]
  },
  "distribucion_seeds": {
    "antes": {
      "1": 30000,
      "2": 30000,
      "3": 30000,
      "4": 30000,
      "5": 30000,
      "6": 30000,
      "7": 30000,
      "8": 30000,
      "9": 30000,
      "10": 30000,
      "11": 30000,
      "12": 30000,
      "13": 30000,
      "14": 30000,
      "15": 30000,
      "16": 30000,
      "17": 30000,
      "18": 30000,
      "19": 30000,
      "20": 30000,
      "21": 30000,
      "22": 30000,
      "23": 30000,
      "24": 30000,
      "25": 30000,
      "26": 30000,
      "27": 30000,
      "28": 30000,
      "29": 30000,
      "30": 30000,
      "31": 30000,
      "32": 30000,
      "33": 30000,
      "34": 30000,
      "35": 30000,
      "36": 30000,
      "37": 30000,
      "38": 30000,
      "39": 30000,
      "40": 30000,
      "41": 30000,
      "42": 30000,
      "43": 30000,
      "44": 30000,
      "45": 30000,
      "46": 30000,
      "47": 30000,
      "48": 30000,
      "49": 30000,
      "50": 30000,
      "51": 30000
    },
    "despues": {
      "1": 124,
      "2": 81,
      "3": 90,
      "4": 116,
      "5": 88,
      "6": 103,
      "7": 97,
      "8": 77,
      "9": 111,
      "10": 80,
      "11": 116,
      "12": 138,
      "13": 157,
      "14": 112,
      "15": 75,
      "16": 85,
      "17": 105,
      "18": 108,
      "19": 108,
      "20": 109,
      "21": 76,
      "22": 77,
      "23": 87,
      "24": 94,
      "25": 96,
      "26": 110,
      "27": 82,
      "28": 122,
      "29": 87,
      "30": 92,
      "31": 118,
      "32": 72,
      "33": 110,
      "34": 102,
      "35": 102,
      "36": 90,
      "37": 96,
      "38": 100,
      "39": 121,
      "40": 89,
      "41": 86,
      "42": 119,
      "43": 169,
      "44": 100,
      "45": 82,
      "46": 104,
      "47": 85,
      "48": 101,
      "49": 90,
      "50": 75,
      "51": 159
    }
  },
  "n_seeds_unicas": 51,
  "shapes": {
    "seed": [
      5173
    ],
    "div_dist_euclidea": [
      5173
    ],
    "fitness": [
      5173
    ],
    "x": [
      5173,
      10
    ],
    "eval_id": [
      5173
    ],
    "generacion": [
      5173
    ]
  },
  "claves": [
    "div_dist_euclidea",
    "eval_id",
    "fitness",
    "generacion",
    "seed",
    "x"
  ],
  "artefactos": {
    "dataset_npz": "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/preprocesado/de/dataset_balanceado_de.npz",
    "dataset_parquet": "results/cec2017/experimentos_mhs_ambos_cec2017_d10_tam_10/f1/preprocesado/de/dataset_balanceado_de.parquet"
  }
}

-----

este es el json resumen de los resultados obitenods:

[
  {
    "algoritmo": "age",
    "modelo": "hgb",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 2869,
    "n_test_media": 717,
    "mae": 6769746935.088579,
    "mae_std": 2914192904.0119443,
    "rmse": 9160672617.48362,
    "rmse_std": 2827060933.380246,
    "r2": 0.8525044692661055,
    "r2_std": 0.15421920804683853,
    "spearman": 0.9326355992997751,
    "spearman_std": 0.02243566330005604,
    "train_time_s": 3.501184958405793,
    "train_time_s_std": 0.3481235751032113,
    "predict_time_s": 0.022125558205880226,
    "predict_time_s_std": 0.0012614571669534132,
    "rank_r2": 6
  },
  {
    "algoritmo": "age",
    "modelo": "mlp",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 2869,
    "n_test_media": 717,
    "mae": 3021897725.645827,
    "mae_std": 1143425638.547985,
    "rmse": 4287953961.7285256,
    "rmse_std": 1344906096.9638917,
    "r2": 0.9759041496595537,
    "r2_std": 0.010235209229747752,
    "spearman": 0.9704754824723463,
    "spearman_std": 0.019252809086555953,
    "train_time_s": 0.4514652167679742,
    "train_time_s_std": 0.07196024041088865,
    "predict_time_s": 0.00021450838539749384,
    "predict_time_s_std": 4.2300481973275985e-05,
    "rank_r2": 3
  },
  {
    "algoritmo": "age",
    "modelo": "random_forest",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 2869,
    "n_test_media": 717,
    "mae": 8392152704.001798,
    "mae_std": 2235419850.988341,
    "rmse": 11074760647.811514,
    "rmse_std": 2023574341.3996966,
    "r2": 0.8116539152254679,
    "r2_std": 0.13382945211682945,
    "spearman": 0.9307414382978795,
    "spearman_std": 0.030430414526895506,
    "train_time_s": 4.01938987497706,
    "train_time_s_std": 0.10167431229991869,
    "predict_time_s": 0.028012841776944696,
    "predict_time_s_std": 0.002386319997468684,
    "rank_r2": 7
  },
  {
    "algoritmo": "age",
    "modelo": "rbf",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 2869,
    "n_test_media": 717,
    "mae": 6628046389.430286,
    "mae_std": 2214024887.9892387,
    "rmse": 8820401125.338804,
    "rmse_std": 1859787217.68949,
    "r2": 0.8723965009615455,
    "r2_std": 0.10027557011593802,
    "spearman": 0.9651129247164683,
    "spearman_std": 0.009616991360430295,
    "train_time_s": 0.0009216167964041233,
    "train_time_s_std": 0.00029470828498990323,
    "predict_time_s": 0.046054958598688245,
    "predict_time_s_std": 0.005272663288545231,
    "rank_r2": 5
  },
  {
    "algoritmo": "age",
    "modelo": "rsm",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 2869,
    "n_test_media": 717,
    "mae": 5.235541477839144e-05,
    "mae_std": 1.992000876176461e-05,
    "rmse": 7.187151043171299e-05,
    "rmse_std": 3.3815592116468826e-05,
    "r2": 1.0,
    "r2_std": 0.0,
    "spearman": 0.9999999528469885,
    "spearman_std": 3.3729070463713414e-08,
    "train_time_s": 0.007779366825707257,
    "train_time_s_std": 0.0013554187277657942,
    "predict_time_s": 0.0005010416265577078,
    "predict_time_s_std": 7.037523613687583e-05,
    "rank_r2": 1
  },
  {
    "algoritmo": "age",
    "modelo": "svr",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 2869,
    "n_test_media": 717,
    "mae": 5508539378.4568205,
    "mae_std": 2128702748.2577865,
    "rmse": 7841949051.426608,
    "rmse_std": 2682244662.684063,
    "r2": 0.9167208431685067,
    "r2_std": 0.04487947567304475,
    "spearman": 0.9550473863263795,
    "spearman_std": 0.017415177452969822,
    "train_time_s": 0.05505714144092053,
    "train_time_s_std": 0.0012439094800917678,
    "predict_time_s": 0.012630449770949782,
    "predict_time_s_std": 0.0010450567606601106,
    "rank_r2": 4
  },
  {
    "algoritmo": "de",
    "modelo": "hgb",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 4138,
    "n_test_media": 1034,
    "mae": 3734609002.144422,
    "mae_std": 199298558.75333518,
    "rmse": 5521616188.803713,
    "rmse_std": 242561058.98355713,
    "r2": 0.9759391379172981,
    "r2_std": 0.002272663598466703,
    "spearman": 0.9822416791068653,
    "spearman_std": 0.005026827826877737,
    "train_time_s": 2.2281616084044797,
    "train_time_s_std": 0.0387765061696551,
    "predict_time_s": 0.014286083565093578,
    "predict_time_s_std": 0.0011120343927170635,
    "rank_r2": 5
  },
  {
    "algoritmo": "de",
    "modelo": "mlp",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 4138,
    "n_test_media": 1034,
    "mae": 2392968738.334103,
    "mae_std": 65824963.52996678,
    "rmse": 3357859508.0188956,
    "rmse_std": 157263630.87391615,
    "r2": 0.9911079564812655,
    "r2_std": 0.0007985165749443367,
    "spearman": 0.9887837090600534,
    "spearman_std": 0.003325924895519295,
    "train_time_s": 0.4472016081912443,
    "train_time_s_std": 0.04082916947135826,
    "predict_time_s": 0.00019056661985814572,
    "predict_time_s_std": 3.804787736768604e-05,
    "rank_r2": 3
  },
  {
    "algoritmo": "de",
    "modelo": "random_forest",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 4138,
    "n_test_media": 1034,
    "mae": 5845705517.204275,
    "mae_std": 289334406.28302,
    "rmse": 8645201794.409107,
    "rmse_std": 575467335.9736439,
    "r2": 0.9409174882160733,
    "r2_std": 0.00790375451811325,
    "spearman": 0.963958261995621,
    "spearman_std": 0.005738491069703968,
    "train_time_s": 3.474176733382046,
    "train_time_s_std": 0.07598694755762114,
    "predict_time_s": 0.0323125833645463,
    "predict_time_s_std": 0.004006973659763299,
    "rank_r2": 7
  },
  {
    "algoritmo": "de",
    "modelo": "rbf",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 4138,
    "n_test_media": 1034,
    "mae": 5146477678.121546,
    "mae_std": 146110084.3536723,
    "rmse": 6820986003.6731205,
    "rmse_std": 211756698.02549773,
    "r2": 0.9632836455250423,
    "r2_std": 0.0030237957065184975,
    "spearman": 0.9812119494320743,
    "spearman_std": 0.0025536526399874263,
    "train_time_s": 0.000825233431532979,
    "train_time_s_std": 1.8313638741549143e-05,
    "predict_time_s": 0.07150096676778048,
    "predict_time_s_std": 0.0023650214655754035,
    "rank_r2": 6
  },
  {
    "algoritmo": "de",
    "modelo": "rsm",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 4138,
    "n_test_media": 1034,
    "mae": 5.1474892404532475e-05,
    "mae_std": 1.002730024402679e-05,
    "rmse": 6.118825640462262e-05,
    "rmse_std": 1.0251097209230007e-05,
    "r2": 1.0,
    "r2_std": 0.0,
    "spearman": 0.9999939594563936,
    "spearman_std": 2.3659329644604634e-06,
    "train_time_s": 0.006575983413495123,
    "train_time_s_std": 0.00033716134432576315,
    "predict_time_s": 0.0004009832162410021,
    "predict_time_s_std": 3.517775929705435e-05,
    "rank_r2": 1
  },
  {
    "algoritmo": "de",
    "modelo": "svr",
    "feature_mode": "x_div",
    "split_strategy": "groupkfold",
    "n_folds": 5,
    "n_train_media": 4138,
    "n_test_media": 1034,
    "mae": 2786010185.5805573,
    "mae_std": 332749747.2391759,
    "rmse": 4080772806.1968756,
    "rmse_std": 422347760.0656628,
    "r2": 0.986718239584478,
    "r2_std": 0.0029466094099085674,
    "spearman": 0.990905631660065,
    "spearman_std": 0.004381817097887771,
    "train_time_s": 0.10124325843062251,
    "train_time_s_std": 0.006443901990656598,
    "predict_time_s": 0.023827299824915826,
    "predict_time_s_std": 0.0007775996895461791,
    "rank_r2": 4
  }
]
