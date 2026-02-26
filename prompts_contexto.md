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
