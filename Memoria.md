TRABAJO FIN DE GRADO
GRADO EN INGENIER´IA INFORM ´ATICA

Mejorando metaheur´ısticas
para el problema del
clustering semi-supervisado

Autor
Teresa C´ordoba Lillo

Directores
Daniel Molina Cabrera
Francisco Javier Rodr´ıguez D´ıaz

Escuela T´ecnica Superior de Ingenier´ıas Inform´atica y de
Telecomunicaci´on
—
Granada, Junio de 2025

1

Mejorando metaheur´ısticas
para el problema del
clustering semi-supervisado

Autor
Teresa C´ordoba Lillo

Directores
Daniel Molina Cabrera
Francisco Javier Rodr´ıguez D´ıaz

Mejorando metaheur´ısticas para el
problema del clustering semi-supervisado

Teresa C´ordoba Lillo

Palabras clave: clustering semi-supervisado, metaheur´ıstica, restricciones
Must-Link/Cannot-Link, Evoluci´on Diferencial, algoritmo mem´etico

Resumen

El clustering es una t´ecnica de aprendizaje no supervisado que agrupa
elementos seg´un su similitud. El clustering semi-supervisado extiende esta
idea incorporando una peque˜na cantidad de conocimiento experto, a menu-
do en forma de restricciones Must-Link y Cannot-Link, que indican si dos
elementos deben o no estar en el mismo grupo. Esta informaci´on permite
obtener soluciones m´as coherentes, interpretables y de mayor calidad, sin
necesidad de contar con datos completamente etiquetados. Por ello, se ha
convertido en una herramienta ´util con aplicaci´on en diversos ´ambitos como
la biomedicina o la clasificaci´on de textos, entre otros.

Recientemente se ha propuesto un algoritmo exacto para el clustering
semi-supervisado con restricciones Must-Link y Cannot-Link, pero resulta
inviable su uso con conjuntos de datos grandes debido a los altos tiempos de
ejecuci´on que requiere. Como alternativa, han surgido numerosos algoritmos
aproximados, muchos de ellos basados en metaheur´ısticas, entre los que des-
taca un algoritmo mem´etico reciente basado en Evoluci´on Diferencial, menos
costoso computacionalmente que el exacto, pero que no garantiza encontrar
la soluci´on ´optima.

El objetivo principal de este trabajo es analizar y mejorar el rendimien-
to de dicho algoritmo mem´etico. Se estudiar´a su comportamiento y se pro-
pondr´an diversas modificaciones orientadas a mejorarlo, tanto en t´erminos
de eficiencia computacional como de calidad de las soluciones generadas.
Adem´as, se realizar´a un an´alisis experimental exhaustivo ejecutando los al-
goritmos en m´ultiples conjuntos de datos, lo que permitir´a evaluar la calidad
de las propuestas y compararlas con el algoritmo original.

Improving metaheuristics for the
semi-supervised clustering problem

Teresa C´ordoba Lillo

Keywords: semi-supervised clustering, metaheuristics, Must-Link/Cannot-
Link constraints, Differential Evolution, memetic algorithm

Abstract

Clustering is an unsupervised learning technique that groups elements
according to their similarity. Semi-supervised clustering extends this idea
by incorporating a small amount of expert knowledge, often in the form
of Must-Link and Cannot-Link constraints, which indicate whether or not
two items should be in the same group. This information allows for mo-
re consistent, interpretable and higher quality solutions, without the need
for fully labeled data. Therefore, it has become a useful tool with applica-
tion in various fields such as biomedicine or text classification, among others.

Recently, an exact algorithm for semi-supervised clustering with Must-
Link and Cannot-Link constraints has been proposed, but it is infeasible
to use with large datasets due to the high execution times required. As an
alternative, numerous approximate algorithms have emerged, many of them
based on metaheuristics, including a recent memetic algorithm based on Dif-
ferential Evolution, which is less computationally expensive than the exact
one, but does not guarantee finding the optimal solution.

The main objective of this project is to analyse and improve the per-
formance of this memetic algorithm. Its behaviour will be studied and se-
veral modifications aimed at improving it will be proposed, both in terms
of computational efficiency and the quality of the solutions generated. In
addition, an exhaustive experimental analysis will be carried out by running
the algorithms on multiple data sets, which will allow us to evaluate the
quality of the proposals and compare them with the original algorithm.

Yo, Teresa C´ordoba Lillo, alumna de la titulaci´on Grado en Ingenier´ıa
Inform´atica de la Escuela T´ecnica Superior de Ingenier´ıas Inform´ati-
ca y de Telecomunicaci´on de la Universidad de Granada, con DNI
52022806A, autorizo la ubicaci´on de la siguiente copia de mi Trabajo Fin
de Grado en la biblioteca del centro para que pueda ser consultada por las
personas que lo deseen.

Fdo: Teresa C´ordoba Lillo

Granada a 15 de junio de 2025.

D. Daniel Molina Cabrera, Profesor del Departamento de Ciencias

de la Computaci´on e Inteligencia Artificial de la Universidad de Granada.

D. Francisco Javier Rodr´ıguez D´ıaz, Profesor del Departamento de
Ciencias de la Computaci´on e Inteligencia Artificial de la Universidad de
Granada.

Informan:

Que el presente trabajo, titulado Mejorando metaheur´ısticas para
el problema del clustering semi-supervisado, ha sido realizado bajo su
supervisi´on por Teresa C´ordoba Lillo, y autorizamos la defensa de dicho
trabajo ante el tribunal que corresponda.

Y para que conste, expiden y firman el presente informe en Granada a

15 de junio de 2025

Los directores:

Daniel Molina Cabrera

Francisco Javier Rodr´ıguez D´ıaz

Agradecimientos

A mi familia, y en especial a mis padres por su apoyo incondicional a lo
largo de estos a˜nos, por ense˜narme el valor del esfuerzo y por estar siempre
ah´ı, celebrando mis logros y anim´andome en cada paso del camino.

A mis tutores, por su gu´ıa, paciencia y dedicaci´on durante el desarrollo

de este trabajo.

A Laura, por acompa˜narme en este proceso, confiar en m´ı y animarme

en los momentos m´as dif´ıciles.

´Indice general

1. Introducci´on

1.1. Conceptos b´asicos y motivaci´on . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
1.2. Objetivos

2. Planificaci´on y presupuesto

2.1. Tareas realizadas . . . . . . . . . . . . . . . . . . . . . . . . .
2.2. Planificaci´on . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.3. Presupuesto . . . . . . . . . . . . . . . . . . . . . . . . . . . .

3. Revisi´on de la literatura

3.1. Algoritmos no basados en metaheur´ısticas . . . . . . . . . . .
3.2. Algoritmos basados en metaheur´ısticas . . . . . . . . . . . . .

1
1
3

4
4
5
6

7
8
10

4. Formalizaci´on del problema del clustering semi-supervisado 12

5. Algorimos de referencia

5.1. S-MDEClust

. . . . . . . . . . . . . . . . . . . . . . . . . . .
5.1.1. Representaci´on de la soluci´on . . . . . . . . . . . . . .
5.1.2. Paso de asignaci´on . . . . . . . . . . . . . . . . . . . .
5.1.3. Operador de cruce . . . . . . . . . . . . . . . . . . . .
5.1.4. Operador de mutaci´on . . . . . . . . . . . . . . . . . .
5.1.5. B´usqueda local
. . . . . . . . . . . . . . . . . . . . . .
Inicializaci´on de la poblaci´on . . . . . . . . . . . . . .
5.1.6.
5.2. Algoritmo exacto: PC-SOS-SDP . . . . . . . . . . . . . . . .
5.2.1. Notaci´on . . . . . . . . . . . . . . . . . . . . . . . . .
5.2.2. Reformulaci´on del problema . . . . . . . . . . . . . . .
5.2.3. C´alculo de la cota inferior . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . .
5.2.4. Desigualdades v´alidas
5.2.5. C´alculo de la cota superior
. . . . . . . . . . . . . . .
5.2.6. Esquema general del algoritmo . . . . . . . . . . . . .

6. Descripci´on de la propuesta

6.1. Asignaci´on . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6.1.1. Asignaci´on greedy aleatorizada . . . . . . . . . . . . .

15
15
16
18
19
20
21
21
22
22
23
24
25
26
27

29
29
30

i

´INDICE GENERAL

6.1.2. Asignaci´on greedy aleatorizada con penalizaci´on . . . .
6.2. B´usqueda Local . . . . . . . . . . . . . . . . . . . . . . . . . .
6.2.1. M´etodo Solis Wets . . . . . . . . . . . . . . . . . . . .
6.2.2. Selecci´on de individuos sobre los que se aplica la B´usque-
da Local . . . . . . . . . . . . . . . . . . . . . . . . . .
6.3. Evoluci´on Diferencial . . . . . . . . . . . . . . . . . . . . . . .
6.3.1. SHADE . . . . . . . . . . . . . . . . . . . . . . . . . .
6.3.2. Cambios en el operador de cruce . . . . . . . . . . . .
6.4. Reinicio de la poblaci´on . . . . . . . . . . . . . . . . . . . . .
6.5. Otro enfoque: GRASP . . . . . . . . . . . . . . . . . . . . . .

7. Dise˜no experimental

7.1. Conjuntos de datos utilizados . . . . . . . . . . . . . . . . . .
7.2. Conjunto de restricciones utilizados . . . . . . . . . . . . . . .
7.3. Detalles de la implementaci´on y par´ametros de los algoritmos
7.3.1. Algoritmo exacto: PC-SOS-SDP . . . . . . . . . . . .
7.3.2. Algoritmo mem´etico: S-MDEClust (versi´on original) .
7.3.3. Algoritmos propuestos . . . . . . . . . . . . . . . . . .
7.4. M´etricas empleadas . . . . . . . . . . . . . . . . . . . . . . . .
7.5. Librer´ıas y dependencias . . . . . . . . . . . . . . . . . . . . .

8. Experimentaci´on y resultados

8.1. Resultados del algoritmo referencia y del algoritmo exacto . .
8.2. Acr´onimos . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8.3. Enfoque GRASP . . . . . . . . . . . . . . . . . . . . . . . . .
8.4. Variantes de la asignaci´on greedy . . . . . . . . . . . . . . . .
8.5. Evoluci´on Diferencial . . . . . . . . . . . . . . . . . . . . . . .
8.5.1. SHADE . . . . . . . . . . . . . . . . . . . . . . . . . .
8.5.2. Cambios en el operador de cruce . . . . . . . . . . . .
8.6. B´usqueda Local . . . . . . . . . . . . . . . . . . . . . . . . . .
8.6.1. Uso del algoritmo Solis Wets
. . . . . . . . . . . . . .
8.6.2. Combinando el uso de varias propuestas . . . . . . . .
8.6.3. Seleccionar a qu´e individuos aplicar la b´usqueda local
Introducci´on de la estrategia de reinicio de poblaci´on . . . . .
8.7.1. Disminuci´on del tama˜no de la poblaci´on . . . . . . . .
8.7.2. Combinando diferentes propuestas . . . . . . . . . . .
8.8. Resumen de los resultados experimentales . . . . . . . . . . .

8.7.

9. Conclusiones y trabajo futuro

Bibliograf´ıa

A. Par´ametros de ejecuci´on del algoritmo PC-SOS-SDP

ii

31
32
32

33
36
36
39
39
40

44
44
46
47
47
48
48
49
50

53
53
55
56
57
59
59
60
63
63
66
69
70
74
75
77

81

86

87

´Indice de figuras

2.1. Planificaci´on seguida . . . . . . . . . . . . . . . . . . . . . . .

3.1. N´umero de publicaciones anuales sobre el clustering semi-

supervisado con restricciones extra´ıdo de Scopus.

. . . . . . .

5.1. Ejemplo de 2 dimensiones con K=3 y n=28. Las restricciones
ML se representan con l´ıneas verdes y las CL con l´ıneas rojas

8.1. Evoluci´on de la diversidad . . . . . . . . . . . . . . . . . . . .
8.2. Comparativa evoluci´on de la diversidad con y sin reinicios . .
8.3. Comparativa evoluci´on del Score con y sin reinicios . . . . . .
8.4. Comparativa del Score de todas las propuestas: veces que ha

6

7

17

61
73
73

sido mejor, igual o peor que el algoritmo S-MDEClust original. 80

iii

´Indice de tablas

2.1. Tiempo dedicado . . . . . . . . . . . . . . . . . . . . . . . . .
2.2. Estimaci´on del presupuesto del proyecto . . . . . . . . . . . .

Informaci´on datasets . . . . . . . . . . . . . . . . . . . . . . .
. . . . .

7.1.
7.2. Par´ametros de ejecuci´on del algoritmo S-MDEClust
7.3. Par´ametros de ejecuci´on del algoritmo S-MDEClust, inclu-

yendo las propuestas . . . . . . . . . . . . . . . . . . . . . . .
7.4. Par´ametros de ejecuci´on del algoritmo GRASP . . . . . . . .

5
6

46
49

51
52

54
8.1. Resultados algoritmo exacto . . . . . . . . . . . . . . . . . . .
54
8.2. Resultados algoritmo S-MDEClust original
. . . . . . . . . .
56
8.3. Resultados GRASP . . . . . . . . . . . . . . . . . . . . . . . .
57
8.4. Resultados AGR-RAND . . . . . . . . . . . . . . . . . . . . .
58
8.5. Resultados AGR-RAND-P . . . . . . . . . . . . . . . . . . . .
59
8.6. Resultados SHADE . . . . . . . . . . . . . . . . . . . . . . . .
62
8.7. Comparativa PBEST1-F y PBEST2-F . . . . . . . . . . . . .
63
8.8. Comparativa PBEST1-F/2 y PBEST2-F/2 . . . . . . . . . .
64
8.9. Comparativa PBEST1-F y SW-V1-WO-PEN . . . . . . . . .
65
8.10. Comparativa PBEST1-F y SW-WO-PEN . . . . . . . . . . .
66
8.11. Comparativa PBEST1-F y SW-W-PEN . . . . . . . . . . . .
67
8.12. Comparativa SW-WO-PEN y SW-WO-PEN-AGR-RAND . .
8.13. Comparativa PBEST1-F y PBEST1-F-SW-WO-PEN . . . . .
68
8.14. Comparativa PBEST1-F-AGR-RAND y PBEST1-F-AGR-RAND-
69
SW . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
70
8.15. Comparativa PBEST1-F-AGR-RAND-SW y SEL-BL . . . . .
71
8.16. Comparativa PBEST1-F-AGR-RAND-SW y R-2IT-4 . . . . .
72
8.17. Comparativa R-3IT-2 y R-3IT-2-SW . . . . . . . . . . . . . .
74
8.18. Comparativa R-3IT-2-DIS y R-3IT-2-DIS-SW . . . . . . . . .
8.19. Comparativa R-3IT-2-PBEST1-F y R-3IT-2-PBEST1-F-SW .
75
8.20. Comparativa R-3IT-2-DIS-PBEST1-F y R-3IT-2-DIS-PBEST1-
F-SW . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8.21. Comparativa R-3IT-2-PBEST1-F y R-3IT-2-PBEST1-F-AGR-
RAND . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

77

76

iv

´INDICE DE TABLAS

v

8.22. Resultados comparativos del Score de las propuestas . . . . .

79

A.1. Par´ametros de configuraci´on del algoritmo exacto PC-SOS-SDP 88

Cap´ıtulo 1

Introducci´on

1.1. Conceptos b´asicos y motivaci´on

El agrupamiento (clustering) busca agrupar un conjunto de instancias
en grupos llamados clusters, de forma que las instancias que pertenecen a
un mismo cluster sean m´as similares entre s´ı que con las que pertenecen a
clusters distintos. De forma habitual, el clustering se utiliza cuando no se
tiene informaci´on a priori sobre la pertenencia de los datos a clases predefi-
nidas. Por esta raz´on, tradicionalmente se considera una t´ecnica propia del
aprendizaje no supervisado. Sin embargo, en ocasiones podemos disponer de
cierta informaci´on parcial o conocimiento experto que podemos utilizar pa-
ra guiar la b´usqueda de soluciones hacia soluciones m´as significativas y m´as
cercanas a la partici´on real de los datos. De esta forma, estamos incorpo-
rando un cierto grado de supervisi´on a la tarea, lo que da lugar al clustering
semi-supervisado.

El clustering semisupervisado presenta ventajas claras frente al enfoque
no supervisado, ya que al aprovechar la informaci´on parcial puede mejo-
rar la calidad de las soluciones obtenidas y hacerlas m´as coherentes con
el conocimiento experto del dominio del problema [1]. Adem´as, favorece la
interpretabilidad, ya que las soluciones reflejan las relaciones conocidas o
deseadas dentro de los datos.

A lo largo de los a˜nos, el clustering semisupervisado se ha utilizado en
una amplia variedad de campos. Entre sus aplicaciones se encuentran el
an´alisis de datos biol´ogicos, la segmentaci´on de im´agenes, v´ıdeo y sonido,
la clasificaci´on de textos y el an´alisis de datos en la web [2]. Incluso se
ha utilizado en tareas como la detecci´on de carriles en carreteras a partir
de datos GPS de coches. Esta ´ultima fue la primera aplicaci´on propuesta
por Wagstaff et al. [3]. Estos ejemplos muestran su versatilidad y utilidad,
especialmente en contextos donde una supervisi´on total no es viable, pero

1

Introducci´on

2

s´ı se puede disponer de conocimiento parcial.

Existen numerosos modelos de clustering, cada uno con diferentes en-
foques y criterios de similitud, as´ı como distintas formas de incorporar co-
nocimiento experto al proceso [4]. En este trabajo nos centraremos en el
Euclidean Minimum Sum of Squares Clustering (MSSC) con restricciones
Must-Link (ML) y Cannot-Link (CL), tambi´en conocido como constrained
clustering. En este enfoque, las instancias se representan como vectores del
espacio eucl´ıdeo d-dimensional Rd, y se busca encontrar una partici´on de
los datos en clusters de forma que se minimice la suma de las distancias
eucl´ıdeas al cuadrado entre cada instancia y el centroide del cluster al que
pertenece, respetando las restricciones y siendo el centroide la media de to-
das las instancias asignadas a dicho cluster. Las restricciones ML indican
que un par de instancias deben pertenecer al mismo cluster, mientras que
las restricciones CL obligan a que est´en en clusters distintos. Resolver este
problema es computacionalmente complejo; si bien el MSSC no supervisado
es NP-hard, como se demuestra en [5], resolver su variante con restricciones
es al menos tan dif´ıcil como su versi´on no supervisada.

Debido a su complejidad, a pesar de que existen algoritmos exactos para
el MSSC con restricciones, su aplicaci´on en conjuntos de datos grandes no
es factible. Es por ello por lo que han surgido en los ´ultimos a˜nos multitud
de algoritmos basados en metaheur´ısticas [6], que buscan encontrar una so-
luci´on pr´oxima a la ´optima globalmente, pero en un tiempo mucho menor.
Entre ellos, se encuentra el algoritmo S-MDEClust (Mansueto y Schoen,
2024) [7], que es un algoritmo mem´etico basado en Evoluci´on Diferencial.
Este algoritmo, que describiremos en m´as detalle en el Cap´ıtulo 5, es com-
parado en el art´ıculo original con un algoritmo branch-and-cut exacto [8]
con buenos resultados: S-MDEClust consigue igualar al exacto en la calidad
de las soluciones en t´erminos de la funci´on objetivo que se busca minimi-
zar (la suma de distancias al cuadrado entre instancias y centroides) en la
mayor´ıa de los conjuntos de datos en los que se prueba, y adem´as logra en-
contrar la soluci´on en un tiempo bastante menor en promedio. Sin embargo,
el tiempo de ejecuci´on en ciertos casos sigue siendo prohibitivo, especialmen-
te en aquellos conjuntos de datos con un gran n´umero de instancias o alta
dimensionalidad, donde el coste computacional aumenta significativamente.

En este trabajo buscamos proponer modificaciones en uno o varios com-
ponentes del algoritmo S-MDEClust con el prop´osito de mejorarlo, ya sea
en t´erminos de la calidad de las soluciones que encuentra, como en cuanto
a eficiencia computacional.

Introducci´on

1.2. Objetivos

3

El objetivo principal de este trabajo es estudiar y mejorar la optimi-
zaci´on del clustering semi-supervisado con restricciones mediante el uso de
metaheur´ısticas, centr´andose en mejorar la eficiencia. En concreto, se par-
te de un algoritmo de referencia basado en una estrategia mem´etica con
Evoluci´on Diferencial: S-MDEClust [7], sobre el que se proponen y eval´uan
distintas modificaciones encaminadas a mejorar su eficiencia y la calidad de
las soluciones obtenidas.

En resumen, los objetivos de este trabajo son:

Analizar en profundidad el funcionamiento del algoritmo de referencia:
S-MDEClust.

Proponer mejoras que incrementen su rendimiento computacional y la
calidad de las particiones generadas.

Dise˜nar e implementar una alternativa metaheur´ıstica con un enfoque
distinto.

Evaluar experimentalmente las propuestas utilizando m´ultiples con-
juntos de datos.

Comparar los resultados obtenidos frente al algoritmo de referencia y
extraer conclusiones sobre su utilidad pr´actica.

Cap´ıtulo 2

Planificaci´on y presupuesto

En este cap´ıtulo se aborda la planificaci´on seguida para el desarrollo del
proyecto, desglosando el trabajo realizado en tareas y detallando el n´umero
de horas dedicadas a cada una. A partir de esta informaci´on, se realiza una
estimaci´on del presupuesto total del proyecto, teniendo en cuenta el tiempo
total invertido.

2.1. Tareas realizadas

A continuaci´on se detallan las tareas realizadas.

Comprensi´on del problema: Revisi´on del material inicial propor-
cionado por los tutores, as´ı como la b´usqueda y lectura de informaci´on
complementaria con el objetivo de comprender el problema del cluste-
ring semi-supervisado.

B´usqueda de informaci´on: B´usqueda y revisi´on art´ıculos, publica-
ciones y otros documentos necesarios para la realizaci´on del proyecto.

Estudio de los algoritmos de referencia: an´alisis y estudio detalla-
do del funcionamiento de los algoritmos usados como base comparativa
en el proyecto.

Implementaci´on: Incluye tanto la implementaci´on de los componen-
tes de la propuesta, como el desarrollo de otros scripts necesarios (au-
tomatizaci´on de ejecuciones, creaci´on de gr´aficas y tablas, generaci´on
de restricciones y formateo de datasets).

Obtenci´on de los resultados: Ejecuci´on de los algoritmos de refe-
rencia y de las propuestas implementadas sobre los distintos conjuntos
de datos.

4

Planificaci´on y presupuesto

5

An´alisis de los resultados: An´alisis e interpretaci´on de los resulta-
dos obtenidos.

Realizaci´on de la memoria: Redacci´on de la memoria y revisi´on de
la misma.

Reuniones con los tutores: Reuniones casi semanales en las que se
discutieron las modificaciones a implementar, los resultados obtenidos
y en las tomaron decisiones clave del proyecto, como la elecci´on de los
conjuntos de datos, las m´etricas de evaluaci´on y otros aspectos.

2.2. Planificaci´on

A continuaci´on se muestra una tabla con el tiempo dedicado a cada una

de las tareas especificadas en el apartado anterior.

Tarea
Comprensi´on del problema
B´usqueda de informaci´on
Estudio de los algoritmos de referencia
Implementaci´on
Obtenci´on de los resultados
An´alisis de los resultados
Realizaci´on de la memoria
Reuniones con los tutores
TOTAL

Duraci´on (horas)
10
20
15
120
191.63
10
150
15
340 + 191.63

Tabla 2.1: Tiempo dedicado

El tiempo total dedicado al proyecto ha sido 340 horas de trabajo, m´as
191.63 horas de c´omputo para la obtenci´on de los resultados, lo que hace un
total de 531.63 horas.

La planificaci´on seguida se detalla en la Figura 2.1 haciendo uso de un
diagrama de Gantt, que permite visualizar de forma clara la distribuci´on
temporal de las tareas.

Como podemos ver, hay varias tareas que se han realizado en paralelo,
como la implementaci´on, la obtenci´on de resultados y su an´alisis, as´ı como
la b´usqueda de informaci´on necesaria para la realizaci´on del proyecto y las
reuniones peri´odicas con los tutores.

Planificaci´on y presupuesto

6

Figura 2.1: Planificaci´on seguida

2.3. Presupuesto

Para realizar una estimaci´on econ´omica del proyecto, debemos tener en
cuenta tres aspectos principales: el precio de la mano de obra, el precio de
c´omputo y el precio del ordenador utilizado para realizar el proyecto.

El precio de la mano de obra se ha estimado en 20e/hora. Por otro lado,
el ordenador utilizado ha sido un ASUS TUF Dash F15, con un procesador
Intel Core i7 12650H y 16 GB de RAM. El precio dicho port´atil actualmente
es de 1100 e. Para estimar el coste asociado al proyecto, se ha supuesto una
vida ´util de cinco a˜nos para el port´atil, y dado que el desarrollo ha durado
aproximadamente un a˜no, se ha considerado un 20 % del precio total, lo que
equivale a 220 e.

Para calcular el coste del precio de c´omputo, se ha utilizado la calcula-
dora de precios de Amazon Web Services (AWS). Al seleccionar el servicio
EC2 e introducir las especificaciones del equipo utilizado, se obtiene que la
instancia m´as similar es la c6g.4xlarge, con un coste por hora bajo demanda
de 0.51 e.

Con estos datos, ya es posible calcular el presupuesto final del proyecto:

20

e

h

· 340h + 0.51

e

h

· 191.63h +

1100e
5

= 7117.73e

Concepto
Mano de obra
C´omputo
Ordenador
Total

Precio Base Duraci´on Precio Total
340 horas
191.63 horas
1 a˜no de 5

20 e/hora
0.51 e/hora
1 100 e

6 800 e
97.73 e
220 e
7117.73 e

Tabla 2.2: Estimaci´on del presupuesto del proyecto

Finalmente, el presupuesto del proyecto queda fijado en 7117.73 e.

Cap´ıtulo 3

Revisi´on de la literatura

En los ´ultimos a˜nos, el inter´es por el problema del clustering semi-
supervisado con restricciones ha crecido notablemente, como muestra la Fi-
gura 3.1. Se ha producido un aumento en el n´umero de publicaciones por
a˜no, de menos de 10 al principio de los 2000s a casi 100 en los ´ultimos a˜nos,
que refleja la importancia y la investigaci´on creciente sobre este tema. Como
vimos en el cap´ıtulo anterior, incorporar conocimiento experto al clustering
permite obtener particiones m´as ´utiles y ajustadas a las necesidades reales,
lo que ha motivado el desarrollo de una amplia variedad de algoritmos.

Figura 3.1: N´umero de publicaciones anuales sobre el clustering semi-
supervisado con restricciones extra´ıdo de Scopus.

En este cap´ıtulo haremos un repaso por algunos de los algoritmos que se

han propuesto para el clustering semi-supervisado con restricciones.

7

Revisi´on de la literatura

8

3.1. Algoritmos no basados en metaheur´ısticas

Multitud de los algoritmos existentes para clustering semi-supervisado
son adaptaciones del cl´asico algoritmo para clustering no supervisado K-
means (MacQueen, 1967) [9]. En este grupo se encuentra uno de los primeros
algoritmos propuestos para este problema: COP-KMEANS (Wagstaff et al.
2001) [3]. COP-KMEANS parte de una soluci´on inicial y repite los siguientes
dos pasos hasta alcanzar la convergencia, es decir, hasta que deja de haber
cambios en la soluci´on.

1. Asignar cada instancia a un cluster de la soluci´on.

2. Modificar los centroides de la soluci´on para que sean la media de las

instancias asignadas a ´el.

La diferencia con K-means es que en el paso 1, en lugar de asignar cada
instancia al centroide m´as cercano, se asigna al m´as cercano tal que se sa-
tisfagan todas las restricciones ML y CL. Este algoritmo sigue un enfoque
voraz (greedy), en el que las decisiones de asignaci´on se toman de forma
inmediata sin realizar backtracking. Esto hace que sea muy r´apido pero pre-
senta un gran inconveniente, y es que aunque exista una soluci´on factible que
satisfaga todas las restricciones, COP-KMEANS no garantiza encontrarla,
ya que el ´exito depende en gran medida del orden en el que se asignan las
instancias. Adem´as, incluso en los casos en los que logra encontrar una solu-
ci´on factible, esta podr´ıa corresponder ´unicamente a un ´optimo local, lejos
de la mejor soluci´on posible en t´erminos de la funci´on objetivo.

Para tratar el problema de la dependencia del orden de asignaci´on en
COP-KMEANS, se han propuesto distintas variantes. Una de ellas es ICOP-
k-means [10], que precalcula el orden de asignaci´on basado en la certeza de
cada instancia, estimada mediante t´ecnicas de clustering ensemble. As´ı, se
asignan primero las instancias m´as seguras, reduciendo el riesgo de violar
restricciones en asignaciones posteriores.

CLC-Kmeans [11] es otra variante de COP-KMEANS que tambi´en modi-
fica el orden de asignaci´on, agrupando las instancias relacionadas por restric-
ciones CL y asign´andolas de forma conjunta a clusters compatibles. De este
modo, se minimizan los conflictos durante la asignaci´on. Ambas propuestas
mejoran la robustez de COP-KMEANS, incrementando la probabilidad de
encontrar soluciones factibles.

Entre las variantes m´as recientes se encuentra BLPKMCC [12], que abor-
da el paso de asignaci´on 1 formul´andolo como un problema de programaci´on
binaria, garantizando as´ı el cumplimiento de todas las restricciones. Profun-
dizaremos m´as en este enfoque en el Cap´ıtulo 5, ya que es empleado por el
algoritmo mem´etico de referencia S-MDEClust.

Revisi´on de la literatura

9

Los algoritmos mencionados hasta ahora tienen en com´un que buscan
cumplir todas las restricciones de forma estricta. Sin embargo, tambi´en exis-
ten otros enfoques que permiten incumplir restricciones a cambio de una
penalizaci´on en la funci´on objetivo. Un ejemplo de esto es PCK-Means [13],
que al asignar las instancias a los clusters en el paso 1 busca minimizar una
versi´on modificada de la funci´on objetivo del MSSC (suma de distancias
eucl´ıdeas al cuadrado entre instancias y centroides), a la que se a˜nade un
t´ermino de penalizaci´on que depende del n´umero de restricciones incumpli-
das.

Por otro lado, tambi´en se han propuesto m´etodos exactos para resolver
el problema del clustering semi-supervisado con restricciones. El primero de
ellos fue propuesto por Xia en 2009 [14], que es una extensi´on para incluir
restricciones ML y CL del m´etodo exacto para MSSC propuesto unos a˜nos
antes en [15]. Este m´etodo logra encontrar la soluci´on ´optima en conjuntos
de datos muy peque˜nos, aproximadamente de 25 instancias seg´un Aloise en
[16], aunque tambi´en puede proporcionar soluciones aproximadas deteniendo
el algoritmo antes de alcanzar el ´optimo global.

Posteriormente, en 2014, Babaki, Guns y Nijssen propusieron un enfoque
basado en generaci´on de columnas [17], donde el problema se resuelve de
forma iterativa a˜nadiendo progresivamente variables (columnas) que tienen
el potencial de mejorar la soluci´on actual. Este m´etodo permite manejar
restricciones ML y CL, as´ı como restricciones m´as generales, aunque su
aplicaci´on pr´actica se limita a conjuntos de datos de menos de 200 instancias.

M´as tarde, en 2016, Guns et al. propusieron el algoritmo CPRBBA [18],
que sigue una estrategia branch-and-bound. CPRBBA tambi´en permite op-
timizar varios objetivos a la vez, como minimizar la funci´on objetivo del
MSSC y maximizar la separaci´on entre clusters, pero, al igual que m´etodos
anteriores, su aplicabilidad est´a restringida a problemas de tama˜no reducido,
de alrededor de 200 instancias.

Recientemente, en 2022, Piccialli et al. propusieron el algoritmo PC-SOS-
SDP [8]. Este algoritmo ha supuesto un gran avance en cuanto a m´etodos
exactos para el MSSC semi-supervisado. Se trata de un algoritmo branch-
and-cut que, para el c´alculo de la cota inferior, utiliza una relajaci´on por
programaci´on semidefinida (SDP) del MSSC y para la cota superior el algo-
ritmo BLPKMCC [12] mencionado con anterioridad, con una inicializaci´on
m´as sofisticada. Este algoritmo ha demostrado ser significativamente m´as
eficiente que sus predecesores, siendo capaz de encontrar la soluci´on ´optima
en conjuntos de datos de hasta 800 instancias, con m´as de 20000 dimensiones
y un n´umero de restricciones aproximadamente igual a la mitad del tama˜no
del conjunto de datos. Adem´as, destaca por su buena escalabilidad con res-
pecto a la dimensionalidad, manteniendo su eficacia incluso en problemas
de alta dimensionalidad donde otros m´etodos exactos no resultan pr´acticos.

Revisi´on de la literatura

10

3.2. Algoritmos basados en metaheur´ısticas

A pesar de los avances logrados con los m´etodos exactos, su aplicabilidad
pr´actica sigue estando limitada a problemas de tama˜no reducido debido
a su elevado coste computacional. Por ello, se han desarrollado m´ultiples
algoritmos aproximados, como los que hemos visto al inicio del cap´ıtulo.
Dentro de este enfoque aproximado, una parte importante de las propuestas
se basa en metaheur´ısticas,

Las metaheur´ısticas son algoritmos de optimizaci´on de prop´osito gene-
ral que, en muchas ocasiones, est´an inspirados en procesos naturales, como
la evoluci´on biol´ogica, el comportamiento social de animales o fen´omenos
f´ısicos. Su objetivo principal es encontrar soluciones de buena calidad en
un tiempo reducido mediante un equilibrio entre la exploraci´on del espacio
de soluciones (buscar en regiones diversas) y la explotaci´on (intensificar la
b´usqueda en ´areas del espacio de soluciones prometedoras).

Un ejemplo de metaheur´ıstica para clustering con restricciones ML y
CL es el algoritmo MCLA [19] (Vu, Labroche y Bouchon-Meunier, 2009),
extensi´on del algoritmo para clustering Leader Ant (LA) que se basa en el
comportamiento de las colonias de hormigas. En este enfoque, cada hormiga
artificial est´a caracterizada por un genoma, que corresponde a una instancia
espec´ıfica del conjunto de datos, y una etiqueta que identifica la colonia
(cluster ) a la que pertenece. En cada iteraci´on, se elige aleatoriamente una
hormiga que no haya sido ya asignada a una colonia y se calcula su distancia
media (similitud) con un determinado n´umero de hormigas de cada colonia
viable, tomando en cuenta ´unicamente aquellas colonias que cumplen las
restricciones. La hormiga se asigna a la colonia con la menor distancia media,
siempre que esta distancia supere un umbral establecido. Si no se alcanza
el umbral o no hay colonias existentes, la hormiga crea una nueva colonia
y se convierte en su l´ıder. En [19] tambi´en se proponen dos modificaciones
del algoritmo con el mismo esquema pero adaptadas para tratar con otros
tipos de restricciones.

Otro ejemplo de algoritmo basado en colonias de hormigas es CAC [20],
basado en el algoritmo RWAC, inspirado en el comportamiento de las hor-
migas al buscar un lugar para dormir. Este algoritmo tiene en cuenta las res-
tricciones, modificando fuerzas atractivas y repulsivas entre hormigas (ins-
tancias) involucradas en las restricciones.

Dentro de las metaheur´ısticas, los algoritmos gen´eticos (AGs) han de-
mostrado ser especialmente efectivos para abordar problemas complejos de
optimizaci´on como el clustering con restricciones. Los AGs, inspirados en
la selecci´on natural, trabajan con una poblaci´on de soluciones candidatas
y aplican operadores de cruce y mutaci´on para generar nuevas soluciones
que compiten con las existentes en la poblaci´on para formar parte de esta.

Revisi´on de la literatura

11

Un ejemplo destacado es el algoritmo propuesto por Gribel et al. en 2022
[21], una adaptaci´on de HG-MEANS [22], algoritmo considerado parte del
estado del arte en MSSC no supervisado. Una caracter´ıstica innovadora de
esta propuesta es la incorporaci´on de un par´ametro que mide la precisi´on de
las restricciones, lo que permite gestionar posibles errores en su definici´on.
Otro algoritmo destacado es S-MDEClust [7], una propuesta reciente que
sigue una estrategia mem´etica basada en Evoluci´on Diferencial en la que
profundizaremos m´as en el Cap´ıtulo 5.

Por otro lado, m´as recientemente, en 2020, Gonz´alez-Almagro et al. pro-
pusieron el algoritmo DILSCC [23], que sigue una estrategia DILS, variante
del m´etodo cl´asico de b´usqueda local iterativa (Iterative Local Search, ILS).
Esta estrategia tiene como objetivo explorar el espacio de b´usqueda para
encontrar la soluci´on que optimice una funci´on objetivo determinada, que
en este caso es el producto de la suma de distancias media intra-cluster y el
n´umero de restricciones incumplidas por la soluci´on.

A diferencia de los algoritmos ILS, los DILS mantienen dos soluciones en
memoria en todo momento en lugar de una: la mejor soluci´on (mb) y la peor
(mw) de la iteraci´on actual del algoritmo, evaluadas seg´un la funci´on obje-
tivo. El algoritmo DILSCC comienza inicializando mb y mw aleatoriamente.
En cada iteraci´on, se combinan ambas soluciones para generar una nueva
soluci´on, a la que posteriormente se aplica un operador de mutaci´on y una
fase de b´usqueda local. Si la nueva soluci´on generada (mt) mejora a mw, esta
´ultima se actualiza con mt. A pesar de que el operador de mutaci´on intro-
duzca diversidad en la b´usqueda, existe el riesgo de caer en un ´optimo local
tras un n´umero reducido de iteraciones. Para tratar este problema, cuando
se detecta estancamiento, medido en t´erminos de la diferencia de funci´on
objetivo de mb y mw, se reinicializa mw aleatoriamente manteniendo la me-
jor soluci´on mb. DILSCC ha sido comparado con varios algoritmos del estado
del arte, mostrando una alta similitud entre las particiones obtenidas y las
particiones reales, gracias a su equilibrio entre exploraci´on y explotaci´on en
la b´usqueda de soluciones ´optimas.

Esta revisi´on ofrece una visi´on general del estado actual del clustering
semi-supervisado con restricciones, proporcionando una base ´util para el
desarrollo posterior del trabajo.

Cap´ıtulo 4

Formalizaci´on del problema
del clustering
semi-supervisado

El problema del agrupamiento por suma m´ınima de cuadrados (mini-
mum sum-of-squares clustering, MSSC) con restricciones puede ser formu-
lado como un problema de optimizaci´on global con restricciones, en el que
se tiene como objetivo minimizar una funci´on objetivo: la suma de las dis-
tancias eucl´ıdeas al cuadrado entre cada instancia y el centroide al que est´a
asignado, asegurando el cumplimiento de las restricciones ML y CL.

Formalmente, dado un conjunto de datos con n instancias D = {p1, p2, ..., pn},

donde pi es un vector real de d dimensiones, es decir, pi ∈ Rd
∀i ∈
{1, ..., n}, y unos conjuntos de restricciones ML y CL denotados por ML,
CL ⊂ {1, ..., n} × {1, ..., n} respectivamente, resolver el problema de cluste-
ring consiste en encontrar una partici´on de los datos en K clusters, C =
{C1, C2, ...CK}.

En esta formulaci´on, se asume que el n´umero de clusters K es conocido
de antemano. Sin embargo, en muchas aplicaciones pr´acticas, el n´umero
´optimo de clusters no se conoce a priori, y encontrar dicho valor constituye
en s´ı mismo un problema complejo, que suele requerir t´ecnicas adicionales,
como m´etodos basados en informaci´on estad´ıstica. Estudiar estos m´etodos
no es el objetivo de este trabajo, as´ı que asumiremos en todo momento que
K es conocido.

El centroide del cluster j, denotado µj ∈ Rd, se define como el promedio

de las instancias asignadas a dicho cluster :

µj =

(cid:80)n
(cid:80)n

i=1 xijpi
i=1 xij

12

(4.1)

Formalizaci´on del problema del clustering semi-supervisado

13

donde, xij ∈ {0, 1} son variables binarias tales que

xij =

(cid:40)

1 si la instancia pi est´a asignada al cluster j
0 en caso contrario

Se busca que la partici´on C encontrada minimice la siguiente funci´on

objetivo:

f (C) =

n
(cid:88)

K
(cid:88)

i=1

j=1

xij ∥pi − µj∥2

(4.2)

sujeto a las siguientes restricciones:

Cada instancia debe pertenecer exactamente a un ´unico cluster.

K
(cid:88)

j=1

xij = 1 ∀i ∈ {1, . . . , n}

(4.3)

Cada cluster debe tener al menos una instancia asignada.

n
(cid:88)

i=1

xij ≥ 1 ∀j ∈ {1, . . . , K}

(4.4)

Si dos instancias est´an involucradas en la misma restricci´on must-link,
deben pertenecer al mismo cluster.

xik = xjk ∀(i, j) ∈ ML, ∀k ∈ {1, . . . , K}

(4.5)

Si dos instancias est´an involucradas en la misma restricci´on cannot-
link, no pueden pertenecer al mismo cluster.

xik + xjk ≤ 1 ∀(i, j) ∈ CL, ∀k ∈ {1, . . . , K}

(4.6)

Con esta formulaci´on, el MSSC semi-supervisado es un problema de pro-
gramaci´on mixta, es decir, un problema de optimizaci´on que involucra tanto
variables enteras (las variables xij ∈ {0, 1}, que indican la asignaci´on de ins-
tancias a clusters) como variables reales continuas (los centroides µj ∈ Rd,
que son vectores de d dimensiones).

Este problema y su versi´on no supervisada son NP-hard [24]. En parte,
esta elevada complejidad se debe a dos factores: por un lado, las restriccio-
nes ML y CL se imponen como restricciones duras, es decir, se impone que
la soluci´on encontrada debe necesariamente respetar estas restricciones, lo

Formalizaci´on del problema del clustering semi-supervisado

14

que reduce el espacio de soluciones factibles y complica significativamente
la exploraci´on de soluciones viables. Por otro lado, la funci´on objetivo es
no lineal y no convexa, y est´a caracterizada por la presencia de m´ultiples
´optimos locales, lo que dificulta encontrar el ´optimo global. Estas dificulta-
des hacen que el MSSC semi-supervisado sea un problema extremadamente
complejo de resolver de forma exacta.

Adem´as, al imponer restricciones duras surge un problema adicional que
no se daba en el MSSC no supervisado, y es que puede ocuirrir que no exista
ninguna soluci´on que satisfaga todas las restricciones. Determinar si dado
un conjunto de datos y un conjunto de restricciones ML y CL, existe una
soluci´on que satisfaga todas las restricciones es a lo que se conoce como
problema de factibilidad (referido en la literatura como feasibility problem),
y fue demostrado en [25] que es NP-completo.

Debido a estas dificultades, gran parte de la investigaci´on en clustering
semi-supervisado se ha centrado en el desarrollo de algoritmos heur´ısticos
y metaheur´ısticos capaces de encontrar soluciones de buena calidad en un
tiempo razonable.

Cap´ıtulo 5

Algorimos de referencia

En este cap´ıtulo describiremos el algoritmo de referencia a partir del
cual se desarrollar´an las propuestas de mejora presentadas en los siguientes
cap´ıtulos: el algoritmo S-MDEClust [7]. Asimismo, se presenta el algoritmo
exacto PC-SOS-SDP [8], que se utilizar´a para obtener las soluciones ´optimas.

5.1. S-MDEClust

S-MDEClust es una adaptaci´on de un algoritmo para MSSC no super-
visado de los mismos autores [26], para incorporar restricciones ML y CL.
Como ya hemos comentado con anterioridad, S-MDEClust es un algoritmo
mem´etico. Este tipo de algoritmos son poblacionales, es decir, en lugar de
tener una ´unica soluci´on, trabajan con un conjunto de soluciones que forman
una poblaci´on. En cada iteraci´on del algoritmo se generan nuevas solucio-
nes combinando de alguna forma las soluciones de la poblaci´on actual y se
eval´ua si dichas soluciones deben entrar a formar parte de la poblaci´on, re-
emplazando a alguna soluci´on actual, o no. Adem´as, antes de este proceso
de reemplazamiento se puede aplicar un procedimiento de b´usqueda local
para refinar la soluci´on, utilizando conocimiento espec´ıfico del problema.

A continuaci´on presentamos el esquema general del algoritmo S-MDEClust

y posteriormente definiremos en detalle cada uno de sus componentes.

15

Algorimos de referencia

16

Algorithm 1 Algoritmo S-MDEC1ust
Require: conjunto de datos (dataset) D = {p1, . . . , pn} ⊂ Rd, n´umero
de clusters K ∈ N, conjuntos de restricciones must-link y cannot-link
ML, CL ⊂ {1, . . . , n} × {1, . . . , n} , tama˜no de la poblaci´on P ∈ N,
NM AX ∈ N, tol ∈ R+.

1: Inicializar poblaci´on P = {S1, . . . , SP }
2: Determinar S⋆ ∈ P tal que f (S⋆) ≤ f (S), ∀S ∈ P
3: Inicializar n⋆
4: while (n⋆
5:

it < NM AX ) ∧ ((cid:80)P

for all Ss ∈ P do

it = 0

(cid:80)

s=1

¯s>s |f (Ss) − f (S¯s)| > tol) do

6:

7:

8:

9:

10:

11:

12:

13:

14:

15:

16:

17:

Seleccionar aleatoriamente S1, S2, S3 ∈ P, todos diferentes entre s´ı
y distintos de Ss
Aplicar operador de cruce con S1, S2, S3 para generar un descen-
diente Os
Aplicar o no el operador de mutaci´on a Os para obtener ˜Os, con un
probabilidad que disminuye a lo largo de las iteraciones
Aplicar b´usqueda local a ˜Os para obtener una soluci´on O′
s
if f (O′

s) < f (Ss) then

Ss = O′
s
s) < f (S⋆) then
if f (O′

S⋆ = O′
s
n⋆
it = 0
else
n⋆
it = n⋆
end if

it + 1

18:

end if
end for
19:
20: end while
21: return S⋆

5.1.1. Representaci´on de la soluci´on

Cada individuo S de la poblaci´on P = {S1, . . . , SP } se representa me-

diante dos estructuras de datos:

Vector de pertenencia ϕS ∈ Nn, tal que ϕS
i = k ⇔ xik = 1, con
k ∈ {1, ..., K}. Es decir, la posici´on i-´esima de este vector indica el
n´umero del cluster al que est´a asignado la instancia i-´esima.

Matriz de coordenadas ψS = [µ1, µ2, ..., µK]T ∈ RK×d, es decir, una
matriz en la que la fila i-´esima contiene las coordenadas del cluster
i-´esimo.

A continuaci´on se muestra un ejemplo para ilustrar esto.

Algorimos de referencia

17

Figura 5.1: Ejemplo de 2 dimensiones con K=3 y n=28. Las restricciones
ML se representan con l´ıneas verdes y las CL con l´ıneas rojas

Matriz de coordenadas

1
2.5
2

1.6
2
0.8

Vector de pertenencia

1

1

1

1

1

1

1

1

1

3

3

3

3

3

3

3

3

3

2

2

2

2

2

2

2

2

2

Con esta notaci´on, podemos expresar la funci´on objetivo 4.2 de la si-

guiente manera.

f (S) =

n
(cid:88)

i=1

(cid:13)
(cid:13)pi − ψS
(cid:13)
ϕS
i

(cid:13)
2
(cid:13)
(cid:13)

.

(5.1)

Cabe destacar que cada estructura, por s´ı sola, permite caracterizar com-
pletamente la soluci´on; sin embargo, es conveniente mantener las dos en
memoria para reducir el tiempo de ejecuci´on. A partir del vector de per-
tenencia ϕ, podemos obtener ψ en O(nKd) recordando que el centroide de
cada cluster es la media de las instancias que pertenecen a dicho cluster, co-
mo se indica en la Ecuaci´on 4.1. Obtener ϕ desde la matriz de coordenadas
ψ es m´as complejo. No basta con asignar cada instancia a su centroide m´as
cercano, como sucede en el MSSC no supervisado; ahora deben considerarse
las restricciones ML y CL.

Algorimos de referencia

18

Este hecho se puede apreciar en la figura del ejemplo anterior 5.1. Si nos
fijamos en la instancia p16, vemos que en lugar de asignarse al cluster 1, que
ser´ıa el m´as cercano, se asigna al cluster 3 para respetar las restricciones.

5.1.2. Paso de asignaci´on

Entendemos como paso de asignaci´on a la asignaci´on de las instancias
a los centroides, es decir, a la obtenci´on de ϕS a partir de ψS. En [7] se
proponen dos m´etodos de asignaci´on: una asignaci´on exacta y una asignaci´on
greedy m´as eficiente pero que no garantiza el cumplimiento de todas las
restricciones ML y CL. La asignaci´on greedy ser´a ´util en partes intermedias
del algoritmo, en las que no es indispensable que las soluciones cumplan
todas las restricciones, para reducir el coste computacional.

Asignaci´on exacta

El m´etodo de asignaci´on exacta consiste en resolver el siguiente problema

de optimizaci´on.

m´ın

N
(cid:88)

K
(cid:88)

i=1

k=1

Asignaci´on Greedy

xik∥pi − ψS

k ∥2,

sujeto a 4.3, 4.4, 4.5, 4.6

(5.2)

Primeramente, se dividen todas las instancias en grupos basados en las
restricciones must-link, de forma que las instancias que est´an relacionadas
mediante este tipo de restricciones se encuentren en el mismo grupo. Una
vez establecidos estos grupos, para cada grupo G, se busca un cluster al que
se asignar´an todas las instancias de dicho grupo, de forma que no exista otro
grupo ˆG asignado al mismo cluster y al que pertenezca una instancia que
tenga una restricci´on cannot-link con una instancia de G. Adem´as, entre los
clusters que cumplan esta condici´on, se elegir´a aquel que minimice la suma
de distancias de las instancias del grupo G a su centroide. Dependiendo del
orden en el que se realicen las asignaciones de los grupos, podr´ıa ocurrir que,
aunque exista una soluci´on factible, no sea posible encontrar ning´un cluster
para alg´un grupo G sin incumplir ninguna restricci´on cannot-link, en cuyo
caso se asignar´a al cluster m´as cercano a los puntos de G sin considerar
ninguna restricci´on.

A continuaci´on, se muestra el pseudoc´odigo de la asignaci´on greedy.

Algorimos de referencia

19

Algorithm 2 Asignaci´on Greedy
Require: dataset D = {p1, . . . , pn} ⊂ Rd, n´umero de clusters K ∈ N, con-
juntos de restricciones must-link y cannot-link ML, CL ⊂ {1, . . . , n} ×
{1, . . . , n} , matriz de centroides ψS.

1: Definir GML = {G ⊂ {1, . . . , N } | ∀i, j ∈ G ,sujeto a i ̸= j, (i, j) ∈ ML}

2: Inicializar A = ∅
3: for all G ∈ GML do
4: Determinar

sujeto a:

kG ∈ arg m´ın

k

∥pi − ψS

k ∥2

(cid:88)

i∈G

k ∈ {1, . . . , K} | ∄( ˆG, k) ∈ A tal que ∃(ic, jc) ∈ CL con ic ∈ G∧jc ∈ ˆG

5:

6:

if kG no est´a definido then

Determinar

kG ∈ arg m´ın

k∈{1,...,K}

∥pi − ψS

k ∥2

(cid:88)

i∈G

end if

7:
8: A = A ∪ {(G, kG)}
9:

for all i ∈ G do
ϕS
i = kG
end for

10:

11:
12: end for
13:
14: return ϕS

Este algoritmo no garantiza que la soluci´on encontrada satisfaga todas
las restricciones. Sin embargo, se puede usar, en lugar del m´etodo exacto,
durante el cruce y mutaci´on. En estos pasos intermedios no es necesario
que se satisfagan todas las restricciones y de esta forma se reduce el coste
computacional.

5.1.3. Operador de cruce

Dada una soluci´on Ss = (ϕSs, ψSs) ∈ P, con s ∈ {1, ..., n}. Se eligen tres
soluciones aleatorias S1, S2, S3 ∈ P, de forma que sean diferentes entre s´ı y
diferentes a Ss. La soluci´on descendiente Os = (ϕOs, ψOs) se obtiene de la

Algorimos de referencia

20

siguiente manera.

ψOs = ψS1 + F (ψS2 − ψS3)

(5.3)

con F ∈ (0.5, 0.8). ϕOs se obtiene realizando un paso de asignaci´on 5.1.2.

Algo a tener en cuenta es que, dada una soluci´on S del problema de
clustering, la permutaci´on de las filas de su matriz de coordenadas ψS da
lugar a otra soluci´on que es totalmente equivalente. Por ello, antes de realizar
el cruce 5.3 se debe de aplicar alguna estrategia para emparejar los centroides
de las soluciones S1, S2 y S3, como la descrita en [27].

5.1.4. Operador de mutaci´on

El operador de mutaci´on tiene como objetivo introducir diversidad en la
poblaci´on, evitando una convergencia prematura hacia soluciones sub´opti-
mas. Consiste en una relocalizaci´on aleatoria de uno de los centroides de la
soluci´on. A medida que avanzan las iteraciones, la probabilidad de aplicar
esta operaci´on disminuye progresivamente, ya que se pretende favorecer la
intensificaci´on en fases m´as avanzadas de la b´usqueda, centr´andose en re-
finar las soluciones prometedoras en lugar de explorar nuevas regiones del
espacio de b´usqueda.

Partiendo de la soluci´on resultante del cruce Os = (ϕOs, ψOs), el operador

de mutaci´on se aplica de la siguiente forma:

1. Elegir un centroide ¯k ∈ {1, ...K} con probabilidad uniforme y elimi-

narlo de la soluci´on.

2. Asignar cada instancia pi, con i ∈ {1, ..., n} a uno de los K − 1 clusters

restantes, dando lugar al vector de pertenencia temporal ˜ϕ.

3. Se elige una instancia p¯i, ¯i ∈ {1, ..., n} como nuevo centroide, basada

en la siguiente probabilidad.

P (¯i) =

(cid:32)

(cid:19)

(cid:18) 1 − α
n

+

α · d¯i
j=1 dj

(cid:80)n

(cid:33)

(5.4)

donde dj es la distancia entre la instancia pj y su centroide asignado,
es decir, dj = ||pj − ψOs
||. Adem´as, el par´ametro α ∈ [0, 1] juega un
˜ϕj
papel fundamental en la elecci´on del nuevo centroide. Cuando α = 0,
todas las instancias tienen la misma probabilidad de ser elegidas como
centroides. En cambio, si α = 1, las instancias m´as alejadas de su
centroide asignado tienen una mayor probabilidad de ser seleccionadas.

Algorimos de referencia

21

Es necesario hacer dos aclaraciones con respecto al paso 2, ya que su

comportamiento depende del tipo de asignaci´on que se emplee.

Asignaci´on exacta: Aunque pueda existir una soluci´on factible que
satisfaga todas las restricciones para K clusters, al reducir el n´umero
de clusters a K − 1 es posible que no exista ninguna partici´on que
cumpla todas las restricciones. Si se da este caso, la asignaci´on exacta
no ser´a viable. En consecuencia, se establecer´a α = 0, es decir, todas
las instancias tendr´an la misma probabilidad de ser seleccionadas como
centroides en el paso 3.

Asignaci´on Greedy: En la asignaci´on solo se deben de considerar
las instancias que estuvieran asignadas al centroide ¯k eliminado en el
primer paso 1.

5.1.5. B´usqueda local

La b´usqueda local es el algoritmo BLPKMCC, una variante de K-Means
dise˜nada para clustering semi-supervisado, propuesta en [12]. Este algoritmo
parte de una configuraci´on inicial de centroides ψ y repite los siguientes dos
pasos hasta cumplir un determinado criterio de parada.

1. Se aplica la asignaci´on exacta, detallada en la Ecuaci´on 5.2, para de-

terminar los valores de ϕ.

2. Se actualizan los centroides ψ de modo que cada ψi se convierta en el
centro geom´etrico de las instancias asignadas al i-´esimo cluster, con-
forme a la ecuaci´on 4.1.

El criterio de parada es que se alcance la convergencia, es decir, que los

centroides no se actualicen en dos iteraciones sucesivas.

La b´usqueda local cumple una doble funci´on: por un lado, permite refinar
las soluciones generadas tras las fases de cruce y mutaci´on; por otro, act´ua
como mecanismo de reparaci´on, ya que al utilizar la asignaci´on exacta se
garantiza que la soluci´on resultante satisfaga todas las restricciones ML y
CL.

5.1.6.

Inicializaci´on de la poblaci´on

Cada individuo de la poblaci´on inicial P = {S1, . . . , SP } se genera apli-
cando la b´usqueda local explicada en la Subsecci´on 5.1.5 sobre una soluci´on
aleatoria. Los centroides de dicha soluci´on aleatoria se obtienen eligiendo K
instancias (sin reemplazo) del conjunto de datos D.

Algorimos de referencia

22

5.2. Algoritmo exacto: PC-SOS-SDP

En esta secci´on, describimos el funcionamiento del algoritmo exacto PC-
SOS-SDP, desarrollado por Piccialli et al. [8]. Este algoritmo pertenece a la
familia de m´etodos branch and cut, que combinan la exploraci´on del espa-
cio de soluciones mediante ramificaci´on (branching), dividiendo el problema
original en subproblemas m´as peque˜nos, con el uso de desigualdades v´alidas
(cutting planes) que permiten descartar regiones del espacio de soluciones
no factibles o sub´optimas y acelerar la convergencia. Utilizamos este algo-
ritmo para obtener la soluci´on ´optima del problema y compararla con los
resultados alcanzados por nuestras propuestas.

5.2.1. Notaci´on

A continuaci´on, describimos brevemente la notaci´on que se va a usar en

esta secci´on.

N = {1, . . . , n} es el conjunto de ´ındices de las instancias del conjunto
de datos.

K = {1, . . . , K} es el conjunto de ´ındices de los clusters.

ML, CL ⊆ N × N son los conjuntos de restricciones ML y CL res-
pectivamente.

S n es el conjunto de todas las matrices reales sim´etricas de tama˜no
n × n.

M ⪰ 0 denota que la matriz M es semidefinida positiva.

+ es el conjunto de todas las matrices semidefinidas positivas de

S n
tama˜no n × n.

⟨·, ·⟩ denota el producto escalar de traza. Es decir, para cualquier
A, B ∈ Rm×n, definimos ⟨A, B⟩ := trace(B⊤A).

Dada una matriz A, denotamos por Ai la i-´esima fila de A.

Denotamos por en el vector de unos de longitud n y por In la matriz
identidad de tama˜no n × n. Se omitir´a el sub´ındice en caso de que la
dimensi´on sea clara por el contexto.

Denotamos por F = {xij ∈ {0, 1} sujeto a 4.3, 4.4, 4.5 y 4.6} a la
regi´on discreta factible del problema del MSSC con restricciones 4.2.

Algorimos de referencia

23

5.2.2. Reformulaci´on del problema

Podemos expresar el problema del MSSC con restricciones 4.2, sustitu-

yendo los centroides µj por la expresi´on 4.1, de la siguiente forma.

m´ın

n
(cid:88)

K
(cid:88)

i=1

j=1

xij

(cid:13)
(cid:13)
(cid:13)
(cid:13)

pi −

(cid:80)n
(cid:80)n

l=1 xljpl
l=1 xlj

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)

(5.5)

sujeto a xij ∈ F,

∀i ∈ N ,

∀j ∈ K.

Este problema, cuando ML = CL = ∅, fue demostrado por Peng y Wei
[28] que es equivalente a un problema SDP (semidefinite programming) no
lineal, llamado 0-1 SDP. En este tipo de problemas, la variable principal
es una matriz semidefinida positiva sobre la cual se optimiza una funci´on
objetivo, sujeta a restricciones que introducen no linealidad. Cualquier so-
luci´on del MSSC sin restricciones est´a asociada a una soluci´on del siguiente
problema.

m´ın

tr(W (I − Z))

(5.6)

sujeto a Ze = e,

tr(Z) = k, Z ≥ 0, Z2 = Z, Z = Z⊤.

donde:

Wp es la matriz con el conjunto de datos, en la que cada fila corres-
ponde con una instancia pi.

W = WpW ⊤
la que Wij = p⊤

i pj ∀i, j ∈ N .

p es la matriz de productos escalares de las instancias, en

X es una matriz de tama˜no n×K que contiene las variables de decisi´on
xij.

La matriz Z se define como Z = X(X ⊤X)−1X ⊤. Se puede comprobar
que Z es una matriz sim´etrica sin elementos negativos, que cumple
Z = Z2. Adem´as, las suma de sus filas y columnas es 1 y su traza K.

Piccialli et al. [8] extienden esta formulaci´on para contemplar las restric-

ciones ML y CL, comprobando que:

Si las instancias pi y pj pertenencen al mismo cluster C, se cumple
Zi = Zj. Adem´as, los elementos distintos de cero de las columnas i y j
de Z son iguales a 1
|C| , donde |C| es el n´umero de elementos del cluster
C.

Algorimos de referencia

24

Si las instancias pi y pj pertenencen al clusters distintos, entonces
Zij = 0.

Esto permite expresar el problema del MSSC con restricciones 4.2 me-

diante la siguiente formulaci´on 0-1 SDP.

sujeto a Ze = e,

tr(Z) = k, Z ≥ 0, Z2 = Z, Z = Z⊤,

m´ın

tr(W (I − Z))

Zih = Zjh ∀h ∈ N ,
Zij = 0 ∀(i, j) ∈ CL.

∀(i, j) ∈ ML,

(5.7)

(5.8)

5.2.3. C´alculo de la cota inferior

A diferencia del problema original 4.2, el problema 5.8 tiene una fun-
ci´on objetivo lineal, y adem´as, todas sus restricciones excepto Z = Z2 son
tambi´en lineales. Relajando esta restricci´on a Z ⪰ 0 obtenemos la siguiente
relajaci´on SDP, con la que podemos obtener una buena cota inferior para el
problema original.

tr(W (I − Z))

m´ın
tr(Z) = k, Z ≥ 0, Z = Z⊤, Z ∈ S n
+
∀(i, j) ∈ ML,

(5.9)

sujeto a Ze = e,

Zih = Zjh ∀h ∈ N ,
Zij = 0 ∀(i, j) ∈ CL.

Adem´as, las restricciones ML cumplen una serie de propiedades que per-
miten reducir la dimensionalidad del problema. A partir del conjunto inicial
ML se puede formar el conjunto de restricciones ML ampliado ML′ apli-
cando las siguientes propiedades.

Simetr´ıa: Si existe la restricci´on (i, j) ∈ ML′, tambi´en debe cumplirse
(j, i) ∈ ML′ ∀i, j ∈ N .

Reflexividad: Cada instancia est´a relacionada consigo misma, es de-
cir, (i, i) ∈ ML′ ∀i ∈ N .

Transitividad: Si (i, h) ∈ ML′ y (h, j) ∈ ML′, entonces (i, j) ∈
ML′,

∀i, h, j ∈ N .

A partir de ML′ podemos agrupar las instancias del conjunto de datos
en s grupos G1, . . . Gs, con s ≤ n, tal que ∀(i, j) ∈ Gl
∀l ∈
{1, . . . , s}. Podemos representar cada grupo Gi mediante un ´unico punto ¯pi,
calculado como la suma de todas las instancias de Gi.

(i, j) ∈ ML′,

Algorimos de referencia

25

¯pi =

(cid:88)

pj

pj ∈Gi

As´ı, el problema se transforma de encontrar una partici´on para las instan-
cias originales p1, . . . pn a encontrar una partici´on para los puntos ¯p1, . . . , ¯ps.
La relajaci´on SDP 5.2.3 es equivalente al siguiente problema.

m´ın ⟨I, W ⟩ − ⟨T sW (T s)⊤, Z⟩
sujeto a Zes = e,

⟨Diag(es), Z⟩ = K,

Zij = 0 ∀(i, j) ∈ CL,
Z ≥ 0, Z ∈ S s
+.

(5.10)

donde

T es una matriz s × n, tal que T s

ij =

(cid:40)
1
0

si j ∈ Gi
si j /∈ Gi

CL es el conjunto de restricciones CL entre los puntos ¯pi, tal que
pl ∈ Gi,
(i, j) ∈ CL si

(l, m) ∈ CL,

pm ∈ Gj.

El algoritmo PC-SOS-SDP produce un ´arbol binario, en el que el nodo
ra´ız corresponde con el problema 0-1 SDP 5.8. En cada ramificaci´on del
´arbol, se eligen dos instancias pi y pj y se generan dos nodos hijos: en el hijo
izquierdo se impone una restricci´on ML entre ellas, mientras que en el hijo
derecho se impone una CL. De este modo, el conjunto de soluciones cada
nodo padre se divide en dos subconjuntos disjuntos.

Para elegir el par de instancias pi y pj se eligen aquellas tales que:

arg m´ax

i,j

(cid:8)m´ın (cid:8)Zij, ∥Zi − Zj∥2(cid:9)(cid:9)

5.2.4. Desigualdades v´alidas

El algoritmo PC-SOS-SDP utiliza tres tipos de desigualdades para re-

forzar la cota.

Desigualdades de pares. para toda soluci´on factible, se cumple lo
siguiente:

Zij ≤ Zii, Zij ≤ Zjj,

∀i, j ∈ N , i ̸= j.

(5.11)

Algorimos de referencia

26

Desigualdades triangulares. Si las instancias pi y pj est´an en el
mismo cluster, y las instancias pj y ph tambi´en est´an en el mismo
cluster, entonces pi y ph deben de estar en el mismo cluster.

Zij + Zih ≤ Zii + Zjh ∀i, j, h ∈ N , siendo i, j, h distintos entre s´ı.

(5.12)

Desigualdades de clique. Si el n´umero de clusters es igual a K, en
cualquier subconjunto de instancias de tama˜no K + 1 deben de haber
el menos dos instancias que pertenezcan al mismo cluster.

(cid:88)

(i,j)∈Q,i<j

Zij ≥

1
n − K + 1

∀Q ⊆ N , |Q| = K + 1.

(5.13)

Estas desigualdades se a˜naden a la relajaci´on SDP mediante un proce-
dimiento de plano de corte ´unicamente si son incumplidas. Es decir, no se
a˜naden todas desde el inicio, sino que solo se a˜naden aquellas que no se satis-
facen en la soluci´on actual del problema relajado. Tras a˜nadirlas, se vuelve a
calcular la soluci´on del problema con las restricciones incorporadas. Adem´as,
despu´es de cada iteraci´on del plano de corte se eliminan las restricciones que
no est´en activas en la soluci´on actual, es decir, aquellas que no est´en limi-
tando directamente la soluci´on ´optima encontrada. De esta forma, se evita
el aumento excesivo del n´umero de restricciones y se mantiene el tama˜no del
problema dentro de unos l´ımites computacionalmente manejables. Por otro
lado, las desigualdades que se incluyeron al problema de un nodo padre en
la ´ultima iteraci´on son directamente heredadas por los nodos hijos desde el
inicio.

5.2.5. C´alculo de la cota superior

El algoritmo utilizado para calcular la cota superior en cada nodo es
una variante del algoritmo BLPKMCC [12]. Este algoritmo, como vimos an-
teriormente en la Secci´on 5.1.5, garantiza que la soluci´on obtenida cumpla
todas las restricciones, siempre que exista una soluci´on factible. Sin embago,
presenta un inconveniente: es altamente sensible a la elecci´on de los centroi-
des iniciales. Para tratar este problema, Piccialli et al. [29] proponen una
t´ecnica de inicializaci´on basada en la soluci´on de la relajaci´on SDP, utiliza-
da previamente para hallar la cota inferior. La idea es que, si la soluci´on Z
de la relajaci´on es ajustada, es decir, es una soluci´on factible del problema
original 5.8, entonces se pueden recuperar los centroides iniciales a partir de
la matriz Z. Si no lo es, se busca una matriz ˆZ, lo m´as cercana posible a Z
en t´erminos de la norma de Frobenius, es decir, se minimiza la suma de los
cuadrados de las diferencias entre las entradas de ambas matrices. Una vez
hallada ˆZ se obtienen los centroides a partir de esta matriz.

Algorimos de referencia

27

La modificaci´on de BLPKMCC que incorpora la inicializaci´on descrita se
llama IPC-k-means, y es la heur´ıstica utilizada para calcular la cota superior
en cada nodo del algoritmo PC-SOS-SDP. El pseudoc´odigo de IPC-k-means
se presenta a continuaci´on.

Algorithm 3 IPC-k-means
Require: Dataset D = {p1, . . . , pn}, n´umero de clusters K, conjuntos de
restricciones ML y CL, matriz de datos Wp, la soluci´on ´optima ˆZ de la
relajaci´on SDP con restricciones ML y CL

1: Resolver ˆZ = arg m´ın{∥ ˆZ − Z∥F sujeto a rango(Z) = K}.
2: Calcular la aproximaci´on de la matriz de centroides ˆM = ˆZWp.
3: Agrupar las filas de ˆM con k-means para obtener los centroides iniciales

de los clusters µ1, . . . , µK.

4: repeat
5:

Calcular las asignaciones ´optimas de clusters x∗

ij resolviendo:

m´ın

n
(cid:88)

K
(cid:88)

i=1

j=1

xij∥pi − µj∥2

sujeto a xij ∈ F,

∀i ∈ N , ∀j ∈ K (5.14)

6: Definir Cj ← {pi : x∗
7: Actualizar los centroides de los clusters µ1, . . . , µK calculando la me-

ij = 1} para cada j = 1, . . . , K.

dia de los puntos asignados a cada cluster C1, . . . , CK.

8: until convergencia;
9: Salida: Clusters C1, . . . , CK

5.2.6. Esquema general del algoritmo

Una vez explicados los distintos componentes que forman el algoritmo
PC-SOS-SDP, presentamos a continuaci´on el pseudc´odigo completo del mis-
mo.

Algorimos de referencia

28

Algorithm 4 Algoritmo PC-SOS-SDP

1: Entrada: Conjuntos de restricciones ML y CL, n´umero de clusters K,

matriz W

2: Construir T s, es y CL a partir de ML y CL.
3: Sea P0 el problema 0-1 SDP inicial 5.8, establecer Q = {P0}
4: Inicializar la mejor soluci´on X ∗ = null y su valor de la funci´on objetivo

v∗ = ∞

5: while Q no est´e vac´ıo do
6:

7:

8:

9:

10:

11:

12:

13:

14:

15:

16:

17:

Seleccionar y extraer un problema P de Q.
Comprobar si el problema P es factible. Si no lo es, ir al Paso 5.
Resolver la relajaci´on SDP 5.10 para obtener una cota inferior LB y
la soluci´on ´optima Z .
if LB ≥ v∗ then
Ir al Paso 5.

end if
Buscar desigualdades de pares 5.11, triangulares 5.12 y de clique 5.13
incumplidas por Z. Si se encuentran, a˜nadirlas a la relajaci´on SDP
actual y volver al Paso 8
Ejecutar el algoritmo Algorithm 3 para obtener una soluci´on X y la
cota superior U B
if U B < v∗ then

Actualizar v∗ ← U B, X ∗ ← X

end if
Seleccionar un par de instancias pi y pj y dividir el problema P en
dos subproblemas. Para cada problema actualizar T s, es y CL en
consecuencia, a˜nadir ambos problemas a Q e ir al Paso 5.

18: end while
19: Salida: Matriz de la soluci´on ´optima X ∗ y su valor de la funci´on objetivo

v∗

Cap´ıtulo 6

Descripci´on de la propuesta

A lo largo de este cap´ıtulo se presentar´an los diversos componentes que
se han ido incorporando al modelo, partiendo del algoritmo S-MDEClust.
[7].

6.1. Asignaci´on

Recordemos que, en el algoritmo original, se propone un m´etodo greedy
para asignar cada instancia a un cluster, una vez fijados los centroides de la
soluci´on. Como vimos en la Secci´on 5.1.2, este m´etodo consiste en agrupar
las instancias de acuerdo a las restricciones ML y luego asignar iterativa-
mente cada grupo de instancias al mejor cluster disponible. Se considera
como mejor aquel cluster que, sin incumplir ninguna restricci´on, minimiza
la suma de las distancias al cuadrado entre las instancias del grupo y el
centroide correspondiente. Si no existe ning´un cluster que satisfaga todas
las restricciones, se elige entonces el que minimice las distancias al cuadrado
sin tenerlas en cuenta.

No obstante, podr´ıa ser interesante no escoger siempre el mejor cluster,
para favorecer la exploraci´on de diferentes regiones del espacio de solucio-
nes. Teniendo en cuenta que la soluci´on obtenida se refinar´a posteriormente
mediante una b´usqueda local en la que s´ı se aplica la asignaci´on exacta
explicada en la Secci´on 5.1.2, puede resultar beneficioso introducir cierta
aleatoriedad en esta etapa inicial. Con esta motivaci´on, se han desarrolla-
do dos variantes del m´etodo greedy: la asignaci´on greedy aleatorizada y la
asignaci´on greedy aleatorizada con penalizaci´on.

29

Descripci´on de la propuesta

30

6.1.1. Asignaci´on greedy aleatorizada

En esta variante, se parte del mismo procedimiento que en el enfoque
greedy original: se agrupan las instancias seg´un las restricciones ML y se
identifican los clusters viables que no incumplen ninguna restricci´on. Sin
embargo, en lugar de asignar el grupo al cluster m´as cercano de forma de-
terminista, se introduce un componente aleatorio. Concretamente, para cada
cluster viable se calcula la suma de distancias al cuadrado entre su centroide
y las instancias del grupo que se desea asignar. Estas distancias se invierten
y se normalizan para construir una distribuci´on de probabilidad. El grupo
se asignar´a a un cluster de forma aleatoria seg´un esta distribuci´on. De esta
forma, los clusters m´as cercanos tendr´an m´as probabilidad de ser selecciona-
dos, aunque los clusters m´as lejanos tambi´en tendr´an opci´on de ser elegidos.
En el caso de que la asignaci´on del grupo de instancias a cualquiera de los
clusters provoque el incumplimiento de alguna restricci´on, se considerar´an
todos los clusters posibles, sin tener en cuenta las restricciones, y se aplicar´a
el mismo procedimiento probabil´ıstico para realizar la asignaci´on.

Algorithm 5 Asignaci´on Greedy Aleatorizada
Require: dataset D = {p1, . . . , pn} ⊂ Rd, n´umero de clusters K ∈ N, con-
juntos de restricciones must-link y cannot-link ML, CL ⊂ {1, . . . , n} ×
{1, . . . , n} , matriz de centroides ψS.

1: Definir GML = {G ⊂ {1, . . . , N } | ∀i, j ∈ G ,sujeto a i ̸= j, (i, j) ∈ ML}

2: Inicializar A = ∅
3: for all G ∈ GML do
4:

Construir la lista ksP osibles ← [k ∈ {1, . . . , K} | ∄( ˆG, k) ∈
A tal que ∃(ic, jc) ∈ CL, ic ∈ G ∧ jc ∈ ˆG]
if |ksP osibles| == 0 then
ksP osibles ← [1 . . . , K]

end if
Calcular la lista dist ← (cid:2)(cid:80)
Calcular la lista invDist ←

k ∥2 ∀ k ∈ ksP osibles(cid:3)

i∈G ∥pi − ψS
(cid:104) 1
d+ϵ ∀ d ∈ dist

(cid:105)

(cid:104)

(cid:105)
(cid:80) invDist ∀ x ∈ invDist

probs ←
Elegir aleatoriamente kG de ksP osibles seg´un la distribuci´on probs

x

5:

6:

7:

8:

9:

10:

11:
12: A = A ∪ {(G, kG)}
13:

for all i ∈ G do
ϕS
i = kG
end for

14:

15:
16: end for
17:
18: return ϕS

Descripci´on de la propuesta

31

El pseudoc´odigo de la asignaci´on greedy aleatorizada se muestra en el

Algoritmo 5. En el paso 9 se utiliza ϵ > 0 para evitar la divisi´on entre 0.

6.1.2. Asignaci´on greedy aleatorizada con penalizaci´on

Esta variante mantiene el esquema general del enfoque greedy aleatori-
zado, pero a˜nade una penalizaci´on adicional que desincentiva la asignaci´on
de grupos a clusters que incumplan restricciones. En lugar de determinar
qu´e clusters son viables, todos los clusters se consideran candidatos. Para
cada uno, se calcula la suma de distancias al cuadrado entre su centroide
y las instancias del grupo, y se a˜nade una penalizaci´on si existen conflictos
de restricciones CL con grupos ya asignados a ese cluster. Esta penalizaci´on
es la mitad de la media del rango de los valores del conjunto de datos en
cada dimensi´on, multiplicada por el n´umero de dimensiones. Las distancias
con la penalizaci´on a˜nadida se transforman en una distribuci´on de proba-
bilidad, invirti´endolas y normaliz´andolas, y se asigna el grupo a un cluster
seleccionado aleatoriamente seg´un esa distribuci´on.

Algorithm 6 Asignaci´on Greedy Aleatorizada con Penalizaci´on
Require: dataset D = {p1, . . . , pn} ⊂ Rd, n´umero de clusters K ∈ N,
restricciones must-link y cannot-link ML, CL, matriz de centroides ψS

1: Calcular penalizaci´on λ = media (m´ax(D) − m´ın(D)) · d/2
2: Definir GML = {G ⊂ {1, . . . , n} | ∀i, j ∈ G, i ̸= j ⇒ (i, j) ∈ ML}
3: Inicializar A = ∅
4: for all G ∈ GML do
5:

ksP osibles ← {1, . . . , K}
ksConf lictivos ← [k ∈ {1, . . . , K} | ∃( ˆG, k) ∈ A tal que ∃(ic, jc) ∈
CL, ic ∈ G ∧ jc ∈ ˆG]
Calcular la lista dist ← [(cid:80)
ksP osibles]
Calcular invDist ←
cero}
Calcular probs ←
Elegir aleatoriamente kG de ksP osibles seg´un la distribuci´on probs

(cid:105)
(cid:80) invDist ∀ x ∈ invDist

k ∥2 + λ · 1k∈ksConf lictivos ∀ k ∈

{ϵ > 0 para evitar divisi´on por

(cid:104) 1
c+ϵ ∀ c ∈ dist

i∈G ∥pi − ψS

(cid:105)

(cid:104)

x

6:

7:

8:

9:

10:
11: A = A ∪ {(G, kG)}
12:

for all i ∈ G do
ϕS
i = kG
end for

13:

14:
15: end for
16:
17: return ϕS

Descripci´on de la propuesta

32

El pseudoc´odigo de la asignaci´on greedy aleatorizada con penalizaci´on
se muestra en el algoritmo 6. Donde 1k∈conf lictivos es una funci´on booleana
que toma el valor 1 si se cumple la condici´on k ∈ conf lictivos y 0 en caso
contrario. As´ı, la penalizaci´on λ se a˜nade ´unicamente para los clusters con
conflictos.

6.2. B´usqueda Local

6.2.1. M´etodo Solis Wets

Con el objetivo de mejorar la calidad de las soluciones antes de aplicar la
b´usqueda local del algoritmo original (Subsecci´on 5.1.5), se ha introducido
una fase previa de b´usqueda local basada en el m´etodo de Solis Wets [30].
Para evitar una sobrecarga computacional excesiva y no perjudicar a la di-
versidad de la poblaci´on, esta estrategia se aplica ´unicamente a un individuo
en cada iteraci´on, el que se considere m´as prometedor. Para seleccionar a
dicho individuo se han considerado dos opciones:

Versi´on 1: se aplica sobre el mejor individuo de la poblaci´on, siempre
y cuando no se haya aplicado ya en iteraciones previas sobre dicho
individuo sin obtener mejora.

Versi´on 2: se aplica sobre el mejor individuo de la poblaci´on.

La b´usqueda local de Solis Wets es un algoritmo con un enfoque de
ascensi´on de colinas aleatorizado que mantiene en todo momento una ´unica
soluci´on en memoria: la mejor encontrada hasta el momento. Las soluciones
con las que trabaja el algoritmo son las matrices de centroides. En cada
iteraci´on del algoritmo, se parte de una soluci´on actual ψS y se genera una
diferencia aleatoria dif a partir de una distribuci´on normal centrada en un
array que llamaremos bias, de las mismas dimensiones que la soluci´on, es
decir, K × d; y una desviaci´on t´ıpica controlada por el par´ametro ρ. Se
eval´uan dos posibles nuevas soluciones: ψS + dif , y si no mejora, ψS − dif .
Si alguna de ellas mejora a la mejor soluci´on actual, evaluadas seg´un la
funci´on objetivo, se actualiza la mejor soluci´on y se registra un ´exito; en
caso contrario, se registra un fallo.

Para evaluar una soluci´on ψS, es necesario realizar primero un paso de
asignaci´on (Subsecci´on 5.1.2) para obtener el vector de pertenencia ϕS y as´ı
construir la soluci´on completa S = (ϕS, ψS), que puede evaluarse mediante la
funci´on objetivo presentada en la Ecuaci´on 5.1. Sin embargo, si no utilizamos
la asignaci´on exacta explicada en la Secci´on 5.1.2, existe la posibilidad de
que la soluci´on no satisfaga todas las restricciones.

Descripci´on de la propuesta

33

Para tratar de evitar que las nuevas soluciones incumplan un gran n´ume-
ro de restricciones, proponemos una funci´on objetivo penalizada, en la que
se introduce una penalizaci´on proporcional al n´umero de restricciones in-
cumplidas por la soluci´on. De este modo, se pueden considerar dos opciones
para la evaluaci´on de soluciones dentro del algoritmo de Solis Wets:

Usar la funci´on objetivo original (Ecuaci´on 5.1).

Utilizar una funci´on objetivo con penalizaci´on, definida como:

fp(S) = f (S) · (1 + δ · inf easibility)

(6.1)

donde f (S) es la funci´on objetivo original (sin penalizaci´on), δ es
una constante que controla el peso del t´ermino de penalizaci´on, e
inf easibility representa el tanto por uno de restricciones incumpli-
das respecto al total.

Por otro lado, como ya hemos mencionado brevemente, este m´etodo em-
plea dos par´ametros: bias y ρ que se van adaptando de forma din´amica y
permiten aprender progresivamente c´omo guiar la b´usqueda de manera m´as
eficiente. El par´ametro bias act´ua como estimaci´on de la direcci´on m´as pro-
metedora hacia el ´optimo local. Se inicializa a 0, ya que en un inicio no
disponemos de informaci´on acerca de cu´al es la mejor direcci´on de b´usque-
da y se actualiza tras cada ´exito o fallo. De este modo, el algoritmo va
aprendiendo progresivamente hacia d´onde conviene desplazar las soluciones,
favoreciendo la exploraci´on en direcciones m´as prometedoras. Por su par-
te, el par´ametro ρ se adapta en funci´on del comportamiento reciente de la
b´usqueda: tras varios ´exitos consecutivos, se incrementa ρ para explorar re-
giones m´as amplias y tras varios fallos consecutivos, se reduce para afinar
m´as la b´usqueda.

El pseudoc´odigo completo del m´etodo de Solis Wets se presenta en el

Algoritmo 7.

6.2.2. Selecci´on de individuos sobre los que se aplica la B´usque-

da Local

En la versi´on original del algoritmo, la b´usqueda local se aplica sobre
todos los individuos de la poblaci´on en cada iteraci´on. Sin embargo, este
enfoque resulta computacionalmente muy costoso, ya que cada b´usqueda
local requiere realizar m´ultiples veces el paso de asignaci´on exacta, el cual
implica resolver un problema de optimizaci´on para asignar las instancias a
los centroides. Esta operaci´on es uno de los cuellos de botella del algoritmo,
especialmente en problemas de gran tama˜no.

Descripci´on de la propuesta

34

Para reducir este coste computacional y a la vez favorecer la diversidad
de la poblaci´on, proponemos aplicar la b´usqueda ´unicamente sobre una frac-
ci´on de la poblaci´on. En concreto, se aplicar´a en cada iteraci´on al 10 % de las
mejores soluciones, con el fin de refinar aquellas soluciones m´as prometedo-
ras, y un 10 % adicional elegido aleatoriamente entre el resto de la poblaci´on,
para favorecer la diversidad de la poblaci´on y disminuir la probabilidad de
quedar estancados en ´optimos locales de forma prematura. Para el resto de
individuos no seleccionados, se omite la b´usqueda local y, en su lugar, se
realiza un ´unico paso de asignaci´on exacta como mecanismo de reparaci´on;
de esta forma, se garantiza que todas las soluciones de la poblaci´on cumplan
todas las restricciones.

Descripci´on de la propuesta

35

Algorithm 7 Algoritmo de Solis-Wets
Require: Matriz de centroides de la soluci´on inicial centers, n´umero m´axi-
mo de evaluaciones maxevals, ρ, dataset D, conjunto de restricciones
M L, CL, funci´on objetivo f obj

1: bias ← 0
2: Inicializar evals ← 0, f ailures ← 0, successes ← 0, num mejoras ← 0

3: labels ← asignar instancias de D a centers
4: f itness sol ← f obj(D, centers, labels, M L, CL)
5: while evals < maxevals and ρ > ϵ do
6:

dif ← N (bias, ρ)
newcenters ← centers + dif
newlabels ← asignar instancias de D a newcenters
f itness new ← f obj(D, newcenters, newlabels, M L, CL)
evals ← evals + 1
if f itness new < f itness sol then

num mejoras ← num mejoras + 1
f ailures ← 0, successes ← successes + 1
bias ← 0.4 · dif + 0.2 · bias
Actualizar centers, labels, f itness sol con newcenters, newlabels
y f itness new respectivamente

else if evals < maxevals then
newcenters ← centers − dif
newlabels ← asignar instancias de D a newcenters
f itness new ← f obj(D, newcenters, newlabels, M L, CL)
evals ← evals + 1
if f itness new < f itness sol then

num mejoras ← num mejoras + 1
f ailures ← 0, successes ← successes + 1
bias ← bias − 0.4 · dif
Actualizar
centers,
newlabels y f itness new respectivamente

labels,

f itness sol

con newcenters,

else

f ailures ← f ailures + 1
successes ← 0
bias ← 0.5 · bias

end if

end if
if successes ≥ 5 then

successes ← 0
ρ ← 2 · ρ

else if f ailures ≥ 3 then

f ailures ← 0
ρ ← ρ/2

7:

8:

9:

10:

11:

12:

13:

14:

15:

16:

17:

18:

19:

20:

21:

22:

23:

24:

25:

26:

27:

28:

29:

30:

31:

32:

33:

34:

35:

36:

37:

end if

38:
39: end while
40: return labels, centers

Descripci´on de la propuesta

36

Donde ϵ, en la l´ınea 5, es un n´umero positivo muy cercano a 0. Con esto se
pretende detener la b´usqueda cuando ρ alcanza valores muy peque˜nos, lo que
indica que las modificaciones en las soluciones son m´ınimas y el algoritmo
ha quedado estancado.

6.3. Evoluci´on Diferencial

6.3.1. SHADE

El operador de cruce del algoritmo original (Subsecci´on 5.1.3) no hace
uso del par´ametro CR. Este par´ametro, habitual en algoritmos de Evoluci´on
Diferencial, controla la probabilidad de aplicar el cruce a cada individuo de
la poblaci´on. En caso de no aplicar el cruce, el descendiente del individuo
ser´ıa ´el mismo. El algoritmo original tampoco contempla mecanismos de
adaptaci´on de par´ametros, lo que podr´ıa limitar su capacidad de explora-
ci´on. Con el objetivo de mejorar la eficacia del proceso evolutivo y guiar la
b´usqueda de manera m´as informada, haremos uso del mecanismo de adap-
taci´on de par´ametros SHADE (Success-History based Adaptive Differential
Evolution) [31].

La principal ventaja de SHADE es que ajusta din´amicamente los valores
de F y CR en cada iteraci´on, en funci´on del historial de configuraciones que
han dado buenos resultados en iteraciones anteriores. SHADE ha demostra-
do ser efectivo en una amplia variedad de problemas benchmark, superando
en muchos casos al rendimiento de algoritmos de Evoluci´on Diferencial sin
adaptaci´on de par´ametros y a otras estrategias de adaptaci´on din´amica de
par´ametros [31].

SHADE selecciona de forma din´amica los par´ametros CR y F para cada
individuo de la poblaci´on. Para ello, mantiene dos listas temporales, SCR y
SF , en las que se almacenan los valores de CR y F utilizados por aquellos
individuos que han logrado mejorar respecto a su predecesor en t´erminos de
la funci´on objetivo. Estas listas se reinician al comienzo de cada iteraci´on.
Por otro lado, se utilizan dos memorias de tama˜no H, una para CR y otra
para F , ambas inicializadas con valores iguales a 0.5 en todas sus posiciones.

En cada iteraci´on, los par´ametros CRi y Fi para un individuo Si se

generan del siguiente modo:

CRi = randn(MCR,ri, 0.1)

(6.2)

Fi = randc(MF,ri, 0.1)
donde randn(µ, σ2) representa un valor aleatorio tomado de una distribuci´on
normal con media µ y varianza σ2, y randc(µ, γ) representa un valor aleatorio

(6.3)

Descripci´on de la propuesta

37

obtenido de una distribuci´on de Cauchy con media µ y escala γ. El ´ındice ri ∈
{1, . . . , H} se selecciona aleatoriamente, y MCR,ri y MF,ri hacen referencia
al elemento ri-´esimo de las memorias MCR y MF , respectivamente. Adem´as,
si los valores generados para CRi o Fi se encuentran fuera del intervalo [0, 1],
se truncan a dicho rango.

Al final de cada iteraci´on, se actualiza la posici´on k de las memorias
MCR y MF utilizando la informaci´on almacenada en SCR y SF . El ´ındice
k se inicializa a 1 y se incrementa en cada actualizaci´on, reinici´andose a 1
cuando supera H. La actualizaci´on de la posici´on k para la iteraci´on i + 1
se realiza conforme a las siguientes expresiones:

MCR,k,i+1 =

(cid:40)

meanWA(SCR)
MCR,k,i

si SCR ̸= ∅
en caso contrario

MF,k,i+1 =

(cid:40)

meanWL(SF )
MF,k,i

si SF ̸= ∅
en caso contrario

(6.4)

(6.5)

donde meanWA(SCR) y meanWL(SF ) son medias ponderadas que se calculan
de acuerdo a las siguientes ecuaciones.

meanW A(SCR) =

|SCR|
(cid:88)

k=1

wk · SCR,k

wk =

∆fk
(cid:80)|SCR|
k=1 ∆fk

(6.6)

(6.7)

meanWL(SF ) =

(cid:80)|SF |

k=1 wk·S2
F,k
k=1 wk·SF,k

(cid:80)|SF |

donde ∆fk es la diferencia en valor absoluto entre el valor objetivo del

k-´esimo individuo de la soluci´on y su predecesor.

En el Algoritmo 8 se muestra el pseudoc´odigo completo del algoritmo
S-MDEClust con la incorporaci´on del m´etodo de adaptaci´on de par´ametros
SHADE que acabamos de describir.

Descripci´on de la propuesta

38

Algorithm 8 Algoritmo S-MDEC1ust con SHADE
Require: Dataset D = {p1, . . . , pn} ⊂ Rd, n´umero de clusters K, restric-
ciones ML, CL, tama˜no de poblaci´on P , n´umero m´aximo de iteraciones
NM AX , tolerancia tol.

1: Inicializar poblaci´on P0 = {S1,0, . . . , SP,0} aleatoriamente
2: Inicializar los H valores de MCR y MF a 0.5
3: Inicializar ´ındice k = 1
4: Determinar S⋆ tal que f (S⋆) ≤ f (S) para todo S ∈ P0
5: Inicializar n⋆
6: while (n⋆
7:

it < NM AX ) ∧ ((cid:80)P
s=1
Inicializar SCR = ∅, SF = ∅
for s = 1 to P do

it = 0

(cid:80)

¯s>s |f (Ss,G) − f (S¯s,G)| > tol) do

8:

9:

10:

11:

12:

13:

14:

15:

16:

17:

18:

19:

20:

21:

22:

23:

24:

25:

26:

27:

28:

29:

30:

31:

32:

33:

34:

Seleccionar ri ∈ [1, H] aleatoriamente
CRs = randn(MCR,ri, 0.1)
Fs = randc(MF,ri, 0.1)
if rand(0,1) < CRs then

Seleccionar aleatoriamente S1, S2, S3 ∈ P, todos diferentes entre
s´ı y distintos de Ss
Aplicar operador de cruce utilizando Fs con S1, S2, S3 para gene-
rar el descendiente Os

else

Os = Ss

end if
Aplicar o no el operador de mutaci´on a Os para obtener ˜Os, con un
probabilidad que disminuye a lo largo de las iteraciones
Aplicar b´usqueda local a ˜Os para obtener una soluci´on O′
s
if f (O′

s) < f (Ss) then

Ss = O′
s
A˜nadir CRs a SCR
A˜nadir Fs a SF
if f (O′

s) < f (S⋆) then

S⋆ = O′
s
n⋆
it = 0
else
it = n⋆
n⋆
end if

it + 1

end if
end for
if SCR ̸= ∅ y SF ̸= ∅ then

Actualizar MCR,k y MF,k de acuerdo a las ecuaciones 6.6 y 6.3.1
k = (k + 1) mod H

end if

35:
36: end while
37: return S⋆

Descripci´on de la propuesta

39

6.3.2. Cambios en el operador de cruce

En la versi´on original del algoritmo, el operador de cruce utilizado es
el cruce cl´asico de la Evoluci´on Diferencial, que genera un descendiente vi
a partir de tres individuos distintos mediante la expresi´on vi = xr1 + F ·
(xr2 − xr3), donde xr1, xr2 y xr3 son individuos aleatorios seleccionados de
la poblaci´on y F es el par´ametro que controla la amplitud de la perturbaci´on
(Subsecci´on 5.1.3). Este operador no incorpora informaci´on sobre la calidad
de los individuos, lo que podr´ıa limitar su capacidad para guiar la b´usqueda
de manera eficiente hacia regiones prometedoras del espacio de soluciones.

Con el objetivo de mejorar el rendimiento del algoritmo, proponemos
sustituir el cruce original por dos nuevas variantes basadas en la estrate-
gia current-to-pbest [32], que introducen un sesgo hacia individuos de alta
calidad.

Para ambas variantes, que llamaremos pbest1 y pbest2, el descendiente
vi de cada individuo xi de la poblaci´on se genera seg´un la siguiente expresi´on:

vi = xi + F · (xb − xi) + F · (xr1 − xr2)

(6.8)

donde xr1 y xr2 son individuos seleccionados de forma aleatoria de la pobla-
ci´on distintos de xi.

La diferencia entre las dos variantes radica en la elecci´on de xb.

Cruce pbest1: xb es un individuo aleatorio seleccionado entre el p %
de los mejores individuos de la poblaci´on en t´erminos del valor de la
funci´on objetivo.

Cruce pbest2: xb es el mejor individuo, seg´un el valor de la funci´on
objetivo, entre un p % aleatorio del total de la poblaci´on.

Estas variantes introducen una presi´on selectiva hacia soluciones de alta
calidad, pero conservando cierta diversidad mediante la aleatoriedad en la
elecci´on de xb.

6.4. Reinicio de la poblaci´on

Uno de los principales desaf´ıos al trabajar con algoritmos evolutivos es
la p´erdida de diversidad en la poblaci´on, que puede limitar la capacidad de
explorar el espacio de soluciones de forma efectiva y provocar una conver-
gencia prematura hacia ´optimos locales. Una forma com´un de abordar este
problema es introducir un mecanismo de reinicio en el algoritmo.

Descripci´on de la propuesta

40

El reinicio consiste en volver a generar parcial o totalmente la poblaci´on
cuando se detecta una situaci´on de estancamiento. De esta forma, se rein-
troduce diversidad a la poblaci´on, lo que permite al algoritmo escapar de
´optimos locales y continuar la b´usqueda hacia mejores soluciones.

A la hora de realizar el reinicio, es importante tener en cuenta varios

elementos que determinan cu´ando y c´omo se lleva a cabo:

Condici´on de activaci´on: el reinicio se activa tras un n´umero de-
terminado de iteraciones consecutivas sin mejora de la mejor soluci´on
de la poblaci´on. Esta condici´on busca detectar situaciones de estanca-
miento en la b´usqueda.

N´umero de reinicios: para evitar un uso excesivo de los reinicios, lo
que podr´ıa implicar un consumo excesivo de recursos computaciona-
les, se establece un n´umero de reinicios a lo largo de la ejecuci´on del
algoritmo.

Generaci´on de la nueva poblaci´on: tras el reinicio, se conserva la
mejor soluci´on encontrada hasta el momento, mientras que el resto de
la poblaci´on se genera de forma aleatoria, de la forma explicada en
Subsecci´on 5.1.6. Esto permite mantener la mejor soluci´on obtenida y,
al mismo tiempo, recuperar diversidad.

Tama˜no de la poblaci´on: se han considerado dos estrategias:

1. Mantener el tama˜no de la poblaci´on constante.

2. Disminuir el tama˜no de la poblaci´on a la mitad en cada reinicio:
se parte de un tama˜no de poblaci´on m´as grande, concretamente
el doble, y se disminuye el tama˜no a la mitad despu´es de cada
reinicio. Esto puede favorecer la diversidad en las primeras fases,
cuando la poblaci´on es m´as grande, y acelarar la convergencia en
etapas posteriores, cuando la poblaci´on es m´as peque˜na.

6.5. Otro enfoque: GRASP

Saliendo de la l´ınea de modificaciones sobre el algoritmo original, se ha
desarrollado un nuevo algoritmo basado en la estrategia GRASP (Greedy
Randomized Adaptive Search Procedure)

Los algoritmos GRASP son m´etodos multiarranque, en los que en cada
iteraci´on tienen dos fases: una fase de construcci´on de una soluci´on mediante
un procedimiento greedy aleatorizado, y una fase de mejora, en la que se
aplica una b´usqueda local sobre la soluci´on obtenida en la iteraci´on. Al
terminar el algoritmo, se devuelve la mejor soluci´on encontrada.

Descripci´on de la propuesta

41

Hay varios aspectos que se han tenido en cuenta al dise˜nar el algoritmo:

Criterio de parada: el algoritmo se detendr´a cuando se alcance un
n´umero m´aximo de iteraciones maxIter, o un n´umero m´aximo de itera-
ciones consecutivas sin mejoras en la mejor soluci´on encontrada N max.

Procedimiento para construir la soluci´on greedy aleatorizada:
debe ser capaz de generar soluciones iniciales razonables que sirvan
como punto de partida para la fase de mejora, con un componente
de aleatoriedad controlada para favorecer la exploraci´on del espacio
de soluciones. El pseudoc´odigo completo del algoritmo dise˜nado se
muestra en Algoritmo 9

Procedimiento de B´usqueda Local: se emplear´a el algoritmo BLPKMCC
[12], el mismo empleado en la b´usqueda local del algoritmo S-MDEClust,
ya que ha demostrado ser efectivo y garantiza el cumplimiento de todas
las restricciones.

Algorithm 9 Soluci´on Greedy Aleatorizada
Require: dataset D = {p1, . . . , pn} ⊂ Rd, n´umero de clusters K ∈ N,

conjuntos de restricciones ML, CL, funci´on objetivo f obj

1: Inicializar idxCenters ← ∅ {´ındices de las instancias elegidas como cen-

troides}

5:

4:

2: for c = 1 to K do
candidates ← ∅
3:
for all i ∈ {1, . . . , n} \ idxCenters do
idxCenters ← idxCenters ∪ {i}
asignar las instancias de D a idxCenters para obtener el vector de
pertenencia labels, teniendo en cuenta las restricciones ML y CL
totalDist ← f obj(D, idxCenters, labels)
A˜nadir (totalDist, i) a candidates
idxCenters ← idxCenters \ {i}

6:

9:

7:

8:

end for

10:
11: Ordenar candidates por totalDist de menor a mayor
12: RCL size ← m´ax(2, ⌊0.1 · |candidates|⌋)
13:

Elegir newCenter aleatoriamente de entre los RCL size primeros ele-
mentos de candidates
idxCenters ← idxCenters ∪ {newCenter}

14:
15: end for
16: centers ← D[idxCenters]
17:
18: return centers

Para construir la soluci´on greedy aleatorizada, se parte de un conjunto

Descripci´on de la propuesta

42

de centroides de la soluci´on vac´ıo, y en cada iteraci´on, una instancia del
conjunto de datos es elegida como nuevo centroide de un cluster. Para ello,
se crea una lista formada por los ´ındices de todas las instancias que a´un no
han sido seleccionadas, y se ordenan seg´un el valor de la funci´on objetivo
de la soluci´on que se obtendr´ıa al considerar esa instancia candidata como
centroide, adem´as de las que ya forman parte de la soluci´on. A partir de
esa lista, se crea la RLC (Restricted List of Candidates) con el 10 % de los
mejores candidatos. Por ´ultimo, se selecciona aleatoriamente una instancia
de esta RCL y se a˜nade al conjunto de centroides.

En cuanto a la l´ınea 6 del algoritmo, hay que tener en cuenta que no se
puede utilizar la asignaci´on exacta vista en la Secci´on 5.1.2, ya que el n´umero
de centroides de la soluci´on, excepto en la ´ultima iteraci´on, ser´a menor que
el n´umero de clusters K, debido a que la soluci´on se va construyendo paso
a paso y a´un no est´a completa. As´ı que, aunque exista una soluci´on factible
del problema para K clusters, podr´ıa no existir una soluci´on factible que
cumpla todas las restricciones para un n´umero de clusters menor que K.
Por esto, se debe usar un enfoque greedy para la asignaci´on, como el que
comentamos en la Secci´on 5.1.2.

Por otro lado, en la l´ınea 12 se establece el tama˜no de la RCL mediante
la expresi´on RCL size ← m´ax(2, ⌊0.1 · |candidates|⌋). De esta forma, se
garantiza que la RCL contenga al menos dos candidatos, incluso cuando
el n´umero total de candidatos sea peque˜no. Esto se hace para mantener la
aleatoriedad en la selecci´on.

Finalmente, en el Algoritmo 10, se muestra el pseudoc´odigo completo de

la propuesta GRASP.

Descripci´on de la propuesta

43

Algorithm 10 GRASP
Require: dataset D = {p1, . . . , pn} ⊂ Rd, n´umero de clusters K ∈ N, con-
juntos de restricciones ML, CL, n´umero m´aximo de iteraciones maxIter,
n´umero m´aximo de iteraciones consecutivas sin mejora N max

1: Inicializar contadores: nIter ← 0, nIterW OImpr ← 0
2: bestScore ← ∞
3: while nIterW OImpr < N max and nIter < maxIter do
centers ← SolGreedyAleatorizada(D, K, ML, CL, f obj)
4:
5: Aplicar b´usqueda local a centers para obtener el vector de pertenencia
labels, la matriz de centroides centers y el valor de la funci´on objetivo
de la soluci´on score
if bestScore > score then
bestLabels ← labels
bestCenters ← centers
bestScore ← score
nIterW OImpr ← 0

7:

8:

9:

6:

10:

else

nIterW OImpr ← nIterW OImpr + 1

11:

12:

13:

end if
nIter ← nIter + 1

14:
15: end while
16: return bestLabels, bestCenters

Cap´ıtulo 7

Dise˜no experimental

En este cap´ıtulo se detallan todos los aspectos relacionados con el dise˜no
experimental seguido. Al ser la mayor´ıa de nuestras propuestas modificacio-
nes sobre el algoritmo mem´etico S-MDEClust [7], los resultados obtenidos se
comparar´an directamente con los resultados proporcionados por la versi´on
original de dicho algoritmo. Sin embargo, como este algoritmo es aproxima-
do, no podemos saber qu´e tan pr´oximas est´an las soluciones obtenidas del
´optimo global. Para obtener una estimaci´on de esto, ejecutaremos tambi´en
el algoritmo exacto PC-SOS-SDP [8], explicado en el Cap´ıtulo 5.

7.1. Conjuntos de datos utilizados

Los conjuntos de datos utilizados en este trabajo son, en su mayor´ıa, los
mismos empleados en el art´ıculo en el que se presenta el algoritmo exacto
[8]. Adem´as, se han incorporado dos nuevos conjuntos de datos: Movement
Libras, para contar con un ejemplo con un n´umero elevado de clases, y
Toxicity, con el objetivo de incluir otro caso con alta dimensionalidad.

En total, son 14 conjuntos de datos reales de problemas de clasificaci´on,
con un n´umero de instancias que var´ıa entre 150 y 801, un n´umero de ca-
racter´ısticas entre 4 y 20531, y un n´umero de clases que oscila entre 2 y 15.
Esta variedad permite evaluar el rendimiento de las propuestas en escenarios
con diferentes grados de complejidad. En concreto, los conjuntos de datos
utilizados son:

1. Iris: conjunto cl´asico de clasificaci´on de flores en tres especies distintas

a partir de medidas de s´epalos y p´etalos.

2. Wine: contiene caracter´ısticas qu´ımicas de vinos para clasificarlos

seg´un su calidad.

44

Dise˜no experimental

45

3. Connectionist: se˜nales de sonar utilizadas para distinguir entre ecos
reflejados por cilindros met´alicos y por rocas con forma cil´ındrica.

4. Seeds: medidas de propiedades geom´etricas de granos pertenecientes

a tres variedades distintas de trigo.

5. Heart: diagn´ostico de enfermedades card´ıacas basado en medidas cl´ıni-

cas.

6. Vertebral : conjunto de datos con caracter´ısticas biomec´anicas usa-
das para clasificar pacientes ortop´edicos como normales o con alguna
patolog´ıa.

7. Computers: recoge series temporales de consumo el´ectrico en hogares
brit´anicos, registradas cada dos minutos durante 24 horas. El objetivo
es clasificar si el dispositivo es un ordenador de sobremesa o un port´atil.

8. Gene: conjunto de datos con expresiones gen´eticas de pacientes con

distintos tipos de tumores.

9. Movement Libras: contiene 15 clases de movimientos de mano en

LIBRAS, la lengua de signos brasile˜na.

10. Toxicity : propiedades moleculares para predecir la toxicidad de com-

puestos qu´ımicos.

11. ECG5000 : contiene latidos card´ıacos extra´ıdos de un paciente con
insuficiencia card´ıaca congestiva, con el objetivo de clasificar los latidos
en cinco categor´ıas.

12. Ecoli : conjunto de datos para predecir la localizaci´on subcelular de
la bacteria Escherichia coli (E. coli) en funci´on de propiedades fisico-
qu´ımicas.

13. Glass: conjunto de datos que clasifica muestras de vidrio en seis tipos

distintos, definidos seg´un su composici´on en ´oxidos.

14. Accent: conjunto de datos con grabaciones de palabras en ingl´es pro-
nunciadas por hablantes de seis pa´ıses distintos, utilizado para la de-
tecci´on y clasificaci´on de acentos.

Estos conjuntos de datos pueden ser encontrados en los repositorios UCI
[33] y UCR [34]. En la Tabla 7.1 se muestran las caracter´ısticas de cada uno
de ellos. En concreto, se detalla:

n: n´umero de instancias del conjunto de datos.

d : n´umero de atributos del conjunto de datos.

Dise˜no experimental

46

Dataset
Iris
Wine
Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent

n
150
178
208
210
299
310
500
801
360
171
500
336
214
329

d
4
13
60
7
12
6
720
20531
90
1203
140
7
9
12

k
3
3
2
3
2
2
2
5
15
2
5
8
6
6

r
50 y 100
50 y 100
50 y 100
50 y 100
100 y 150
100 y 150
150 y 250
200 y 400
100 y 150
50 y 100
150 y 250
100 y 150
50 y 100
100 y 150

Tabla 7.1: Informaci´on datasets

k : n´umero de clases del conjunto de datos.

r : n´umero de restricciones utilizado.

7.2. Conjunto de restricciones utilizados

Para ejecutar los algoritmos, tanto el exacto como el mem´etico y nues-
tras propuestas, es necesario disponer de los conjuntos de restricciones ML
y CL. En el caso de los conjunto de datos empleados en el art´ıculo que intro-
duce el algoritmo exacto [8], se utilizar´an exactamente los mismos conjuntos
de restricciones utilizados en dicho trabajo. Para los dos conjuntos de da-
tos adicionales, se generar´an restricciones siguiendo el mismo procedimiento
descrito en ese art´ıculo.

Para cada conjunto de datos con n instancias, se consideran seis con-
figuraciones distintas de restricciones: tres con un n´umero de restricciones
aproximadamente igual a n/2 y otros tres con n/4 restricciones. Dentro de
cada grupo se incluyen tres variantes: una con un 50 % de restricciones ML y
50 % de CL, otra con solo restricciones ML, y la ´ultima con solo restricciones
CL.

Para cada una de las seis configuraciones de restricciones descritas, se
generan cinco conjuntos diferentes de restricciones de forma aleatoria si-
guiendo el procedimiento cl´asico para generaci´on de restricciones descrito
en [3], lo que da lugar a un total de 30 variantes distintas para cada con-
junto de datos. Dicho procedimiento consiste en elegir un par de instancias

Dise˜no experimental

47

del conjunto de datos de forma aleatoria y generar una restricci´on ML si las
instancias pertenecen a la misma clase o una restricci´on CL si son de clases
distintas. El procedimiento contin´ua hasta que se haya generado el n´umero
deseado de restricciones ML y CL.

7.3. Detalles de la implementaci´on y par´ametros

de los algoritmos

Todas las pruebas experimentales se han realizado en un port´atil ASUS
TUF Dash F15, con un procesador Intel Core i7 12650H con 10 n´ucleos, 16
GB de RAM y Ubuntu 22.04.

7.3.1. Algoritmo exacto: PC-SOS-SDP

El algoritmo PC-SOS-SDP [8] est´a implementado en C++ y tiene algu-
nas rutinas implementadas en MATLAB. Para resolver la relajaci´on SDP,
se emplea el software SDPNAL+ de MATLAB [35]. Adem´as, recordemos
que para calcular la cota superior, es necesario resolver el problema de op-
timizaci´on formulado en la ecuaci´on 5.14; para ello, se hace uso de Gu-
robi [36]. El c´odigo fuente del algoritmo se puede encontrar en https:
//github.com/antoniosudoso/pc-sos-sdp.

Por otro lado, PC-SOS-SDP tiene varios par´ametros de configuraci´on.
Estos par´ametros no se deben especificar expl´ıcitamente en cada ejecuci´on,
sino que sus valores est´an recogidos en el archivo clustering c++/config.txt.
Para nuestras ejecuciones, se han mantenido todos los par´ametros por de-
fecto, excepto SDP SOLVER FOLDER, que se ha ajustado para indicar la ruta
local de instalaci´on de SDPNAL+.

Uno de estos par´ametros es BRANCH AND BOUND MAX NODES, que indica
el n´umero m´aximo de nodos en el ´arbol de b´usqueda del algoritmo. Este
par´ametro est´a establecido en 200, y una vez que se alcanza este n´umero
m´aximo, se detiene el algoritmo y se devuelve la mejor soluci´on encontra-
da. Otro par´ametro es BRANCH AND BOUND VISITING STRATEGY, que define
la estrategia utilizada para explorar el ´arbol de b´usqueda. En este caso, su
valor est´a fijado en 0, lo que implica que se utiliza la estrategia best-first, es
decir, se exploran primero los nodos m´as prometedores.

La totalidad de estos par´ametros, sus valores y su descripci´on se especi-

fican en la Tabla A.1 del Ap´endice.

Los par´ametros de ejecuci´on se detallan a continuaci´on, especificando el

orden en el que se deben proporcionar:

1. DATASET: ruta del conjunto de datos.

Dise˜no experimental

48

2. K: n´umero de clusters

3. CONSTRAINTS: ruta del archivo de restricciones.

4. LOG: ruta del archivo de log.

5. RESULT: ruta del archivo con la soluci´on encontrada.

7.3.2. Algoritmo mem´etico: S-MDEClust (versi´on original)

El algoritmo S-MDEClust est´a implementado enteramente en Python.
El c´odigo fuente de la versi´on original del algoritmo, es decir, sin ninguna
de las modificaciones explicadas en el Cap´ıtulo 6, se encuentra en https:
//github.com/pierlumanzu/s_mdeclust.

Para obtener los resultados de este algoritmo, se emplea la asignaci´on
greedy y se hace uso del operador de mutaci´on, lo cual se indica mediante los
correspondientes par´ametros de ejecuci´on. El resto de par´ametros se man-
tienen con sus valores por defecto, que son los recomendados en el art´ıculo
original donde se presenta el algoritmo [7]. En la Tabla 7.2 se detallan todos
los par´ametros del algoritmo, incluyendo el valor empleado en nuestra ejecu-
ci´on, su descripci´on y, cuando corresponde, los valores posibles que pueden
tomar.

7.3.3. Algoritmos propuestos

Se han implementado dos programas: uno que incorpora todas las modifi-
caciones sobre el algoritmo S-MDEClust presentadas en el cap´ıtulo anterior,
y otro que implementa el enfoque GRASP descrito en la Secci´on 6.5. Todo
el c´odigo fuente est´a disponible en https://github.com/teresaCL/TFG.

S-MDEClust modificado

Todos los par´ametros descritos en la subsecci´on anterior se siguen utili-
zando, y sus valores en las pruebas experimentales ser´an los mismos, salvo
que se indique lo contrario. Para incorporar las modificaciones propuestas
al algoritmo original, se han a˜nadido nuevos par´ametros y, en algunos casos,
ampliado los valores permitidos de ciertos par´ametros ya existentes.

Los nuevos par´ametros y los par´ametros modificados se muestran en la
Tabla 7.3, junto a su descripci´on y su valor por defecto, que ser´an los que se
utilicen para las pruebas experimentales a no ser que se indique lo contrario.

Dise˜no experimental

49

Par´ametro
--dataset
--constraints
--K
--seed
--verbose
--assignment

Valor
-
-
-
42
False
greedy

Descripci´on
Ruta del archivo con el conjunto de datos.
Ruta al archivo con las restricciones.
N´umero de clusters.
Semilla para el generador pseudoaleatorio.
Muestra el progreso del algoritmo si se especifica.
M´etodo de asignaci´on. Valores posibles:

exact: asignaci´on exacta.

greedy: asignaci´on greedy.

--mutation
--P
--Nmax

--max iter
--tol pop

True
20
5000

∞
10−4

--Nmax ls

1

--max iter ls ∞
--tol sol

10−6

Emplea el operador de mutaci´on si se especifica.
Tama˜no de la poblaci´on.
N´umero m´aximo de evaluaciones consecutivas sin
mejora de la mejor soluci´on.
N´umero m´aximo de iteraciones del algoritmo.
M´ınimo de diversidad de la poblaci´on antes de dete-
ner el algoritmo.
N´umero m´aximo de iteraciones consecutivas sin me-
jora en la b´usqueda local.
N´umero m´aximo de iteraciones en la b´usqueda local.
Tolerancia para comparar si una soluci´on es mejor
que otra.

--F

mdeclust Par´ametro F usado en el cruce. Valores posibles:

random: valor aleatorio en el intervalo (0, 2).

mdeclust: valor aleatorio en el intervalo (0.5,
0.8).

valor num´erico espec´ıfico en el intervalo (0, 2).

--alpha

0.5

Par´ametro α usado en la mutaci´on.

Tabla 7.2: Par´ametros de ejecuci´on del algoritmo S-MDEClust

GRASP

En la Tabla 7.4 se detallan los par´ametros utilizados en la ejecuci´on del
algoritmo GRASP, incluyendo su descripci´on y los valores empleados en los
experimentos.

7.4. M´etricas empleadas

Para evaluar el rendimiento de los algoritmos se han utilizado dos m´etri-

cas principales:

Dise˜no experimental

50

Score: valor de la funci´on objetivo del problema MSSC con restric-
ciones, presentada en la ecuaci´on 4.2.

Tiempo de ejecuci´on: tiempo necesario (en segundos) para com-
pletar la ejecuci´on del algoritmo. Esta m´etrica permite comparar la
eficiencia computacional de las distintas propuestas.

Adem´as, en el caso del algoritmo exacto, se considera una m´etrica adicio-
nal: Gap, que mide la diferencia relativa entre la cota superior y la inferior.

Gap % =

CS − CI
CS

· 100

(7.1)

donde CS hace referencia a la cota superior y CI a la cota inferior.
Cuanto menor sea el valor de esta m´etrica, m´as pr´oxima estar´a la soluci´on
del ´optimo global.

7.5. Librer´ıas y dependencias

La ejecuci´on del algoritmo S-MDEClust y de las propuestas desarrolladas
requiere de un entorno Anaconda configurado con las siguientes librer´ıas y
versiones:

Python 3.11.9

pip 24.0

NumPy 2.0.0

SciPy 1.14.0

pandas 2.2.2

gurobipy 11.0.2

NetworkX 3.3

Algunas partes del c´odigo hacen uso del solver Gurobi. Para su correcta
ejecuci´on es necesario tener instalado el Gurobi Optimizer, as´ı como dis-
poner de una licencia v´alida. En el caso de uso acad´emico, Gurobi ofrece
licencias gratuitas que pueden obtenerse desde su p´agina oficial [36].

Par´ametro

--assignment

Valor por
defecto
greedy

Descripci´on

M´etodo de asignaci´on. Valores posibles:

exact: asignaci´on exacta.

greedy: asignaci´on greedy.

greedy rand: asignaci´on greedy
aleatorizada.

greedy rand penalty:
ci´on greedy
penalizaci´on.

aleatorizada

asigna-
con

--F

mdeclust

Par´ametro F usado en el cruce. Valores
posibles:

random: valor aleatorio en el in-
tervalo (0, 2).

mdeclust: valor aleatorio en el
intervalo (0.5, 0.8).

half mdeclust: valor aleatorio
en el intervalo (0.25, 0.4).

valor num´erico espec´ıfico en el in-
tervalo (0, 2).

--crossover

original

Tipo de operador de cruce utilizado.
Valores posibles:

original: cruce original.

pbest v1: cruce pbest1.

pbest v2: cruce pbest2.

--solis

no

Indica si aplicar el algoritmo de Solis
Wets. Valores posibles:

no: no se aplica Solis Wets.

wo penalty: se aplica Solis Wets
usando funci´on objetivo sin pe-
nalizaci´on.

w penalty: se aplica Solis Wets
usando funci´on objetivo con pe-
nalizaci´on.

--apply LS all

True

--restart
--decrease pop size reset False

0

--shade

False

Indica si se debe de aplicar la b´usqueda
local a todos los individuos de la pobla-
ci´on o no
Indica el n´umero de reinicios
Indica si disminuir el tama˜no de la po-
blaci´on en cada reinicio o no
Indica si utilizar SHADE o no.

Tabla 7.3: Par´ametros de ejecuci´on del algoritmo S-MDEClust, incluyendo
las propuestas

Dise˜no experimental

52

Par´ametro
--dataset
--constraints
--K
--seed
--verbose
--assignment

Valor Descripci´on
-
-
-
42
False
greedy M´etodo de asignaci´on. Valores posibles:

Ruta del archivo con el conjunto de datos.
Ruta al archivo con las restricciones.
N´umero de clusters.
Semilla para el generador pseudoaleatorio.
Muestra el progreso del algoritmo si se especifica.

greedy: asignaci´on greedy.

greedy: asignaci´on greedy.

greedy rand: asignaci´on greedy aleatorizada.

--Nmax

--max iter
--Nmax ls

3

30
1

--max iter ls ∞
--tol sol

10−6

N´umero m´aximo de iteraciones consecutivas sin me-
jora.
N´umero m´aximo de iteraciones del algoritmo.
N´umero m´aximo de iteraciones consecutivas sin me-
jora en la b´usqueda local.
N´umero m´aximo de iteraciones en la b´usqueda local.
Tolerancia para comparar si una soluci´on es mejor
que otra.

Tabla 7.4: Par´ametros de ejecuci´on del algoritmo GRASP

Cap´ıtulo 8

Experimentaci´on y
resultados

En este cap´ıtulo se presentan los experimentos realizados para evaluar
el rendimiento de las propuestas desarrolladas. Se muestran los resultados
obtenidos, compar´andolos con los del algoritmo de referencia, con el objetivo
de analizar la efectividad de las modificaciones introducidas y justificar su
utilidad a trav´es de m´etricas objetivas.

8.1. Resultados del algoritmo referencia y del al-

goritmo exacto

El algoritmo de referencia es el algoritmo mem´etico S-MDEClust [7].
Recordamos que uno de los objetivos principales de este proyecto es mejorar
el rendimiento de dicho algoritmo. Sin embargo, dado que se trata de un
m´etodo aproximado, con el fin de obtener soluciones ´optimas que permitan
evaluar la calidad de los resultados, tambi´en se utilizar´a el algoritmo exacto
PC-SOS-SDP [8].

En la Tabla 8.1 se recogen los resultados del algoritmo exacto. Para cada
conjunto de datos, se muestran los valores medios de las 30 ejecuciones de la
funci´on objetivo (Score), el tiempo de ejecuci´on en segundos (Time) y del
Gap %, la m´etrica presentada en la Ecuaci´on 7.1. Cabe se˜nalar que faltan
los resultados de algunos conjuntos de datos, ya que no fue posible ejecutar
el algoritmo exacto sobre ellos debido a las elevadas demandas de recursos
computacionales.

Por otro lado, en la Tabla 8.2 se muestran los resultados obtenidos con
la versi´on original del algoritmo S-MDEClust, es decir, sin ninguna de las
modificaciones propuestas en el Cap´ıtulo 6. Al igual que antes, para cada

53

Experimentaci´on y resultados

54

Dataset
Iris
Wine
Connectionist
Seeds
Heart
Vertebral
Computers
Gene

Score
84.756
3421828.000
314.774
624.257
3347.195
559533.900
320716.600
17808460.000

Gap Time(s)
15.9
45.5
69.4
80.0
151.9
178.9
822.5
706.4

0.023 %
0.028 %
0.005 %
0.045 %
0.004 %
0.022 %
0.003 %
0.007 %

Tabla 8.1: Resultados algoritmo exacto

conjunto de datos se presentan los valores medios tanto de Score, como de
Time.

Dataset
Iris
Wine
Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent

Score Time(s)
1.7
84.756
2.2
3421827.432
2.7
314.774
2.5
624.257
6.8
3347.200
1.6
559533.960
12.9
320716.689
225.6
17808477.337
89.2
381.828
0.8
4.926e15
39.0
12734.756
39.9
15.627
13.9
90.755
28.7
29545.555

Tabla 8.2: Resultados algoritmo S-MDEClust original

Los resultados mostrados en la Tabla 8.2 se utilizar´an como referencia
para comparar el rendimiento de las propuestas desarrolladas. Por tanto, en
las tablas que se presentan a lo largo de este cap´ıtulo no se mostrar´an los
valores absolutos de Score y Time, sino su diferencia absoluta con respecto
a los resultados de referencia, es decir, la resta directa entre ambos valores
(Score propuesta - Score original). Cabe recordar que el MSSC con restric-
ciones es un problema de minimizaci´on, por lo que las diferencias negativas
indican un mejor desempe˜no de nuestra propuesta. Cuanto mayor es el valor
absoluto de dichas diferencias, mayor es la mejora obtenida.

Adem´as, en las tablas presentadas se incluir´a la media de estas diferen-
cias y las veces que mejora, se mantiene constante o empeora el Score en

Experimentaci´on y resultados

55

comparaci´on con el algoritmo de referencia.

Por otro lado, podemos ver que en la mayor´ıa de los casos las diferencias
de Score entre las soluciones obtenidas por el algoritmo S-MDClust original
y el exacto son de tan solo unas pocas d´ecimas o cent´esimas. En algunos con-
juntos de datos m´as sencillos, como Iris, Wine, Connectionist o Seeds, estas
diferencias incluso desaparecen por completo, lo que indica que el algoritmo
es capaz de alcanzar soluciones pr´acticamente ´optimas en dichos casos. Esto
indica que las mejoras en Score que pueden lograrse con las propuestas no
ser´an de gran magnitud. No obstante, esto no implica que dichas mejoras
carezcan de relevancia, especialmente si vienen acompa˜nadas de reducciones
en el tiempo de ejecuci´on.

8.2. Acr´onimos

Para simplificar la referencia a cada una de las propuestas realizadas, se
utilizar´an acr´onimos a lo largo del cap´ıtulo. A continuaci´on, se muestra la
lista con todos los acr´onimos empleados y su correspondiente descripci´on.

AGR-RAND: Asignaci´on Greedy aleatorizada (Subsecci´on 6.1.1).

AGR-RAND-P: Asignaci´on Greedy aleatorizada con penalizaci´on (Sub-
secci´on 6.1.2).

SHADE: Uso de la estrategia para adaptaci´on de par´ametros SHADE
(Subsecci´on 6.3.1).

PBEST1-F: Cruce pbest1 con F ∈ [0.5, 0.8].(Subsecci´on 6.3.2)

PBEST2-F: Cruce pbest2 con F ∈ [0.5, 0.8].(Subsecci´on 6.3.2)

PBEST1-F/2: Cruce pbest1 con F ∈ [0.25, 0.4].(Subsecci´on 6.3.2)

PBEST2-F/2: Cruce pbest2 con F ∈ [0.25, 0.4].(Subsecci´on 6.3.2)

SW-V1-WO-PEN: Versi´on 1 del Algoritmo Solis Wets usando la fun-
ci´on objetivo sin penalizaci´on. (Subsecci´on 6.2.1)

SW-WO-PEN o SW: Versi´on 2 del Algoritmo Solis Wets usando la
funci´on objetivo sin penalizaci´on. (Subsecci´on 6.2.1)

SW-W-PEN: Versi´on 2 del Algoritmo Solis Wets usando la funci´on
objetivo con penalizaci´on. (Subsecci´on 6.2.1)

SEL-BL: Elegir a qu´e individuos de la poblaci´on aplicar la b´usqueda
local, seg´un lo explicado en la Subsecci´on 6.2.2

Experimentaci´on y resultados

56

R-XIT-Y : Aplicar la estrategia de reinicio detallada en la Secci´on 6.4
manteniendo el tama˜no de la poblaci´on constante. X e Y toman valores
num´ericos: X indica el m´aximo de iteraciones sin mejora antes de
activar el reinicio, mientras que Y especifica el n´umero de reinicios.
Por ejemplo, R-3IT-2 indica que el m´aximo de iteraciones seguidas sin
mejora es 3 y que se realizan 2 reinicios.

R-XIT-Y -DIS: similar al anterior. La ´unica diferencia es que se dismi-
nuye el tama˜no de la poblaci´on despu´es de cada reinicio.

GRASP: enfoque GRASP propuesto en la Secci´on 6.5

Para los algoritmos que combinan varios componentes, se encadenar´an

los acr´onimos de los componentes involucrados.

8.3. Enfoque GRASP

En la Tabla 8.3 se muestran los resultados de la propuesta GRASP,

presentada en la Secci´on 6.5.

Dataset
Iris

+0.025
Wine +24461.053
+0.041
+0.008
+1.681
+10.438
+134.362
+101.498
+6.344
-6.710e+12
+130.177
+0.172
+2.320
+93.915
-4.793e+11
1
0
13

Score Time (s)
-0.3
-0.1
-0.6
+0.5
-1.6
+2.0
+2.0
-68.1
+72.2
+0.5
+18.6
+4.4
-0.8
+1.6
+2.2
6
0
8

Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

Tabla 8.3: Resultados GRASP

Se observa que, aunque el promedio del Score sea menor que cero, esto
se debe ´unicamente a la mejora obtenida en el conjunto de datos Toxicity.

Experimentaci´on y resultados

57

Sin embargo, en el resto de los conjuntos de datos se aprecia un claro em-
peoramiento de la calidad de las soluciones en t´erminos de Score. Adem´as,
el tiempo de ejecuci´on promedio tambi´en empeora con respecto al algoritmo
S-MDEClust.

8.4. Variantes de la asignaci´on greedy

Las primeras modificaciones sobre el algoritmo S-MDEClust propuestas
fueron acerca de la asignaci´on greedy. En concreto, se plantearon dos varian-
tes: la asignaci´on greedy aleatorizada y la asignaci´on greedy aleatorizada con
penalizaci´on, descritas en la Subsecci´on 6.1.1 y Subsecci´on 6.1.2, respectiva-
mente. Estas modificaciones se introdujeron con el objetivo de fomentar una
mayor exploraci´on del espacio de soluciones al a˜nadir aleatoriedad durante
el proceso de asignaci´on.

En la Tabla 8.4 se muestran los resultados del algoritmo con la asignaci´on
greedy aleatorizada y en la Tabla 8.5 con la asignaci´on greedy aleatorizada
con penalizaci´on.

Movement Libras

Dataset

Computers

Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.002
Heart
Vertebral +0.000
-0.017
Gene +0.000
-0.069
Toxicity +0.000
ECG5000 +0.023
Ecoli +0.000
-0.021
Glass
Accent +0.501
Promedio +0.030
4
8
2

Score Time (s)
+0.0
+0.1
+0.4
+0.1
+0.5
+0.1
+1.4
-16.6
-0.4
+0.1
+1.6
-1.4
+1.4
+0.1
-0.9
3
1
10

Veces mejor
Veces igual
Veces peor

Tabla 8.4: Resultados AGR-RAND

Experimentaci´on y resultados

58

Dataset
Iris

+0.000
Wine +39.893
+0.000
+0.000
-0.004
+0.000
-0.017
+0.000
+0.030
+0.000
-0.044
+0.002
-0.035
-0.070
+2.840
5
6
3

Score Time (s)
+0.0
+0.1
+0.4
+0.2
+0.0
+0.1
+1.4
-15.5
-2.2
+0.1
+1.6
+3.2
+3.5
+2.9
-0.3
2
2
10

Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

Tabla 8.5: Resultados AGR-RAND-P

Si comparamos las dos tablas anteriores, podemos ver que el rendimiento
del algoritmo con asignaci´on greedy aleatorizada (AGR-RAND) parece lige-
ramente superior con respecto a la versi´on con penalizaci´on (AGR-RAND-
P).

En t´erminos de Score, aunque AGR-RAND mejora una vez menos que la
versi´on con penalizaci´on, tambi´en empeora una vez menos. Adem´as, cuando
se producen empeoramientos, estos son menos acusados en la versi´on sin
penalizaci´on. Por otro lado, aunque en ambas se consigue una reducci´on
del promedio del tiempo de ejecuci´on, con AGR-RAND esta reducci´on es
m´as pronunciada, consiguiendo mejorar en los tres conjuntos de datos en
los que el algoritmo original tardaba m´as: Gene, Movement Libras y Ecoli.
Adem´as, aunque el tiempo de ejecuci´on empeora m´as veces de las que mejo-
ra, estos incrementos son en la mayor´ıa de los casos de unas pocas d´ecimas
de segundo.

Por lo anterior, si tuvi´eramos que elegir entre una de las dos variantes,
optar´ıamos por AGR-RAND. Sin embargo, no est´a claro si supone o no una
mejora respecto al algoritmo original: el tiempo s´ı se reduce en promedio,
pero el Score, aunque mejore en m´as ocasiones de las que empeora, estas
mejoras son muy peque˜nas.

Experimentaci´on y resultados

59

8.5. Evoluci´on Diferencial

8.5.1. SHADE

En la Subsecci´on 6.3.1 describimos la estrategia de adaptaci´on de par´ame-
tros SHADE, que adapta los par´ametros F y CR en funci´on de los valores
de estos que dieron buenos resultados en iteraciones anteriores. Esta mo-
dificaci´on se introdujo con el objetivo de guiar la b´usqueda de forma m´as
informada, con la esperanza de mejorar los resultados obtenidos.

En la Tabla 8.6 se muestran los resultados obtenidos al a˜nadir esta es-

trategia al algoritmo original.

Como podemos ver, el Score de las soluciones obtenidas solo empeora
respecto al algoritmo original en dos conjuntos de datos, mientras que me-
jora en cinco. Adem´as, tambi´en se obtiene una reducci´on promedio en el
tiempo de ejeucuci´on, destancando especialmente la reducci´on obtenida en
el conjunto de datos Gene, que es el que m´as tiempo de ejecuci´on requer´ıa
con la versi´on original del algoritmo. Sin embargo, los empeoramientos en
t´erminos de Score son, en general, de mayor magnitud que las mejoras, lo
que provoca que el promedio de las diferencias del Score sea muy cercano a
cero.

Dataset

Computers

Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.149
Gene +0.000
Movement Libras +0.129
Toxicity +0.000
-0.084
-0.002
-0.006
Accent +0.091
-0.002
5
7
2

Score Time (s)
+0.2
-0.1
+0.2
+0.0
-0.4
-0.1
+1.6
-24.2
+4.6
+0.1
-1.7
-0.8
+0.9
-1.4
-1.5
7
1
6

Promedio
Veces mejor
Veces igual
Veces peor

ECG5000
Ecoli
Glass

Tabla 8.6: Resultados SHADE

Experimentaci´on y resultados

60

8.5.2. Cambios en el operador de cruce

Se propusieron dos nuevos operadores de cruce en la Subsecci´on 6.3.2:
pbest1 y pbest2, ambos basados en la estrategia de cruce current-to-pbest.
Con estas dos propuestas, se pretend´ıa equilibrar explotaci´on y exploraci´on,
guiando la b´usqueda hacia soluciones prometedoras sin perder diversidad en
la poblaci´on.

Hasta ahora, excepto al utilizar la estrategia de adaptaci´on de par´ame-
tros SHADE, siempre se generaba el par´ametro de cruce F aleatoriamente
en cada iteraci´on dentro del rango (0.5, 0.8), ya que este rango de valores es
el considerado como ´optimo seg´un los autores del algoritmo S-MDEClust.
Sin embargo, con los dos nuevos operadores de cruce se han realizado prue-
bas tanto escogiendo el par´ametro F en el rango mencionado, como en el
rango (0.25, 0.4), es decir, reduciendo F a la mitad. Esto se debe a la forma
en la que se genera la nueva soluci´on con el cruce pbest1 y pbest2: como
se observa en la Ecuaci´on 6.8, se suma al individuo actual dos diferencias
distintas, ambas escaladas por F . Esto implica un desplazamiento poten-
cialmente mayor respecto al cruce original (Ecuaci´on 5.3), en el que solo se
suma una diferencia. Para compensar esto, es por lo que se reduce F a la
mitad.

Por otro lado, como vimos anteriormente, el algoritmo S-MDEClust tiene

tres criterios de parada, que vienen configurados por tres par´ametros:

1. Nmax: n´umero m´aximo de evaluaciones consecutivas sin mejora.

2. max iter: n´umero m´aximo de iteraciones del algoritmo.

3. tol pop: m´ınimo de diversidad de la poblaci´on. Esta diversidad, como
vimos en el algoritmo 1, se mide como la suma de las diferencias del
valor de la funci´on objetivo para cada par de soluciones de la poblaci´on.
Cuando se alcanza este umbral se detiene el algoritmo.

De estos tres criterios de parada, el ´unico que realmente ha sido utilizado
hasta ahora es el tercero. El primero no se activa porque, por defecto, el valor
de max iter est´a fijado en infinito, y el segundo (Nmax) tambi´en es demasiado
alto: est´a establecido en 5000 evaluaciones, lo que para una poblaci´on de 20
individuos, equivale a 250 iteraciones completas sin mejora. En todas las
ejecuciones que se han realizado hasta el momento, el algoritmo siempre se
detiene porque la poblaci´on alcanza el umbral m´ınimo de diversidad. Sin
embargo, al ejecutar el algoritmo con los dos nuevos cruces, especialmente
las versiones en las que se reduce F a la mitad, la diversidad decrece m´as
despacio y cuanto m´as cercana a 0 es, m´as se ralentiza su decrecimiento,
como podemos ver en la Figura 8.1. Esto hace que no se alcance el umbral de
diversidad m´ınima y que el algoritmo llegue a las 250 iteraciones completas

Experimentaci´on y resultados

61

(a) Algoritmo S-MDEClust

(b) Algoritmo PBEST1-F/2

Figura 8.1: Evoluci´on de la diversidad

sin mejora antes de detenerse, lo que aumenta much´ısimo los tiempos de
ejecuci´on. Sin embargo, hemos observado que despu´es de 5 iteraciones sin
mejora, es poco frecuente que la mejor soluci´on vuelva a mejorar. Por lo
cual, fijaremos el par´ametro Nmax a 5 · 20 = 100. Este valor se mantendr´a
para todas las ejecuciones en las que se use el cruce pbest1 o pbest2, a no
ser que se indique lo contrario.

El motivo de este ralentizamiento de la disminuci´on de la diversidad
puede ser en parte porque, al estar guiados por los mejores individuos de la
poblaci´on, tienden a explorar de forma m´as localizada en torno a las regio-
nes prometedoras. Esto evita que toda la poblaci´on converja r´apidamente
a un mismo punto, estabilizando la diversidad en un valor bajo pero no
nulo, suficiente para impedir que se active el criterio de parada basado en
diversidad.

El la Tabla 8.7 se presentan los resultados del algoritmo con el cruce

pbest1 y pbest2, con el par´ametro F en el rango (0.5, 0.8).

Como podemos ver, aunque con el cruce pbest2 se mejore el Score en dos
conjuntos de datos m´as y empeore en uno menos, el promedio de las diferen-
cias del Score es mayor que cero. Al analizar los casos en los que mejora el
resultado con el pbest2, se observa que dichas mejoras suelen ser de menor
magnitud en comparaci´on con las obtenidas con el cruce pbest1. Adem´as, en
el ´unico conjunto de datos en el que empeora el Score con pbest2 (ECG500 )

Experimentaci´on y resultados

62

Dataset

PBEST1-F
Score Time (s)

Movement Libras

Computers

Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.154
Gene +0.000
-0.303
Toxicity +0.000
ECG5000 +0.020
Ecoli +0.003
Glass +0.000
-0.025
-0.033
4
8
2

Accent
Promedio
Veces mejor
Veces igual
Veces peor

PBEST2-F
Score Time (s)
+0.0
+0.0
+0.0
+0.2
-0.6
+0.1
-2.3
-20.0
-31.6
+0.1
-9.5
-13.0
-2.8
-3.0
-5.9
8
3
3

+0.2 +0.000
+0.2 +0.000
+0.2 +0.000
+0.1 +0.000
+0.0
-0.004
+0.2 +0.000
-1.1
-0.094
-6.2 +0.000
-0.027
-21.3
+0.3 +0.000
-8.1 +0.241
-0.004
-4.7
-0.009
-1.9
-0.1
-0.050
-3.0 +0.004
6
7
1

7
1
6

Tabla 8.7: Comparativa PBEST1-F y PBEST2-F

podemos ver que este empeoramiento es considerablemente mayor que el
observado con pbest1. Por esto, podemos decir que con el cruce pbest1 se
obtienen mejores resultados que con el cruce pbest2.

Adem´as, en este caso s´ı que se puede apreciar que con el algoritmo
PBEST1-F se consigue mejorar con respecto al algoritmo original, tanto
en el Score como ya hemos comentado, pero tambi´en en el tiempo. Este se
reduce en promedio en 3 segundos, siendo la mejora especialmente notable
(de hasta 21 segundos) en los conjuntos de datos en los que el algoritmo
original presentaba mayores tiempos de ejecuci´on, como Movement Libras
y Gene.

Por otro lado, en la Tabla 8.8 se muestran los resultados correspondientes
al uso de los cruces pbest1 y pbest2, pero generando el par´ametro F en el
rango (0.25, 0.4).

Podemos observar que PBEST2-F/2 proporciona mejores resultados que
PBEST1-F/2 tanto en t´erminos de Score, como en tiempo de ejecuci´on. Sin
embargo, parece que PBEST1-F sigue siendo superior a PBEST2-F/2: aun-
que el Score con el primer algoritmo mejore en un conjunto de datos menos
que con el segundo, en promedio, PBEST1-F consigue encontrar soluciones
con mejor Score. Adem´as, con PBEST2-F/2 la reducci´on del tiempo pro-

Experimentaci´on y resultados

63

Dataset

PBEST1-F/2
Score Time (s)

Movement Libras

Computers

Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.203
Gene +0.000
-0.177
Toxicity +0.000
ECG5000 +0.164
Ecoli
-0.002
Glass +0.034
-0.057
-0.018
5
7
2

Accent
Promedio
Veces mejor
Veces igual
Veces peor

PBEST2-F/2
Score Time (s)
+0.2
+0.4
+0.5
+0.4
-0.3
+0.3
-0.2
+8.5
-9.8
+0.3
-1.4
-5.9
-0.2
+0.9
-0.4
6
0
8

+0.4 +0.000
+0.4 +0.000
+0.5 +0.000
+0.3 +0.000
+0.0
-0.004
+0.3 +0.000
-0.203
-0.3
+13.6 +0.000
-0.024
-8.7
+0.4 +0.000
-3.6
-0.085
-2.2 +0.001
-1.6 +0.007
-0.070
+4.4
-0.027
+0.3
5
5
7
1
2
8

Tabla 8.8: Comparativa PBEST1-F/2 y PBEST2-F/2

medio de ejecuci´on es muy limitada, de apenas 0.4 segundos, y en el peor
caso el tiempo empeora en m´as de 8 segundos. En cambio, con PBEST1-F la
reducci´on promedio es de 3 segundos, y el mayor empeoramiento observado
no supera las 0.3 d´ecimas de segundo. Esta diferencia en tiempos se debe a
que al utilizar un valor de F menor en PBEST2-F/2, las modificaciones in-
troducidas al generar nuevas soluciones mediante el cruce son m´as peque˜nas,
por lo que el algoritmo tarda m´as en converger.

8.6. B´usqueda Local

8.6.1. Uso del algoritmo Solis Wets

El algoritmo Solis Wets, como comentamos en la Subsecci´on 6.2.1, se
aplica a un ´unico individuo en cada iteraci´on, justo antes de la fase de
b´usqueda local, con el objetivo de refinar la soluci´on m´as prometedora antes
de dicha fase. Recordamos que este algoritmo emplea dos par´ametros que
se adaptan din´amicamente: bias, que estima la direcci´on hasta el ´optimo
local y se inicializa a cero, pues a priori no disponemos de informaci´on
acerca de d´onde se encuentra el ´optimo; y el par´ametro ρ, que regula la

Experimentaci´on y resultados

64

magnitud de los desplazamientos aleatorios que se aplican a la soluci´on en
cada paso. Un valor com´unmente utilizado para inicializar ρ es 0.1, pero
como en nuestro caso los datos no est´an normalizados, este valor fijo puede
resultar inadecuado. Por ello, el valor de ρ incial se calcular´a como el 10 %
del rango promedio de los atributos del conjunto de datos. Esta forma de
inicializaci´on permite ajustar ρ a la escala del problema.

En la Tabla 8.9 se muestran los resultados del algoritmo SW-V1-WO-
PEN junto con los resultados de la mejor propuesta hasta el momento:
PBEST1-F.

Dataset

PBEST1-F
Score Time (s)

Movement Libras

Computers

Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.154
Gene +0.000
-0.303
Toxicity +0.000
ECG5000 +0.020
Ecoli +0.003
Glass +0.000
-0.025
-0.033
4
8
2

Accent
Promedio
Veces mejor
Veces igual
Veces peor

SW-V1-WO-PEN
Score Time (s)
-0.1
+0.0
+0.4
+0.3
+0.9
+0.0
+1.0
+26.7
+2.7
+0.1
+2.0
-0.1
+2.2
+4.1
+2.9
2
2
10

+0.2 +0.000
+0.2 +0.000
+0.2 +0.000
+0.1 +0.000
+0.0
-0.004
+0.2 +0.000
-0.154
-1.1
-6.2 +0.000
-21.3 +0.041
+0.3 +0.000
-8.1 +0.689
-4.7 +0.001
-0.005
-1.9
-0.069
-0.1
-3.0 +0.036
4
7
3

7
1
6

Tabla 8.9: Comparativa PBEST1-F y SW-V1-WO-PEN

Por otro lado, en la Tabla 8.10 se presentan los resultados del algoritmo

SW-WO-PEN comparados tambi´en con los de PBEST1-F.

Como podemos ver, tanto en el algoritmo SW-V1-WO-PEN como en
el SW-WO-PEN, la diferencia se Score promedio es mayor que cero. Sin
embargo, esta diferencia es considerablemente mayor en el segundo caso, de-
bido principalmente al empeoramiento del Score obtenido para el conjunto
de datos Wine con el algoritmo SW-WO-PEN. Cabe se˜nalar que el valor
absoluto del Score en este conjunto de datos es muy alto (m´as de tres millo-
nes), por lo que el impacto relativo de dicho empeoramiento es en realidad
menor de lo que sugiere el valor bruto. Adem´as, en otros conjuntos de da-

Experimentaci´on y resultados

65

Dataset

Computers

Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.154
Gene +0.000
-0.303
Toxicity +0.000
ECG5000 +0.020
Ecoli +0.003
Glass +0.000
-0.025
-0.033
4
8
2

PBEST1-F
Score Time (s)
+0.2
+0.000
+0.2 +39.893
+0.000
+0.2
+0.000
+0.1
-0.004
+0.0
+0.000
+0.2
-0.203
-1.1
+0.000
-6.2
-0.003
-21.3
+0.000
+0.3
-0.060
-8.1
+0.001
-4.7
+0.036
-1.9
+0.308
-0.1
+2.855
-3.0
4
7
6
1
4
6

SW-WO-PEN
Score Time (s)
-0.1
+0.0
+0.6
+0.2
+0.7
+0.0
+2.0
+14.3
+2.9
+0.1
+2.9
-1.0
+2.2
+4.9
+2.1
2
2
10

Accent
Promedio
Veces mejor
Veces igual
Veces peor

Movement Libras

Tabla 8.10: Comparativa PBEST1-F y SW-WO-PEN

tos como Computers la mejora es m´as grande con SW-WO-PEN que con
SW-V1-WO-PEN, por lo que el rendimiento de ambos algoritmos en cuanto
al Score realmente est´a m´as igualado de lo que puede parecer en un primer
momento mirando las tablas.

Donde s´ı se aprecia una diferencia m´as clara es en el tiempo de ejecuci´on:
SW-WO-PEN presenta un tiempo promedio inferior, y el empeoramiento en
el peor caso se limita a 14 segundos, frente a casi 27 segundos en el caso de
SW-V1-WO-PEN.

De cualquier caso, queda claro que el rendimiento de estos dos algoritmos

no supera al de la mejor propuesta hasta ahora: PBEST1-F.

En la tabla Tabla 8.11 se muestran los resultados del algoritmo SW-W-
PEN, que es equivalente a SW-WO-PEN pero utilizando la funci´on objetivo
con penalizaci´on. Una vez m´as, los resultados se comparan con los obtenidos
por la mejor propuesta hasta el momento: PBEST1-F.

En este caso, se ve claramente que la modificaci´on no ha tenido un im-
pacto positivo. Aunque el tiempo, en promedio, se reduce levemente, los
resultados en cuanto al Score empeoran notablemente: solo mejora en dos
de los 14 conjuntos de datos y empeora en seis de ellos.

Experimentaci´on y resultados

66

Dataset

Computers

Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.154
Gene +0.000
-0.303
Toxicity +0.000
ECG5000 +0.020
Ecoli +0.003
Glass +0.000
-0.025
-0.033
4
8
2

PBEST1-F
Score Time (s)
+0.2
+0.000
+0.2 +39.893
+0.000
+0.2
+0.000
+0.1
-0.004
+0.0
+0.000
+0.2
-0.203
-1.1
+0.000
-6.2
+0.078
-21.3
+0.000
+0.3
+0.847
-8.1
+0.003
-4.7
+0.059
-1.9
+0.308
-0.1
+2.927
-3.0
2
7
6
1
6
6

SW-W-PEN
Score Time (s)
+0.1
+0.0
+0.1
+0.3
+0.6
+0.1
-1.0
+12.1
-9.7
+0.3
-5.4
-9.1
-0.9
+5.3
-0.5
5
1
8

Accent
Promedio
Veces mejor
Veces igual
Veces peor

Movement Libras

Tabla 8.11: Comparativa PBEST1-F y SW-W-PEN

8.6.2. Combinando el uso de varias propuestas

Ahora vamos a estudiar qu´e sucede al combinar dos de las propuestas
vistas: SW-WO-PEN y AGR-RAND, compar´andolo con el resultado de SW-
WO-PEN. Los resultados se muestran en la Tabla 8.12

Como podemos ver, al a˜nadir la asignaci´on greedy aleatorizada se obtie-
nen mejores resultados en t´erminos de Score. Podemos observar que en el
conjunto de datos Wine ya no empeora el Score como suced´ıa con el algorit-
mo SW-WO-PEN, y en otros conjuntos de datos como el Movement Libras
se consigue una mejora significativa. Tambi´en es cierto que el tiempo de
ejecuci´on aumenta y hay en algunos casos en los que el Score empeora con
respecto al algoritmo SW-WO-PEN, aunque en promedio s´ı que se consigue
una mejora del Score.

Experimentaci´on y resultados

67

SW-WO-PEN
Score Time (s)

Dataset
Iris

+0.000
Wine +39.893
+0.000
+0.000
-0.004
+0.000
-0.203
+0.000
-0.003
+0.000
-0.060
+0.001
+0.036
+0.308
+3.074
4
5
4

Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

SW-WO-PEN-AGR-RAND
Time (s)
+0.3
+0.6
+1.0
+0.8
+1.3
+0.5
+4.2
+29.2
+15.5
+0.4
+5.6
+4.9
+4.0
+4.9
+3.4
0
0
10

Score
-0.1 +0.000
+0.0 +0.000
+0.6 +0.000
+0.2 +0.000
+0.7
-0.004
+0.0 +0.000
-0.154
+2.0
+14.3 +0.000
-0.328
+2.9
+0.1 +0.000
+2.9 +0.058
-1.0 +0.003
-0.031
+2.2
+4.9 +0.064
-0.030
+1.2
4
2
7
2
3
10

Tabla 8.12: Comparativa SW-WO-PEN y SW-WO-PEN-AGR-RAND

A continuaci´on, en la Tabla 8.13, se muestran los resultados de la mejor
propuesta hasta ahora: PBEST1-F junto con los resultados obtenidos com-
binando este operador de cruce y la b´usqueda local Solis Wets (PBEST1-F-
SW-WO-PEN).

Como se puede observar, al incorporar el uso del algoritmo de Solis Wets
a la versi´on con el operador de cruce pbest1, el rendimiento general se ve
ligeramente reducido en comparaci´on con la versi´on que no lo incluye. La
reducci´on en el tiempo promedio de ejecuci´on es menor. Adem´as, la mejora
del Score promedio tambi´en disminuye, y es que aunque en dos conjuntos
de datos (ECG5000 y Ecoli ) se mejora el Score con respecto al algoritmo
PBEST1-F, hay tres casos en los que dicho valor empeora (Glass, Accent y
Movement Libras).

Ahora vamos a analizar los resultados obtenidos al combinar el nuevo
operador de cruce pbest1 con la asignaci´on greedy aleatorizada, con y sin el
uso del algoritmo Solis Wets. Los resultados se muestran en la Tabla 8.14.

Podemos observar que las soluciones obtenidas por el algoritmo PBEST1-
F-AGR-RAND son ligeramente inferiores, tanto en Score como en tiempo
de ejecuci´on, en comparaci´on con la versi´on original sin asignaci´on greedy
aleatorizada (PBEST1-F), que hasta el momento se considera la mejor pro-

Experimentaci´on y resultados

68

Dataset

PBEST1-F
Score Time (s)

Movement Libras

Computers

Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.154
Gene +0.000
-0.303
Toxicity +0.000
ECG5000 +0.020
Ecoli +0.003
Glass +0.000
-0.025
-0.033
4
8
2

Accent
Promedio
Veces mejor
Veces igual
Veces peor

PBEST1-F-SW-WO-PEN
Time (s)
+0.2
+0.5
+0.4
+0.5
-0.3
+0.2
+0.8
+15.4
-15.7
+0.2
-2.7
-6.6
+0.5
-1.5
-0.6
5
0
9

Score
+0.2 +0.000
+0.2 +0.000
+0.2 +0.000
+0.1 +0.000
+0.0
-0.004
+0.2 +0.000
-1.1
-0.154
-6.2 +0.000
-0.110
-21.3
+0.3 +0.000
-8.1
-0.055
-4.7 +0.001
-1.9 +0.010
-0.017
-0.1
-0.024
-3.0
5
7
7
1
2
6

Tabla 8.13: Comparativa PBEST1-F y PBEST1-F-SW-WO-PEN

puesta. No obstante, al incorporar tambi´en el algoritmo de Solis Wets, los
resultados mejoran notablemente, superando incluso a PBEST1-F. Aunque
el Score promedio es muy similar, destaca el hecho de que no hay ning´un
conjunto de datos en el que se obtenga un Score peor que con el algoritmo
original. Adem´as, aunque el tiempo promedio de ejecuci´on es superior al
obtenido con PBEST1-F, sigue siendo competitivo, mejorando en promedio
al algoritmo de referencia.

Experimentaci´on y resultados

69

Dataset

Time (s)

Computers

Score
Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.154
Gene +0.000
-0.103
Toxicity +0.000
ECG5000 +0.077
Ecoli +0.001
Glass +0.018
-0.033
-0.014
4
7
3

PBEST1-F-AGR-RAND PBEST1-F-AGR-RAND-SW
Time (s)
+0.4
+0.5
+0.7
+0.6
+0.2
+0.4
+1.8
+10.9
-13.2
+0.3
-0.7
-3.9
+0.7
-1.4
-0.2
4
0
10

Score
+0.3 +0.000
+0.4 +0.000
+0.6 +0.000
+0.5 +0.000
-0.5
-0.004
+0.4 +0.000
+0.4
-0.144
-7.9 +0.000
-0.153
-17.0
+0.3 +0.000
-0.087
-2.4
-0.001
-5.9
-0.016
-1.0
-0.070
-2.8
-0.034
-2.5
7
7
7
0
0
7

Accent
Promedio
Veces mejor
Veces igual
Veces peor

Movement Libras

Tabla 8.14: Comparativa PBEST1-F-AGR-RAND y PBEST1-F-AGR-
RAND-SW

8.6.3. Seleccionar a qu´e individuos aplicar la b´usqueda local

En la Subsecci´on 6.2.2 se propuso, en lugar de aplicar la b´usqueda local
a todas las soluciones de la poblaci´on, seleccionar ciertas soluciones y aplicar
la b´usqueda local ´unicamente sobre ellas. En concreto, se selecciona el 10 %
de las mejores soluciones, con el objetivo de refinar aquellas soluciones m´as
prometedoras, y un 10 % adicional elegido de forma aleatoria entre el resto
de la poblaci´on, con la intenci´on de favorecer la diversidad de la poblaci´on.

En la Tabla 8.15 se muestran los resultados obtenidos siguiendo esta
estrategia de selecci´on, compar´andola con la mejor propuesta hasta ahora.

Podemos ver que claramente la introducci´on de esta estrategia de selec-
ci´on ha tenido un impacto negativo en los resultados, tanto en Score como
en el tiempo de ejecuci´on.

Como ya se coment´o en la Subsecci´on 6.2.2, la b´usqueda local es uno
de los componentes m´as costosos computacionalmente del algoritmo, ya que
implica resolver un problema de optimizaci´on para asignar las instancias del
conjunto de datos a los centroides de la soluci´on. Al no aplicar la b´usqueda
local sobre todas las soluciones, disminuye el tiempo de ejecuci´on prome-

Experimentaci´on y resultados

70

PBEST1-F-AGR-RAND-SW
Time (s)

Movement Libras

Dataset

Computers

Score
Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.144
Gene +0.000
-0.153
Toxicity +0.000
-0.087
-0.001
-0.016
-0.070
-0.034
7
7
0

ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

SEL-BL
Score Time (s)
-0.2
+0.3
+0.3
-0.1
-0.7
+0.0
-1.6
-17.8
+41.3
+0.3
+7.5
+6.6
+5.4
+4.9
+3.3
5
1
8

+0.4 +0.000
+0.5 +0.000
+0.7 +0.000
+0.6 +0.000
+0.2
-0.001
+0.4 +0.000
+1.8 +0.669
+10.9 +0.000
-13.2 +0.212
+0.3 +0.000
-0.7 +0.021
-3.9 +0.003
+0.7 +0.112
-1.4 +0.433
-0.2 +0.103
1
7
6

4
0
10

Tabla 8.15: Comparativa PBEST1-F-AGR-RAND-SW y SEL-BL

dio por iteraci´on. Sin embargo, esto provoca un aumento en el n´umero de
iteraciones necesarias para que el algoritmo converja, y lo hace hacia solu-
ciones de menor calidad. Como consecuencia, se incrementa el tiempo total
de ejecuci´on y se obtiene un Score promedio peor.

8.7.

Introducci´on de la estrategia de reinicio de
poblaci´on

En la Secci´on 6.4 comentamos que existen varios factores que determi-
nan c´uando y c´omo se llevan a cabo los reinicios. En un primer momento, se
establece un m´aximo de 4 reinicios, los cuales se desencadenan tras detectar
un total de 2 iteraciones completas sin mejora en la soluci´on. Es decir, para
el par´ametro --Nmax se usar´a el valor 20 · 2 = 40 y para --restart el valor
4. Adem´as, se mantendr´a el tama˜no de la poblaci´on constante en todo mo-
mento. Los resultados se muestran en la Tabla 8.16, junto con los resultados
de la mejor propuesta hasta el momento: PBEST1-F-AGR-RAND-SW.

Observamos que, en general, las mejoras del Score son m´as pronuncia-
das que con el algoritmo PBEST1-F-AGR-RAND-SW, siendo la excepci´on

Experimentaci´on y resultados

71

Dataset

Computers

Score
Iris +0.000
Wine +0.000
Connectionist +0.000
Seeds +0.000
-0.004
Heart
Vertebral +0.000
-0.144
Gene +0.000
-0.153
Toxicity +0.000
-0.087
-0.001
-0.016
-0.070
-0.034
7
7
0

PBEST1-F-AGR-RAND-SW
Time (s)
+0.4
+0.5
+0.7
+0.6
+0.2
+0.4
+1.8
+10.9
-13.2
+0.3
-0.7
-3.9
+0.7
-1.4
-0.2
4
0
10

ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

Movement Libras

R-2IT-4

Score Time (s)
+7.7
+9.3
+10.4
+10.4
+15.9
+6.5
+39.6
+607.2
+149.6
+4.5
+96.8
+86.0
+32.8
+64.4
+81.5
0
0
14

+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.198
+0.000
-0.194
-4.983e+13
-0.085
-0.004
-0.033
+0.036
-3.559e+12
7
6
1

Tabla 8.16: Comparativa PBEST1-F-AGR-RAND-SW y R-2IT-4

el conjunto de datos Accent. Destaca especialmente la mejora obtenida en
el conjunto de datos Toxicity. Sin embargo, esta mejora en calidad viene
acompa˜nada de un aumento dr´astico en el tiempo de ejecuci´on, como era
de esperar, que pr´acticamente se multiplica por cinco en todos los casos al
introducir los cuatro reinicios.

Seguidamente, vamos a comprobar si a´un se consigue una mejora en los
resultados reduciendo el n´umero de reinicios a 2 y aumentando a 3 el n´umero
de iteraciones sin mejora necesarias para activar el reinicio. De esta forma, se
espera reducir el tiempo de ejecuci´on promedio respecto a la versi´on R-2IT-
4, manteniendo al mismo tiempo una buena calidad en las soluciones. En la
Tabla 8.17 se muestran los resultados obtenidos con esta configuraci´on, con
y sin el uso del algoritmo de Solis Wets.

Vemos que con el algoritmo R-3IT-2 se consigue reducir pr´acticamente a
la mitad el incremento promedio de tiempo de ejecuci´on respecto a la varian-
te con 4 reinicios R-2IT-4, aunque se obtienen peores resultados de media.
Al introducir tambi´en la b´usqueda de Solis Wets, observamos que aunque
el tiempo promedio empeore levemente, los resultados en t´erminos de Score
mejoran notablemente. En concreto, no se observa ning´un empeoramiento
respecto al algoritmo original, y las mejoras obtenidas superan en la mayor´ıa

Experimentaci´on y resultados

72

Dataset
Iris
Wine
Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

R-3IT-2

Score Time (s)
+4.1
+4.8
+5.2
+5.6
+8.8
+3.4
+19.7
+353.1
+67.4
+2.4
+46.9
+40.3
+15.4
+35.7
+43.8
0
0
14

+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.198
+0.000
-0.081
-4.983e+13
+0.002
-0.003
-0.034
+0.254
-3.559e+12
6
6
2

R-3IT-2-SW
Score Time (s)
+4.3
+4.6
+5.6
+5.7
+9.2
+3.7
+20.1
+405.7
+73.9
+2.5
+49.0
+39.7
+17.8
+41.5
+48.8
0
0
14

+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.203
+0.000
-0.164
-4.983e+13
-0.069
-0.004
-0.038
-0.063
-3.559e+12
8
6
0

Tabla 8.17: Comparativa R-3IT-2 y R-3IT-2-SW

de los casos a las logradas por la propuesta PBEST1-AGR-RAND-SW.

Uno de los motivos principales por los que se consider´o la estrategia de
reinicio es mantener la diversidad de la poblaci´on y, con ello, evitar una con-
vergencia prematura hacia ´optimos locales. La p´erdida de diversidad puede
provocar que las soluciones generadas en las ´ultimas iteraciones sean muy
similares entre s´ı, lo que limita la exploraci´on del espacio de b´usqueda y
reduce la probabilidad de encontrar mejores soluciones. En la Figura 8.2
podemos ver una comparativa de la evoluci´on de la diversidad de la pobla-
ci´on a lo largo de las iteraciones en la versi´on original del algoritmo y en la
variante con dos reinicios R-3IT-2. Por otro lado, en la Figura 8.3 podemos
ver la evoluci´on del Score de la mejor soluci´on, el Score medio y el peor
Score de la poblaci´on a lo largo de las iteraciones para ambos algoritmos.

Se aprecia como en el caso de la versi´on original del algoritmo, la diver-
sidad decrece r´apidamente en unas pocas iteraciones, mientras que al intro-
ducir los reinicios, la diversidad aumenta de nuevo hasta un valor cercano
al inicial, ya que todas las soluciones de la poblaci´on, excepto la mejor, se
vuelven a generar de forma aleatoria. Adem´as, si observamos la Figura 8.3,
podemos ver que, tras el segundo reinicio, se produce una mejora en el Score
de la mejor soluci´on. Esto muestra la utilidad de los reinicios para escapar

Experimentaci´on y resultados

73

(a) Algoritmo S-MDEClust

(b) Algoritmo R-3IT-2

Figura 8.2: Comparativa evoluci´on de la diversidad con y sin reinicios

(a) Algoritmo S-MDEClust

(b) Algoritmo R-3IT-2

Figura 8.3: Comparativa evoluci´on del Score con y sin reinicios

Experimentaci´on y resultados

74

de ´optimos locales y encontrar soluciones de mayor calidad, especialmente
cuando el algoritmo se hab´ıa estancado previamente sin registrar mejoras
durante varias iteraciones.

8.7.1. Disminuci´on del tama˜no de la poblaci´on

A continuaci´on, en la Tabla 8.18, se muestran los resultados de las varian-
tes R-3IT-2-DIS y R-3IT-2-DIS-SW, an´alogas a las dos ´ultimas propuestas
analizadas, pero disminuyendo el tama˜no de la poblaci´on despu´es de cada
reinicio. Con ello se pretende favorecer una mayor diversidad en las primeras
etapas del algoritmo y acelerar la convergencia en las etapas finales, en las
que la poblaci´on es m´as reducida, centrando el esfuerzo computacional en
refinar las mejores soluciones ya encontradas.

En concreto, el tama˜no de la poblaci´on antes del primer reinicio se esta-
blece como el doble del valor habitual. Dado que en todos los experimentos
se trabaja con una poblaci´on de 20 individuos, esto implica que el tama˜no
inicial ser´a de 40, tras el primer reinicio se reduce a 20 y por ´ultimo, tras el
segundo reinicio ser´a de 10.

Dataset
Iris
Wine
Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

R-3IT-2-DIS
Score Time (s)
+4.7
+5.6
+6.9
+6.7
+11.5
+4.1
+24.0
+461.8
+126.9
+3.1
+60.3
+58.2
+23.7
+49.6
+60.5
0
0
14

+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.203
+0.000
-0.191
-4.983e+13
-0.088
-0.004
-0.043
+0.023
-3.559e+12
7
6
1

R-3IT-2-DIS-SW

Score Time (s)
+5.1
+6.0
+7.2
+7.1
+12.4
+4.3
+26.9
+472.4
+127.9
+3.3
+63.4
+63.1
+25.5
+50.4
+62.5
0
0
14

+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.154
+0.000
-0.156
-4.983e+13
-0.038
-0.004
-0.041
-0.035
-3.559e+12
8
6
0

Tabla 8.18: Comparativa R-3IT-2-DIS y R-3IT-2-DIS-SW

Podemos ver que en la variante sin el algoritmo de Solis Wets (R-3IT-

Experimentaci´on y resultados

75

2-DIS), el tiempo promedio de ejecuci´on es ligeramente menor. Adem´as,
aunque hay un conjunto de datos en el que el Score empeora, la mejora en
el resto de los conjuntos de datos es, en general, mayor que la conseguida con
R-3IT-2-DIS-SW. No obstante, el tiempo promedio ha aumentado conside-
rablemente, en casi 20 segundos, con respecto a las variantes en las que no
se disminu´ıa el tama˜no de la poblaci´on. Esto se debe a que, al ser el tama˜no
de la poblaci´on m´as grande al inicio, se requiere de un mayor n´umero de
evaluaciones de la funci´on objetivo en cada iteraci´on.

8.7.2. Combinando diferentes propuestas

En la Tabla 8.19 se presentan los resultados de utilizar el cruce pbest1 al
mismo tiempo que la estrategia de reinicio. De nuevo, tenemos dos variantes:
una haciendo uso del algoritmo Solis Wets y otra sin ´el.

Dataset
Iris
Wine
Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

R-3IT-2-PBEST1-F

Score Time (s)
+4.0
+4.9
+5.3
+5.5
+9.1
+3.6
+19.7
+350.6
+56.8
+2.3
+41.3
+39.9
+15.2
+35.3
+42.4
0
0
14

+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.154
+0.000
-0.354
-4.983e+13
-0.090
-0.003
-0.028
-0.070
-3.559e+12
8
6
0

R-3IT-2-PBEST1-F-SW
Time (s)
+4.0
+4.7
+5.6
+5.7
+8.8
+3.4
+20.3
+370.9
+53.8
+2.3
+44.2
+34.9
+15.6
+30.7
+43.2
0
0
14

Score
+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.154
+0.000
-0.153
-4.983e+13
-0.090
-0.004
-0.028
-0.063
-3.559e+12
8
6
0

Tabla 8.19: Comparativa R-3IT-2-PBEST1-F y R-3IT-2-PBEST1-F-SW

En este caso, ambas variantes presentan un rendimiento igualado. Sin
embargo, la versi´on sin la b´usqueda Solis Wets parece presentar un desem-
pe˜no ligeramente superior, tanto en Score como en tiempo. Como podemos
ver, hay dos conjuntos de datos para los que R-3IT-2-PBEST1-F encuentra
mejores resultados que R-3IT-2-PBEST1-F-SW: Movement Libras y Accent,
siendo esta mejora significativamente mayor en el primero. Adem´as, el tiem-

Experimentaci´on y resultados

76

po de ejecuci´on promedio es menor en la versi´on sin Solis Wets, llegando
incluso a superar levemente a la variante R-3IT-2 con el operador de cruce
original, presentada previamente en la Tabla 8.17.

Seguidamente, en la Tabla 8.20 se muestran los resultados de a˜nadir a
las dos variantes que acabamos de analizar la t´ecnica de disminuci´on de la
poblaci´on.

Dataset
Iris
Wine
Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

R-3IT-2-DIS-PBEST1-F R-3IT-2-DIS-PBEST1-F-SW
Time (s)
+5.1
+6.1
+7.1
+7.2
+12.5
+4.3
+25.0
+479.7
+97.8
+3.2
+60.3
+55.5
+20.9
+45.9
+59.3
0
0
14

Score
+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.154
+0.000
-0.377
-4.983e+13
-0.090
+0.000
-0.047
-0.070
-3.559e+12
7
7
0

Score
+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.154
+0.000
-0.434
-4.983e+13
-0.073
-0.004
-0.029
-0.070
-3.559e+12
8
6
0

Time (s)
+5.3
+6.0
+6.8
+6.8
+11.5
+3.9
+23.5
+454.7
+92.3
+3.2
+61.0
+47.6
+21.4
+44.1
+56.3
0
0
14

Tabla 8.20: Comparativa R-3IT-2-DIS-PBEST1-F y R-3IT-2-DIS-PBEST1-
F-SW

Podemos observar que el rendimiento de ambas propuestas, de nuevo, es
bastante similar. En algunos casos, se obtiene un mejor Score con R-3IT-
2-DIS-PBEST1-F, mientras que en otros es R-3IT-2-DIS-PBEST1-F-SW la
que ofrece mejores resultados. Cabe destacar especialmente la mejora ob-
tenida por esta ´ultima en el conjunto de datos Movement Libras, siendo la
m´as significativa lograda hasta el momento en dicho dataset. En cuanto al
tiempo de ejecuci´on, R-3IT-2-DIS-PBEST1-F-SW presenta un promedio li-
geramente superior, una tendencia que ya se ha observado en otras variantes
que incorporan la b´usqueda de Solis Wets. Sin embargo, mejora en cuanto
al Score en un conjunto de datos m´as que R-3IT-2-DIS-PBEST1-F.

Por ´ultimo, analizaremos los resultados obtenidos al incorporar la asig-
naci´on greedy aleatorizada a una de las propuestas que mejor desempe˜no ha

Experimentaci´on y resultados

77

mostrado: R-3IT-2-PBEST1-F. Los resultados se muestran en la Tabla 8.21

Dataset
Iris
Wine
Connectionist
Seeds
Heart
Vertebral
Computers
Gene
Movement Libras
Toxicity
ECG5000
Ecoli
Glass
Accent
Promedio
Veces mejor
Veces igual
Veces peor

R-3IT-2-PBEST1-F

Score Time (s)
+4.0
+4.9
+5.3
+5.5
+9.1
+3.6
+19.7
+350.6
+56.8
+2.3
+41.3
+39.9
+15.2
+35.3
+42.4
-
-
-

+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.154
+0.000
-0.354
-4.983e+13
-0.090
-0.003
-0.028
-0.070
-3.559e+12
8
6
0

R-3IT-2-PBEST1-F-AGR-RAND
Time (s)
+4.4
+5.7
+6.4
+6.5
+10.4
+4.0
+24.1
+348.9
+66.1
+2.5
+50.4
+40.6
+17.6
+34.7
+44.5
-
-
-

Score
+0.000
+0.000
+0.000
+0.000
-0.004
+0.000
-0.154
+0.000
-0.340
-4.983e+13
-0.090
-0.004
-0.047
-0.070
-3.559e+12
8
6
0

Tabla 8.21: Comparativa R-3IT-2-PBEST1-F y R-3IT-2-PBEST1-F-AGR-
RAND

Aunque en ambas variantes observamos que el Score de las soluciones
encontradas mejora con respecto a la versi´on original del algoritmo para
todos los conjuntos de datos, esta mejora parece ligeramente mayor en la
versi´on que hace uso de la asignaci´on greedy aleatorizada: hay en dos casos en
los que la mejora en Score es mayor que la conseguida con la variante R-3IT-
2-PBEST1-F, aunque hay un caso (Movement Libras) en el que la mejora es
m´as peque˜na. Adem´as, el tiempo de ejecuci´on promedio es levemente mayor
en dicha variante.

8.8. Resumen de los resultados experimentales

En esta secci´on haremos un breve repaso por las propuestas desarro-
lladas, valorando si han supuesto una mejora significativa con respecto al
algoritmo original.

GRASP. Esta propuesta ha resultado claramente negativa, ya que

Experimentaci´on y resultados

78

las soluciones obtenidas presentan un Score muy deficiente en compa-
raci´on con el algoritmo original-.

Variaciones de la asignaci´on greedy . Se probaron dos variantes: la
asignaci´on greedy aleatorizada y la asignaci´on greedy aleatorizada con
penalizaci´on. De estas, la asignaci´on greedy aleatorizada parece ser la
m´as prometedora, aunque su impacto por s´ı sola no queda completa-
mente claro respecto al algoritmo original. No obstante, al combinarla
con otras propuestas, se observa una mejora m´as consistente en los
resultados.

SHADE. La incorporaci´on de SHADE ha mejorado el Score en va-
rios conjuntos de datos y reducido el tiempo de ejecuci´on promedio.
Sin embargo, en algunos casos ha disminuido la calidad de los resul-
tados, lo que sugiere que su efectividad depende del contexto y las
caracter´ısticas espec´ıficas del problema abordado.

Cambios en el operador de cruce. Estas modificaciones han teni-
do un impacto positivo en el rendimiento. En particular, la variante
pbest1 con el par´ametro F en el rango (0.5, 0.8) ha mejorado tanto el
Score como el tiempo de ejecuci´on promedio, convirti´endose en una de
las propuestas m´as destacadas.

B´usqueda local Solis Wets. Los resultados obtenidos con esta t´ecni-
ca son ambiguos. La versi´on que utiliza la funci´on objetivo sin penali-
zaci´on parece ser mejor que la que la incluye. Aunque en algunos casos
mejora el Score, en otros no, y adem´as implica un incremento leve en
el tiempo de ejecuci´on, por lo que su aplicaci´on debe evaluarse caso
por caso.

Selecci´on de los individuos a los que se aplica la b´usqueda lo-
cal. Esta estrategia ha permitido reducir el tiempo de ejecuci´on prome-
dio por iteraci´on, aunque ha aumentado el n´umero total de iteraciones
necesarias para converger, lo que ha provocado una disminuci´on en la
calidad final de las soluciones y, en consecuencia, un empeoramiento
del Score promedio y del tiempo total.

Estrategia de reinicio de poblaci´on. Incorporar reinicios ha sido
´util para mantener la diversidad y evitar la convergencia prematura
a ´optimos locales, permitiendo mejorar el Score de las soluciones ob-
tenidas especialmente cuando se combina con otras t´ecnicas como el
operador pbest1 y la disminuci´on del tama˜no de la poblaci´on en cada
reinicio. No obstante, presenta y gran inconveniente que es un aumento
notable en el tiempo de ejecuci´on promedio.

En la Figura 8.4 se muestra una comparativa global del Score de todas
las propuestas desarrolladas, detallando para cada variante, el n´umero de

Experimentaci´on y resultados

79

casos en los que el resultado mejor´o, se mantuvo igual o empeor´o respecto
a la versi´on original del algoritmo. En la Tabla 8.22 se muestran los mismos
resultados, pero en formato de tabla.

Podemos observar que, en general, el rendimiento de las propuestas ha
ido mejorando progresivamente a medida que se han incorporado nuevas
ideas y ajustes. Especialmente en las variantes finales, se aprecia que el
n´umero de casos en los que el Score empeora respecto a la versi´on original
se reduce a cero, lo que indica una mayor robustez y efectividad de las
´ultimas propuestas evaluadas.

ID Algoritmo

AGR-RAND
AGR-RAND-P
SHADE
PBEST1-F
PBEST2-F
PBEST1-F/2
PBEST2-F/2
SW-V1-WO-PEN
SW-WO-PEN
SW-W-PEN
SW-WO-PEN-AGR-RAND

1 GRASP
2
3
4
5
6
7
8
9
10
11
12
13 PBEST1-F-SW-WO-PEN
14 PBEST1-F-AGR-RAND
15 PBEST1-F-AGR-RAND-SW
16
SEL-BL
17 R-2IT-4
18 R-3IT-2
19 R-3IT-2-SW
20 R-3IT-2-DIS
21 R-3IT-2-DIS-SW
22 R-3IT-2-PBEST1-F
23 R-3IT-2-PBEST1-F-SW
24 R-3IT-2-DIS-PBEST1-F
25 R-3IT-2-DIS-PBEST1-F-SW
26 R-3IT-2-PBEST1-F-AGR-RAND

Veces
Mejor
1
4
5
3
4
6
5
5
4
4
2
4
5
4
7
1
7
6
8
7
8
8
8
7
8
8

Veces
Igual
0
8
6
6
8
7
7
7
7
6
6
7
7
7
7
7
6
6
6
6
6
6
6
7
6
6

Veces
Peor
13
2
3
5
2
1
2
2
3
4
6
3
2
3
0
6
1
2
0
1
0
0
0
0
0
0

Tabla 8.22: Resultados comparativos del Score de las propuestas

Figura 8.4: Comparativa del Score de todas las propuestas: veces que ha
sido mejor, igual o peor que el algoritmo S-MDEClust original.

Cap´ıtulo 9

Conclusiones y trabajo
futuro

En este trabajo, nos hemos enfocado en el estudio del clustering semi-
supervisado con restricciones. Esta t´ecnica combina m´etodos de clustering
no supervisado con un conocimiento parcial del problema, que se expresa a
trav´es de restricciones entre pares de datos. Este enfoque es especialmente
´util en situaciones donde conseguir datos etiquetados resulta complicado o
costoso, algo que sucede con frecuencia en muchos campos pr´acticos.

El objetivo principal de este trabajo ha sido mejorar un algoritmo ya
competente de clustering semisupervisado: S-MDEClust, mediante la pro-
puesta de m´ultiples variantes y estrategias de mejora. Podemos decir que
hemos alcanzado con creces este objetivo: hemos desarrollado una amplia
bater´ıa de propuestas, muchas de las cuales han demostrado tener un im-
pacto positivo en la calidad de las soluciones, evaluadas a trav´es de la funci´on
objetivo.

El desarrollo de este trabajo ha abarcado todas las etapas de una expe-
rimentaci´on completa. Comenzamos con una revisi´on bibliogr´afica detallada
que nos ayud´o a contextualizar el problema y a elegir las t´ecnicas m´as re-
levantes. Luego, realizamos un estudio te´orico para entender a fondo tanto
el problema como los algoritmos de referencia que utilizamos. A ello le han
seguido la implementaci´on de propuestas, su evaluaci´on experimental y un
an´alisis exhaustivo de los resultados.

A nivel experimental, se ha comprobado que las propuestas que mejor
resultado han dado han sido aquellas basadas en modificaciones sobre el
operador de cruce, especialmente el uso del cruce pbest1, as´ı como la intro-
ducci´on de estrategias de reinicio, especialmente cuando estas se combinan
con t´ecnicas adicionales como la disminuci´on progresiva del tama˜no de la
poblaci´on o el propio cruce pbest1. Adem´as, se ha observado que el uso del

81

Conclusiones y trabajo futuro

82

cruce pbest1 no solo mejora la calidad de las soluciones, sino que tambi´en
reduce el tiempo de ejecuci´on en promedio.

Sin embargo, las propuestas que incorporan estrategias de reinicios, a
pesar de ser las que logran las mayores mejoras en t´erminos de la funci´on
objetivo, tambi´en presentan un incremento considerable en el tiempo de eje-
cuci´on. Por lo tanto, su uso deber´a valorarse seg´un el contexto, teniendo en
cuenta los recursos computacionales disponibles y las exigencias de tiempo.

Por ´ultimo, veremos algunas posibles l´ıneas de extensi´on del proyecto.

Mejora del tiempo de ejecuci´on. Dado que algunas de las propues-
tas m´as efectivas en cuanto a la mejora del Score, como la estrategia
de reinicio, implican un aumento significativo del tiempo de ejecuci´on,
una posible l´ınea de trabajo relevante ser´ıa optimizar la implementa-
ci´on mediante t´ecnicas de paralelizaci´on. Una alternativa interesante
es el modelo de islas, en el que varias subpoblaciones evolucionan de
manera paralela y peri´odicamente intercambian sus mejores soluciones
con las subpoblaciones vecinas mediante migraci´on. Este enfoque no
solo puede mejorar el rendimiento computacional aprovechando m´ulti-
ples n´ucleos o m´aquinas, sino tambi´en favorecer la diversidad gen´etica
y evitar la convergencia prematura.

Considerar una estrategia de reinicio adaptativo. En lugar de
utilizar un n´umero fijo de reinicios y una reducci´on predeterminada
del tama˜no poblacional, se podr´ıa investigar una estrategia adaptati-
va que tome decisiones basadas en el comportamiento din´amico del
algoritmo. Por ejemplo, se puede monitorizar la tasa de mejora de la
mejor soluci´on o la diversidad de la poblaci´on, y activar el reinicio solo
cuando estas caigan por debajo de ciertos umbrales. Tambi´en pueden
usarse mecanismos probabil´ısticos, donde la probabilidad de reinicio
aumenta con el tiempo sin mejoras, o aplicar reinicios parciales que
reemplacen solo a los individuos m´as similares.

Explorar otros enfoques. El trabajo se ha centrado principalmente
en mejorar el algoritmo mem´etico S-MDEClust, proponiendo modi-
ficaciones sobre este. Aunque tambi´en se ha propuesto un algoritmo
GRASP, no se ha profundizado mucho en ´el ni se han propuesto modi-
ficaciones para mejorar a la versi´on inicial propuesta. Ser´ıa interesan-
te dedicar esfuerzos a explorar variantes m´as sofisticadas de GRASP,
adem´as de otros enfoques completamente distintos.

Bibliograf´ıa

[1] Kiri Wagstaff and Claire Cardie. Clustering with instance-level cons-

traints. AAAI/IAAI, 1097(577-584):197, 2000.

[2] S. Basu, Ian Davidson, and K.L. Wagstaff. Constrained clustering:

Advances in algorithms, theory, and applications. 01 2008.

[3] Kiri Wagstaff, Claire Cardie, Seth Rogers, Stefan Schr¨odl, et al. Cons-
trained k-means clustering with background knowledge. In Icml, volu-
me 1, pages 577–584, 2001.

[4] Germ´an Gonz´alez-Almagro, Daniel Peralta, Eli De Poorter, Jos´e-
Ram´on Cano, and Salvador Garc´ıa. Semi-supervised constrained clus-
tering: an in-depth overview, ranked taxonomy and future research di-
rections. Artificial Intelligence Review, 58(5):157, March 2025. doi:
10.1007/s10462-024-11103-8.

[5] Daniel Aloise, Amit Deshpande, Pierre Hansen, and Preyas Popat. Np-
hardness of euclidean sum-of-squares clustering. Machine Learning,
75(2):245–248, May 2009. doi:10.1007/s10994-009-5103-0.

[6] Kashif Hussain, Mohd Najib Mohd Salleh, Shi Cheng, and Yuhui Shi.
Metaheuristic research: a comprehensive survey. Artificial Intelligence
Review, 52(4):2191–2233, 2019. doi:10.1007/s10462-017-9605-z.

[7] Pierluigi Mansueto and Fabio Schoen. Memetic differential evo-
arXiv preprint ar-

lution methods for semi-supervised clustering.
Xiv:2403.04322, 2024.

[8] Veronica Piccialli, Anna Russo Russo, and Antonio M. Sudo-
so.
An exact algorithm for semi-supervised minimum sum-of-
squares clustering. Computers Operations Research, 147:105958,
2022. URL: https://www.sciencedirect.com/science/article/
pii/S0305054822002076, doi:10.1016/j.cor.2022.105958.

[9] James MacQueen. Some methods for classification and analysis of mul-
tivariate observations. In Proceedings of the Fifth Berkeley Symposium

83

BIBLIOGRAF´IA

84

on Mathematical Statistics and Probability, Volume 1: Statistics, volu-
me 5, pages 281–298. University of California press, 1967.

[10] WEI TAN, YAN YANG, and TIANRUI LI.

AN IMPROVED
COP-KMEANS ALGORITHM FOR SOLVING CONSTRAINT
VIOLATION, pages 690–696. URL: https://www.worldscientific.
com/doi/abs/10.1142/9789814324700_0104,
arXiv:https://www.
worldscientific.com/doi/pdf/10.1142/9789814324700_0104,
doi:10.1142/9789814324700_0104.

[11] Tonny Rutayisire, Yan Yang, Chao Lin, and Jinyuan Zhang. A modified
cop-kmeans algorithm based on sequenced cannot-link set. In Rough
Sets and Knowledge Technology: 6th International Conference, RSKT
2011, Banff, Canada, October 9-12, 2011. Proceedings 6, pages 217–225.
Springer, 2011.

[12] Philipp Baumann. A binary linear programming-based k-means al-
gorithm for clustering with must-link and cannot-link constraints. In
2020 IEEE International Conference on Industrial Engineering and En-
gineering Management (IEEM), pages 324–328, 2020. doi:10.1109/
IEEM45057.2020.9309775.

[13] Sugato Basu, Arindam Banerjee, and Raymond J. Mooney. Ac-
tive Semi-Supervision for Pairwise Constrained Clustering, pages
URL: https://epubs.siam.org/doi/abs/10.1137/1.
333–344.
9781611972740.31,
arXiv:https://epubs.siam.org/doi/pdf/10.
1137/1.9781611972740.31, doi:10.1137/1.9781611972740.31.

[14] Yu Xia. A global optimization method for semi-supervised clustering.

Data mining and knowledge discovery, 18:214–256, 2009.

[15] Jiming Peng and Yu Xia. A Cutting Algorithm for the Minimum
Sum-of-Squared Error Clustering, pages 150–160.
URL: https:
//epubs.siam.org/doi/abs/10.1137/1.9781611972757.14, arXiv:
https://epubs.siam.org/doi/pdf/10.1137/1.9781611972757.14,
doi:10.1137/1.9781611972757.14.

[16] Daniel Aloise and Pierre Hansen. A branch-and-cut sdp-based algo-
rithm for minimum sum-of-squares clustering. Pesquisa Operacional,
29:503–516, 2009.

[17] Behrouz Babaki, Tias Guns, and Siegfried Nijssen. Constrained clus-
tering using column generation. In Helmut Simonis, editor, Integration
of AI and OR Techniques in Constraint Programming, pages 438–454,
Cham, 2014. Springer International Publishing.

BIBLIOGRAF´IA

85

[18] Tias Guns, Thi-Bich-Hanh Dao, Christel Vrain, and Khanh-Chuong
Duong. Repetitive branch-and-bound using constraint programming
for constrained minimum sum-of-squares clustering.
In ECAI 2016,
pages 462–470. IOS Press, 2016.

[19] Viet-Vu Vu, Nicolas Labroche, and Bernadette Bouchon-Meunier. Lea-
der ant clustering with constraints. In 2009 IEEE-RIVF International
Conference on Computing and Communication Technologies, pages 1–8,
2009. doi:10.1109/RIVF.2009.5174648.

[20] Xiaohua Xu, Lin Lu, Ping He, Zhoujin Pan, and Ling Chen.
Neu-
and
URL: https://www.

Improving constrained clustering via swarm intelligence.
rocomputing,
116:317–325,
Methodology in Intelligent Computing.
sciencedirect.com/science/article/pii/S0925231212007278,
doi:10.1016/j.neucom.2012.03.031.

Advanced Theory

2013.

[21] Daniel Gribel, Michel Gendreau, and Thibaut Vidal.

supervised clustering with inaccurate pairwise annotations.
formation Sciences, 607:441–457, 2022.
sciencedirect.com/science/article/pii/S0020025522004558,
doi:10.1016/j.ins.2022.05.035.

Semi-
In-
URL: https://www.

[22] Daniel Gribel and Thibaut Vidal.

brid genetic algorithm for minimum sum-of-squares
Pattern Recognition, 88:569–583, 2019.
sciencedirect.com/science/article/pii/S0031320318304436,
doi:10.1016/j.patcog.2018.12.022.

Hg-means: A scalable hy-
clustering.
URL: https://www.

[23] Germ´an Gonz´alez-Almagro, Juli´an Luengo, Jos´e-Ram´on Cano, and
Salvador Garc´ıa. Dils: Constrained clustering through dual
ite-
rative local search. Computers Operations Research, 121:104979,
2020. URL: https://www.sciencedirect.com/science/article/
pii/S0305054820300964, doi:10.1016/j.cor.2020.104979.

[24] Ian Davidson and S. S. Ravi. Intractability and clustering with cons-
traints. In Proceedings of the 24th International Conference on Machine
Learning, ICML ’07, page 201–208, New York, NY, USA, 2007. Asso-
ciation for Computing Machinery. doi:10.1145/1273496.1273522.

[25] Ian Davidson and S. S. Ravi. Clustering With Constraints: Fea-
sibility Issues and the ¡italic¿k¡/italic¿-Means Algorithm, pages
URL: https://epubs.siam.org/doi/abs/10.1137/1.
138–149.
9781611972757.13,
arXiv:https://epubs.siam.org/doi/pdf/10.
1137/1.9781611972757.13, doi:10.1137/1.9781611972757.13.

BIBLIOGRAF´IA

86

[26] Pierluigi Mansueto and Fabio Schoen. Memetic differential evolution
methods for clustering problems. Pattern Recognition, 114:107849,
2021. URL: https://www.sciencedirect.com/science/article/
pii/S0031320321000364, doi:10.1016/j.patcog.2021.107849.

[27] David F. Crouse. On implementing 2d rectangular assignment al-
IEEE Transactions on Aerospace and Electronic Systems,

gorithms.
52(4):1679–1696, 2016. doi:10.1109/TAES.2016.140952.

[28] Jiming Peng and Yu Wei. Approximating k-means-type clustering via
semidefinite programming. SIAM Journal on Optimization, 18(1):186–
arXiv:https://doi.org/10.1137/050641983, doi:10.
205, 2007.
1137/050641983.

[29] Veronica Piccialli, Antonio M Sudoso, and Angelika Wiegele. Sos-sdp:
INFORMS

an exact solver for minimum sum-of-squares clustering.
Journal on Computing, 34(4):2144–2162, 2022.

[30] Francisco J Solis and Roger J-B Wets. Minimization by random search
techniques. Mathematics of operations research, 6(1):19–30, 1981.

[31] Ryoji Tanabe and Alex Fukunaga. Success-history based parameter
adaptation for differential evolution. In 2013 IEEE Congress on Evo-
lutionary Computation, pages 71–78, 2013. doi:10.1109/CEC.2013.
6557555.

[32] Jingqiao Zhang and Arthur C. Sanderson. Jade: Adaptive differential
evolution with optional external archive. IEEE Transactions on Evolu-
tionary Computation, 13(5):945–958, 2009. doi:10.1109/TEVC.2009.
2014613.

[33] Dheeru Dua and Casey Graff. Uci machine learning repository, 2017.

URL: http://archive.ics.uci.edu/ml.

[34] Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony
Bagnall, Abdullah Mueen, and Gustavo Batista. The ucr time se-
ries classification archive, July 2015. www.cs.ucr.edu/~eamonn/time_
series_data/.

[35] Defeng Sun, Kim-Chuan Toh, Yancheng Yuan, and Xin-Yuan Zhao.
Sdpnal+: A matlab software for semidefinite programming with bound
constraints (version 1.0). Optimization Methods and Software, 35(1):87–
115, 2020.

[36] Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2024.

URL: https://www.gurobi.com.

Ap´endice A

Par´ametros de ejecuci´on del
algoritmo PC-SOS-SDP

87

Par´ametros de ejecuci´on del algoritmo PC-SOS-SDP

88

Par´ametro
BRANCH AND BOUND TOL

Valor Descripci´on
1 · 10−4 Tolerancia de optimalidad del

BRANCH AND BOUND PARALLEL

BRANCH AND BOUND MAX NODES

16

200

BRANCH AND BOUND VISITING STRATEGY

0

SDP SOLVER SESSION THREADS ROOT

16

SDP SOLVER SESSION THREADS
SDP SOLVER FOLDER
SDP SOLVER TOL
SDP SOLVER VERBOSE
SDP SOLVER MAX CP ITER ROOT

SDP SOLVER MAX CP ITER

SDP SOLVER CP TOL

SDP SOLVER MAX INEQ

SDP SOLVER INHERIT PERC

SDP SOLVER EPS INEQ

SDP SOLVER EPS ACTIVE

algoritmo.
N´umero de hilos para ejecu-
ci´on paralela.
N´umero m´aximo de nodos en
el ´arbol de b´usqueda.
Estrategia de
recorrido: 0
(best first), 1 (depth first), 2
(breadth first).
Hilos para la sesi´on MATLAB
en la ra´ız.
Hilos para nodos ML y CL.
Ruta completa de SDPNAL+.

1
-
1 · 10−5 Exactitud de SDPNAL+.
Mostrar log: 0 (no), 1 (s´ı).
0
Iteraciones m´aximas. de plano
80
de corte en la ra´ız.
Iteraciones m´aximas.
del
plano de corte en nodos ML
y CL.

40

1 · 10−6 Tolerancia del plano de corte
entre dos iteraciones seguidas.
100000 N´umero m´aximo de desigual-
dades consideradas.
Fracci´on de desigualdades he-
redadas.

1.0

1 · 10−4 Tolerancia para detectar las

desigualdades incumplidas.

1 · 10−6 Tolerancia para desigualdades

activas.

SDP SOLVER MAX PAIR INEQ

100000 N´umero m´aximo de desigual-

SDP SOLVER PAIR PERC

SDP SOLVER MAX TRIANGLE INEQ

SDP SOLVER TRIANGLE PERC

0.05

dades de pares.
Fracci´on de las desigualdades
de pares incumplidas a a˜nadir.
100000 N´umero m´aximo de desigual-
dades triangulares.
Fracci´on de las desigualda-
des triangulares incumplidas a
a˜nadir.

0.05

Tabla A.1: Par´ametros de configuraci´on del algoritmo exacto PC-SOS-SDP

