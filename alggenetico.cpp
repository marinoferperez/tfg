#include "algGenetico.h"
#include "ProblemInfluencer.h"
#include <numeric> // Para accumulate
#include <set>
using namespace std;


void algGenetico::mutacion(tSolution &sol, size_t num_nodos) {
    int pos = Random::get<int>(0, sol.size() - 1);

    // genes actuales
    set<int> usados(sol.begin(), sol.end());

    vector<int> candidatos;
    for (int i = 0; i < (int)num_nodos; ++i) {
        if (usados.find(i) == usados.end()) {
            candidatos.push_back(i);
        }
    }

    
    int gen = candidatos[Random::get<int>(0, candidatos.size() - 1)];
    sol[pos] = gen;
}

tSolution algGenetico::torneo(const vector<tSolution> &poblacion, const vector<tFitness> &fitness_poblacion) {
    int idx1 = Random::get<int>(0, tam_poblacion - 1);
    int idx2 = Random::get<int>(0, tam_poblacion - 1);
    int idx3 = Random::get<int>(0, tam_poblacion - 1);

    int mejor = (fitness_poblacion[idx1] > fitness_poblacion[idx2]) ? idx1 : idx2;
    mejor = (fitness_poblacion[mejor] > fitness_poblacion[idx3]) ? mejor : idx3;
    return poblacion[mejor];  
}

void algGenetico::reparacion(tSolution &hijo, const tSolution &padre, int inicio, int fin, int num_nodos) {
    using namespace std;

    set<int> genes_validos(padre.begin(), padre.end());
    vector<int> frecuencia(num_nodos, 0);

    // calcular frecuencia real del hijo completo
    for (int gen : hijo) frecuencia[gen]++;

    // detectar valor de padre que no aparecen en hijo

    vector<int> faltantes;
    for (int gen : genes_validos) if (frecuencia[gen] == 0) faltantes.push_back(gen);

    // corregimos todos los duplicados fuera del segmento
    for (int i = 0; i < hijo.size(); ++i) {
        if (i < inicio || i > fin) {
            int gen = hijo[i];
            if (frecuencia[gen] > 1 && !faltantes.empty()) {
                int nuevo = faltantes.back();
                faltantes.pop_back();

                hijo[i] = nuevo;
                frecuencia[gen]--;
                frecuencia[nuevo]++;
            }
        }
    }
}

void algGenetico::cruce_1(tSolution &padre1, tSolution &padre2, int num_nodos) {
    tSolution combinado = padre1;
    combinado.insert(combinado.end(), padre2.begin(), padre2.end());

    // ordenamos 

    sort(combinado.begin(), combinado.end());

    // generamos hijos 

    tSolution hijo1, hijo2;

    for (int i = 0; i < combinado.size(); i++) {
        if (i % 2 == 0) hijo1.push_back(combinado[i]);
        else hijo2.push_back(combinado[i]);
    }

    // fuera de la solucion no existen hijos

    padre1 = hijo1;
    padre2 = hijo2;
}

void algGenetico::cruce_2(tSolution &padre1, tSolution &padre2, int num_nodos) {
    int inicio = Random::get<int>(0, padre1.size() - 2);
    int fin = Random::get<int>(inicio + 1, padre1.size() - 1);

    tSolution hijo1 = padre1;
    tSolution hijo2 = padre2;

    for (int i = inicio; i <= fin; i++) {
        hijo1[i] = padre2[i];
        hijo2[i] = padre1[i];
    }

    // Reparación
    reparacion(hijo1, padre1, inicio, fin, num_nodos);
    reparacion(hijo2, padre2, inicio, fin, num_nodos);

    padre1 = hijo1;
    padre2 = hijo2;
}

void algGenetico::cruce(tSolution &padre1, tSolution &padre2, int num_nodos) {
    if (con_orden) cruce_1(padre1, padre2, num_nodos);
    else cruce_2(padre1, padre2, num_nodos);
}

ResultMH algGenetico::optimize_generacional(Problem *problem, int maxevals) {
    vector<tSolution> poblacion;
    vector<tFitness> fitness_poblacion;
    int num_nodos = problem->getSize();
    int ev = 0;

    for (int i = 0; i < tam_poblacion; i++) {
        tSolution pob_inicial = problem->createSolution();
        poblacion.push_back(pob_inicial);
        fitness_poblacion.push_back(problem->fitness(poblacion[i]));
        ev++;
    }

    while (ev < maxevals) {        
        // seleccion por torneo de 3 con repeticion de indices

        vector<tSolution> nueva_poblacion;
        for (int i = 0; i < tam_poblacion; i++) {
            nueva_poblacion.push_back(torneo(poblacion, fitness_poblacion));
        }

        // cruzamos de dos en dos:
        // * 1 y 2
        // * 3 y 4
        // * 5 y 6
        // * ...

        // establecemos el numero de cruces esperados

        int cruces_esperados = (int)ceil(prob_cruce * (tam_poblacion / 2.0));

        for (int i = 0; i < cruces_esperados; i++) {
            cruce(nueva_poblacion[i * 2], nueva_poblacion[i * 2 + 1], num_nodos);
        }

        // mutacion por individuo

        // calculamos el numero de indiviudos mutados 
        int num_mutaciones = static_cast<int>(tam_poblacion * prob_mutacion);
        
        // pra cada indv se escoge aleatoriamente un solo gen y lo mutamos 
        for (int i = 0; i < num_mutaciones; i++) {
            int individuo = Random::get<int>(0, tam_poblacion - 1);     // (gen)
            mutacion(nueva_poblacion[individuo], num_nodos);
        }
        
        // evaluamos la nueva poblacion

        vector<tFitness> fitness_nueva_poblacion;

        for (int i = 0; i < tam_poblacion; i++) {
            fitness_nueva_poblacion.push_back(problem->fitness(nueva_poblacion[i]));
            ev++;
        }  

        // elitismo: conservar el mejor individuo
        
        // buscamos el mejor individuo de la poblacion actual y el peor de la nueva poblacion
        // sustituimos el mejro por el peor de la nueva poblacion si es mejor

        int mejor_individuo = max_element(fitness_poblacion.begin(), fitness_poblacion.end()) - fitness_poblacion.begin();
        int peor_individuo = min_element(fitness_nueva_poblacion.begin(), fitness_nueva_poblacion.end()) - fitness_nueva_poblacion.begin();

        if (fitness_poblacion[mejor_individuo] > fitness_nueva_poblacion[peor_individuo]) {
            nueva_poblacion[peor_individuo] = poblacion[mejor_individuo];
            fitness_nueva_poblacion[peor_individuo] = fitness_poblacion[mejor_individuo];
        }

        // actualizamos la poblacion

        poblacion = nueva_poblacion;
        fitness_poblacion = fitness_nueva_poblacion;
    }

    // buscamos el mejor individuo de la poblacion

    int mejor_individuo = max_element(fitness_poblacion.begin(), fitness_poblacion.end()) - fitness_poblacion.begin();
    tSolution mejor_solucion = poblacion[mejor_individuo];
    tFitness mejor_fitness = fitness_poblacion[mejor_individuo];

    return ResultMH(mejor_solucion, mejor_fitness, ev);
}


ResultMH algGenetico::optimize_estacionario(Problem *problem, int maxevals) {
    vector<tSolution> poblacion;
    vector<tFitness> fitness_poblacion;
    int num_nodos = problem->getSize();
    int ev = 0;

    // poblacion inicial
    for (int i = 0; i < tam_poblacion; i++) {
        tSolution pob_inicial = problem->createSolution();
        poblacion.push_back(pob_inicial);
        fitness_poblacion.push_back(problem->fitness(poblacion[i]));
        ev++;
    }

    while (ev < maxevals) {
        // seleccion de dos padres por torneo de 3 

        tSolution padre1 = torneo(poblacion, fitness_poblacion);
        tSolution padre2 = torneo(poblacion, fitness_poblacion);

        // clonamos los padres para obtener los hijos

        tSolution hijo1 = padre1;
        tSolution hijo2 = padre2;

        // cruzamos los hijos con probabilildad 1

        cruce(hijo1, hijo2, num_nodos);

        // mutamos los hijos con p fija

        if (Random::get<double>(0, 1) < prob_mutacion) {
            mutacion(hijo1, num_nodos);
        }

        if (Random::get<double>(0, 1) < prob_mutacion) {
            mutacion(hijo2, num_nodos);
        }

        // evaluamos los hijos

        tFitness fitness_hijo1 = problem->fitness(hijo1);
        tFitness fitness_hijo2 = problem->fitness(hijo2);
        ev += 2;         // cada vez que evaluamos se incrementa el contador

        // seleccionamos mejor hijo de los dos en cuanto a su fitness

        tSolution mejor_hijo;
        tFitness mejor_fitness;

        if (fitness_hijo1 > fitness_hijo2) {
            mejor_hijo = hijo1;
            mejor_fitness = fitness_hijo1;
        }
        else {
            mejor_hijo = hijo2;
            mejor_fitness = fitness_hijo2;
        }

        // seleccionamos el peor de la poblacion y lo sustituimos por el mejor hijo

        int peor_individuo = min_element(fitness_poblacion.begin(), fitness_poblacion.end()) - fitness_poblacion.begin();

        if (fitness_poblacion[peor_individuo] < mejor_fitness) {
            poblacion[peor_individuo] = mejor_hijo;
            fitness_poblacion[peor_individuo] = mejor_fitness;
        }
    }

    // buscamos el mejor individuo de la poblacion

    int mejor_individuo = max_element(fitness_poblacion.begin(), fitness_poblacion.end()) - fitness_poblacion.begin();
    tSolution mejor_solucion = poblacion[mejor_individuo];
    tFitness mejor_fitness = fitness_poblacion[mejor_individuo];

    return ResultMH(mejor_solucion, mejor_fitness, ev);
}