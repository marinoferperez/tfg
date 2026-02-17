#include <algorithm>
#include "ProblemInfluencer.h"
#include <random.hpp>

tFitness ProblemInfluencer::fitness(tSolution &solution) {
    double influencia = 0.0;
    effolkronium::random_local r;
    r.seed(42);
    
    for (int i = 0; i < ev; ++i) {
        vector<int> infectados_total(solution.begin(), solution.end());
        vector<int> infectados_ini = infectados_total;

        while (!infectados_ini.empty()) {
            vector<int> infectados_nuevos;
            for (int node : infectados_ini) {
                for (size_t vecino = 0; vecino < num_nodos; vecino++) {
                    if (matriz[node][vecino] == true && find(infectados_nuevos.begin(), infectados_nuevos.end(), vecino) == infectados_nuevos.end()) {
                        if (r.get<double>(0, 1) <= p) {
                            infectados_nuevos.push_back(vecino);
                            infectados_total.push_back(vecino);
                        }
                    }
                }
            }
        
            infectados_ini = infectados_nuevos;
        }

        influencia += infectados_total.size();
    }
    return influencia / ev;
}

// utilizamos para crear una solucion para el algoritmo Random y BL
tSolution ProblemInfluencer::createSolution() {
    tSolution sol;

    while (sol.size() < K) {
        int nodo_actual = Random::get<int>(0, num_nodos - 1);
        if (find(sol.begin(), sol.end(), nodo_actual) == sol.end()) {
            sol.push_back(nodo_actual);
        }   
    }

    return sol;
}

tHeuristic ProblemInfluencer::heuristic(tSolution &sol, tOption option) {
    vector<int> vecinos_directos = getVecinos(option);
    int num_directos = vecinos_directos.size();
    int vecinos_no_directos = 0;
    for (int vec : vecinos_directos) {
        vecinos_no_directos += getVecinos(vec).size();
    }
    return num_directos + vecinos_no_directos;
}

vector<int> ProblemInfluencer::getVecinos(int node) const {
    vector<int> vecinos;
    for (int i = 0; i < (int) matriz[node].size(); i++) {
        if (matriz[node][i]) vecinos.push_back(i);
    }

    return vecinos;
}

void ProblemInfluencer::leerDatos(const string &filename) {
    ifstream file("data/" + filename);
    if (!file) {
        cerr << "Error al abrir el archivo: " << filename << endl;
        return;
    }

    string line;
    num_nodos = 0;
    while (getline(file, line)) {
        if (line[0] == '#') continue;
        istringstream iss(line);
        int from, to;
        if (iss >> from >> to) {
            // cout << "Leyendo arista: " << from << " -> " << to << endl;
            num_nodos = max(num_nodos, static_cast<size_t>(max(from, to) + 1));
        }
    }
    cout << "Total de nodos: " << num_nodos << endl;
    matriz.resize(num_nodos, vector<bool>(num_nodos, false));
    file.clear();
    file.seekg(0, ios::beg);
    
    while (getline(file, line)) {
        if (line[0] == '#') continue;
        istringstream iss(line);
        int from, to;
        if (iss >> from >> to) {
            matriz[from][to] = true;
        }
    }
}

tSolution ProblemInfluencer::Int(const tSolution &sel, int i , int j) {
    tSolution aux = sel;
    aux[i] = j;

    return aux;
}
