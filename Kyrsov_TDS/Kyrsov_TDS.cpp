#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <chrono>
#include <random>
#include <cmath>
#include <queue>
#include <functional>
#include <numeric>

using namespace std;
using namespace std::chrono;

// Функция для вычисления стоимости пути
int calculateCost(const vector<vector<int>>& graph, const vector<int>& path) {
    int cost = 0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        cost += graph[path[i]][path[i + 1]];
    }
    cost += graph[path.back()][path.front()]; // возвращение в начальный город
    return cost;
}

// Полный перебор (Brute Force)
int tspBruteForce(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> vertices;
    for (int i = 1; i < n; ++i) {
        vertices.push_back(i);
    }

    int minCost = INT_MAX;
    do {
        int currentCost = calculateCost(graph, vertices);
        minCost = min(minCost, currentCost);
    } while (next_permutation(vertices.begin(), vertices.end()));

    return minCost;
}

// Метод ближайшего соседа (Nearest Neighbor)
int tspNearestNeighbor(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<bool> visited(n, false);
    int totalCost = 0;
    int currentNode = 0;
    visited[currentNode] = true;

    for (int i = 1; i < n; ++i) {
        int nextNode = -1;
        int minCost = INT_MAX;

        for (int j = 0; j < n; ++j) {
            if (!visited[j] && graph[currentNode][j] < minCost) {
                minCost = graph[currentNode][j];
                nextNode = j;
            }
        }

        totalCost += minCost;
        currentNode = nextNode;
        visited[currentNode] = true;
    }

    totalCost += graph[currentNode][0]; // возвращаемся в начальный город

    return totalCost;
}

// Динамическое программирование (Held-Karp)
int tspHeldKarp(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<vector<int>> dp(1 << n, vector<int>(n, INT_MAX));
    dp[1][0] = 0;

    for (int mask = 1; mask < (1 << n); ++mask) {
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                for (int j = 0; j < n; ++j) {
                    if ((mask & (1 << j)) && graph[j][i] != INT_MAX && dp[mask ^ (1 << i)][j] != INT_MAX) {
                        dp[mask][i] = min(dp[mask][i], dp[mask ^ (1 << i)][j] + graph[j][i]);
                    }
                }
            }
        }
    }

    int minCost = INT_MAX;
    for (int i = 1; i < n; ++i) {
        if (graph[i][0] != INT_MAX) {
            minCost = min(minCost, dp[(1 << n) - 1][i] + graph[i][0]);
        }
    }

    return minCost;
}


// Жадный алгоритм (Greedy Algorithm)
int tspGreedy(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> path;
    vector<bool> visited(n, false);
    path.push_back(0);
    visited[0] = true;

    int totalCost = 0;
    int currentNode = 0;

    for (int i = 1; i < n; ++i) {
        int nextNode = -1;
        int minCost = INT_MAX;

        for (int j = 0; j < n; ++j) {
            if (!visited[j] && graph[currentNode][j] < minCost) {
                minCost = graph[currentNode][j];
                nextNode = j;
            }
        }

        path.push_back(nextNode);
        visited[nextNode] = true;
        totalCost += minCost;
        currentNode = nextNode;
    }

    totalCost += graph[currentNode][0]; // возвращаемся в начальный город

    return totalCost;
}

// Муравьиный алгоритм (Ant Colony Optimization)
const int NUM_ANTS = 50;
const int NUM_ITERATIONS = 1000;
const double ALPHA = 1.0;
const double BETA = 5.0;
const double EVAPORATION = 0.5;
const double Q = 100.0;

double calculateProbability(double pheromone, double distance, double alpha, double beta) {
    return pow(pheromone, alpha) * pow(1.0 / distance, beta);
}

int tspAntColonyOptimization(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<vector<double>> pheromones(n, vector<double>(n, 1.0));
    vector<int> bestPath;
    int bestCost = INT_MAX;

    random_device rd;
    mt19937 gen(rd());

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        vector<vector<int>> paths(NUM_ANTS);
        vector<int> costs(NUM_ANTS, 0);

        for (int ant = 0; ant < NUM_ANTS; ++ant) {
            vector<bool> visited(n, false);
            int currentNode = 0;
            paths[ant].push_back(currentNode);
            visited[currentNode] = true;

            for (int i = 1; i < n; ++i) {
                vector<double> probabilities(n, 0.0);
                double sum = 0.0;

                for (int j = 0; j < n; ++j) {
                    if (!visited[j]) {
                        probabilities[j] = calculateProbability(pheromones[currentNode][j], graph[currentNode][j], ALPHA, BETA);
                        sum += probabilities[j];
                    }
                }

                for (int j = 0; j < n; ++j) {
                    probabilities[j] /= sum;
                }

                discrete_distribution<> d(probabilities.begin(), probabilities.end());
                int nextNode = d(gen);

                paths[ant].push_back(nextNode);
                visited[nextNode] = true;
                costs[ant] += graph[currentNode][nextNode];
                currentNode = nextNode;
            }

            costs[ant] += graph[currentNode][0]; // возвращаемся в начальный город

            if (costs[ant] < bestCost) {
                bestCost = costs[ant];
                bestPath = paths[ant];
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                pheromones[i][j] *= (1 - EVAPORATION);
            }
        }

        for (int ant = 0; ant < NUM_ANTS; ++ant) {
            for (int i = 0; i < n - 1; ++i) {
                int from = paths[ant][i];
                int to = paths[ant][i + 1];
                pheromones[from][to] += Q / costs[ant];
                pheromones[to][from] += Q / costs[ant];
            }
        }
    }

    return bestCost;
}

// Минимальное остовное дерево (MST) для TSP
int mstTSP(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> parent(n);
    vector<int> key(n, INT_MAX);
    vector<bool> inMST(n, false);

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < n - 1; ++count) {
        int minKey = INT_MAX, u = -1;

        for (int v = 0; v < n; ++v) {
            if (!inMST[v] && key[v] < minKey) {
                minKey = key[v];
                u = v;
            }
        }

        inMST[u] = true;

        for (int v = 0; v < n; ++v) {
            if (graph[u][v] && !inMST[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    int cost = 0;
    for (int i = 1; i < n; ++i) {
        cost += graph[i][parent[i]];
    }

    return cost * 2; // потому что мы должны пройти через все ребра дважды
}

// Метод ветвей и границ (Branch and Bound)
struct Node {
    vector<int> path;
    int cost;
    int bound;
    int level;
    bool operator<(const Node& other) const {
        return bound > other.bound;
    }
};

int calculateBound(const vector<vector<int>>& graph, const Node& node) {
    int bound = node.cost;
    int n = graph.size();
    vector<bool> visited(n, false);
    for (int i = 0; i < node.path.size(); ++i) {
        visited[node.path[i]] = true;
    }

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            int minEdge = INT_MAX;
            for (int j = 0; j < n; ++j) {
                if (i != j && !visited[j]) {
                    minEdge = min(minEdge, graph[i][j]);
                }
            }
            bound += minEdge;
        }
    }

    return bound;
}

int tspBranchAndBound(const vector<vector<int>>& graph) {
    int n = graph.size();

    auto cmp = [](const Node& a, const Node& b) { return a.bound > b.bound; };
    priority_queue<Node, vector<Node>, decltype(cmp)> pq(cmp);

    Node root;
    root.path.push_back(0);
    root.cost = 0;
    root.bound = calculateBound(graph, root);
    root.level = 1;
    pq.push(root);

    int bestCost = INT_MAX;

    while (!pq.empty()) {
        Node node = pq.top();
        pq.pop();

        if (node.bound < bestCost) {
            for (int i = 1; i < n; ++i) {
                if (find(node.path.begin(), node.path.end(), i) == node.path.end()) {
                    Node child = node;
                    child.path.push_back(i);
                    child.cost += graph[node.path.back()][i];
                    if (child.level == n - 1) {
                        child.cost += graph[i][0];
                        if (child.cost < bestCost) {
                            bestCost = child.cost;
                        }
                    }
                    else {
                        child.bound = calculateBound(graph, child);
                        if (child.bound < bestCost) {
                            child.level++;
                            pq.push(child);
                        }
                    }
                }
            }
        }
    }

    return bestCost;
}

// Метод ближайшего соседа с 2-opt улучшением (Nearest Neighbor with 2-opt)
int twoOptSwap(vector<int>& path, int i, int k) {
    while (i < k) {
        swap(path[i], path[k]);
        i++;
        k--;
    }
    return 0;
}

int tspNearestNeighborWith2Opt(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> path(n);
    vector<bool> visited(n, false);
    path[0] = 0;
    visited[0] = true;

    for (int i = 1; i < n; ++i) {
        int last = path[i - 1];
        int nextNode = -1;
        int minCost = INT_MAX;
        for (int j = 0; j < n; ++j) {
            if (!visited[j] && graph[last][j] < minCost) {
                minCost = graph[last][j];
                nextNode = j;
            }
        }
        path[i] = nextNode;
        visited[nextNode] = true;
    }

    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 1; i < n - 1; ++i) {
            for (int k = i + 1; k < n; ++k) {
                int currentCost = calculateCost(graph, path);
                twoOptSwap(path, i, k);
                int newCost = calculateCost(graph, path);
                if (newCost < currentCost) {
                    improved = true;
                }
                else {
                    twoOptSwap(path, i, k);
                }
            }
        }
    }

    return calculateCost(graph, path);
}

// Случайный поиск (Random Search)
int tspRandomSearch(const vector<vector<int>>& graph, int iterations = 10000) {
    int n = graph.size();
    vector<int> bestPath(n);
    iota(bestPath.begin(), bestPath.end(), 0);
    int bestCost = calculateCost(graph, bestPath);

    random_device rd;
    mt19937 g(rd());

    for (int i = 0; i < iterations; ++i) {
        vector<int> newPath = bestPath;
        shuffle(newPath.begin() + 1, newPath.end(), g);
        int newCost = calculateCost(graph, newPath);
        if (newCost < bestCost) {
            bestCost = newCost;
            bestPath = newPath;
        }
    }

    return bestCost;
}

// Генетический алгоритм (Genetic Algorithm)
struct Individual {
    vector<int> path;
    int cost;
};

int calculateCostGA(const vector<vector<int>>& graph, const vector<int>& path) {
    int cost = 0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        cost += graph[path[i]][path[i + 1]];
    }
    cost += graph[path.back()][path.front()];
    return cost;
}

vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    int n = parent1.size();
    vector<int> child(n, -1);
    int start = rand() % n;
    int end = start + (rand() % (n - start));

    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }

    int index = 0;
    for (int i = 0; i < n; ++i) {
        if (find(child.begin(), child.end(), parent2[i]) == child.end()) {
            while (child[index] != -1) {
                ++index;
            }
            child[index] = parent2[i];
        }
    }

    return child;
}

void mutate(vector<int>& individual) {
    int n = individual.size();
    int i = rand() % n;
    int j = rand() % n;
    swap(individual[i], individual[j]);
}

bool compare(const Individual& a, const Individual& b) {
    return a.cost < b.cost;
}

int tspGeneticAlgorithm(const vector<vector<int>>& graph, int populationSize = 100, int generations = 1000) {
    int n = graph.size();
    vector<Individual> population(populationSize);

    for (int i = 0; i < populationSize; ++i) {
        population[i].path.resize(n);
        iota(population[i].path.begin(), population[i].path.end(), 0);
        random_shuffle(population[i].path.begin() + 1, population[i].path.end());
        population[i].cost = calculateCostGA(graph, population[i].path);
    }

    for (int generation = 0; generation < generations; ++generation) {
        sort(population.begin(), population.end(), compare);

        vector<Individual> newPopulation(populationSize);

        for (int i = 0; i < populationSize; ++i) {
            int parent1 = rand() % (populationSize / 2);
            int parent2 = rand() % (populationSize / 2);
            newPopulation[i].path = crossover(population[parent1].path, population[parent2].path);
            if (rand() % 100 < 5) {
                mutate(newPopulation[i].path);
            }
            newPopulation[i].cost = calculateCostGA(graph, newPopulation[i].path);
        }

        population = newPopulation;
    }

    sort(population.begin(), population.end(), compare);
    return population[0].cost;
}

// Симулированный отжиг (Simulated Annealing)
int tspSimulatedAnnealing(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> path(n);
    iota(path.begin(), path.end(), 0);

    random_device rd;
    mt19937 gen(rd());

    int bestCost = calculateCost(graph, path);
    vector<int> bestPath = path;

    double temperature = 10000.0;
    double coolingRate = 0.003;

    while (temperature > 1.0) {
        vector<int> newPath = path;
        uniform_int_distribution<> dist(1, n - 1);
        int i = dist(gen);
        int j = dist(gen);
        swap(newPath[i], newPath[j]);

        int currentCost = calculateCost(graph, path);
        int newCost = calculateCost(graph, newPath);

        if (newCost < currentCost || exp((currentCost - newCost) / temperature) >((double)rand() / RAND_MAX)) {
            path = newPath;
            currentCost = newCost;
        }

        if (currentCost < bestCost) {
            bestCost = currentCost;
            bestPath = path;
        }

        temperature *= 1 - coolingRate;
    }

    return bestCost;
}

// Лазерный лучевой поиск (Beam Search)
const int BEAM_WIDTH = 3;

int tspBeamSearch(const vector<vector<int>>& graph) {
    int n = graph.size();
    priority_queue<Node> pq;
    vector<int> bestPath;
    int bestCost = INT_MAX;

    Node start;
    start.path.push_back(0);
    start.cost = 0;
    start.bound = calculateBound(graph, start);
    start.level = 1;
    pq.push(start);

    while (!pq.empty()) {
        vector<Node> candidates;
        for (int i = 0; i < BEAM_WIDTH && !pq.empty(); ++i) {
            candidates.push_back(pq.top());
            pq.pop();
        }

        for (const Node& node : candidates) {
            if (node.bound < bestCost) {
                for (int i = 1; i < n; ++i) {
                    if (find(node.path.begin(), node.path.end(), i) == node.path.end()) {
                        Node child;
                        child.path = node.path;
                        child.path.push_back(i);
                        child.cost = node.cost + graph[node.path.back()][i];
                        child.bound = calculateBound(graph, child);
                        child.level = node.level + 1;

                        if (child.level == n - 1) {
                            child.path.push_back(0);
                            child.cost += graph[i][0];
                            if (child.cost < bestCost) {
                                bestCost = child.cost;
                                bestPath = child.path;
                            }
                        }
                        else {
                            pq.push(child);
                        }
                    }
                }
            }
        }
    }

    return bestCost;
}

int main() {

    setlocale(LC_ALL, "Russian");

    vector<vector<int>> graph = {
        {0, 29, 20, 21, 16, 31, 100},
        {29, 0, 15, 29, 28, 40, 72},
        {20, 15, 0, 15, 14, 25, 81},
        {21, 29, 15, 0, 4, 12, 92},
        {16, 28, 14, 4, 0, 16, 94},
        {31, 40, 25, 12, 16, 0, 98},
        {100, 72, 81, 92, 94, 98, 0}
    };

    // Измерение времени выполнения для каждого метода
    auto start = high_resolution_clock::now();
    int minCostBruteForce = tspBruteForce(graph);
    auto end = high_resolution_clock::now();
    auto durationBruteForce = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostNearestNeighbor = tspNearestNeighbor(graph);
    end = high_resolution_clock::now();
    auto durationNearestNeighbor = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostHeldKarp = tspHeldKarp(graph);
    end = high_resolution_clock::now();
    auto durationHeldKarp = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostGreedy = tspGreedy(graph);
    end = high_resolution_clock::now();
    auto durationGreedy = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostAntColony = tspAntColonyOptimization(graph);
    end = high_resolution_clock::now();
    auto durationAntColony = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostMST = mstTSP(graph);
    end = high_resolution_clock::now();
    auto durationMST = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostBranchAndBound = tspBranchAndBound(graph);
    end = high_resolution_clock::now();
    auto durationBranchAndBound = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostNearestNeighborWith2Opt = tspNearestNeighborWith2Opt(graph);
    end = high_resolution_clock::now();
    auto durationNearestNeighborWith2Opt = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostRandomSearch = tspRandomSearch(graph);
    end = high_resolution_clock::now();
    auto durationRandomSearch = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostGeneticAlgorithm = tspGeneticAlgorithm(graph);
    end = high_resolution_clock::now();
    auto durationGeneticAlgorithm = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostSimulatedAnnealing = tspSimulatedAnnealing(graph);
    end = high_resolution_clock::now();
    auto durationSimulatedAnnealing = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    int minCostBeamSearch = tspBeamSearch(graph);
    end = high_resolution_clock::now();
    auto durationBeamSearch = duration_cast<microseconds>(end - start).count();

    // Вывод результатов
    cout << "Полный перебор: " << minCostBruteForce << ", Время: " << durationBruteForce << " микросекунд" << endl << endl;
    cout << "Метод ближайшего соседа: " << minCostNearestNeighbor << ", Время: " << durationNearestNeighbor << " микросекунд" << endl << endl;
    cout << "Динамическое программирование: " << minCostHeldKarp << ", Время: " << durationHeldKarp << " микросекунд" << endl << endl;
    cout << "Жадный алгоритм: " << minCostGreedy << ", Время: " << durationGreedy << " микросекунд" << endl << endl;
    cout << "Муравьиный алгоритм: " << minCostAntColony << ", Время: " << durationAntColony << " микросекунд" << endl << endl;
    cout << "Минимальное остовное дерево: " << minCostMST << ", Время: " << durationMST << " микросекунд" << endl << endl;
    cout << "Метод ветвей и границ: " << minCostBranchAndBound << ", Время: " << durationBranchAndBound << " микросекунд" << endl << endl;
    cout << "Метод ближайшего соседа с 2-opt улучшением: " << minCostNearestNeighborWith2Opt << ", Время: " << durationNearestNeighborWith2Opt << " микросекунд" << endl << endl;
    cout << "Случайный поиск: " << minCostRandomSearch << ", Время: " << durationRandomSearch << " микросекунд" << endl << endl;
    cout << "Генетический алгоритм: " << minCostGeneticAlgorithm << ", Время: " << durationGeneticAlgorithm << " микросекунд" << endl << endl;
    cout << "Симулированный отжиг: " << minCostSimulatedAnnealing << ", Время: " << durationSimulatedAnnealing << " микросекунд" << endl << endl;
    cout << "Лазерный лучевой поиск: " << minCostBeamSearch << ", Время: " << durationBeamSearch << " микросекунд" << endl << endl;

    return 0;
}
