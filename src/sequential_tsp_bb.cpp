/**
 * sequential_tsp_bb.cpp
 * Sequential Branch and Bound solver for TSP (baseline).
 * Usage: ./sequential_tsp_bb [input_file]
 */
#include "tsp_common.h"

#ifndef FILENAME
#define FILENAME "Input.txt"
#endif

int main(int argc, char* argv[]) {
    const char* input_file = (argc > 1) ? argv[1] : FILENAME;
    CityGraph city_graph;
    memset(&city_graph, 0, sizeof(CityGraph));
    if (!parse_tsp_input(input_file, &city_graph)) return 1;

    int n = city_graph.n_city;
    std::cout << "=== Sequential TSP Branch & Bound ===\nCities: " << n << std::endl;

    float global_upper_bound = INF;
    long long nodes_processed = 0, solutions_found = 0;
    int best_path[MAXSIZE]; memset(best_path, 0, sizeof(best_path));
    PriorityQueue local_pq;

    BBNode* root = create_root_node(&city_graph);
    local_pq.push(root);

    TimePoint start_time = Clock::now();

    while (!local_pq.empty()) {
        BBNode* node = local_pq.top(); local_pq.pop();
        nodes_processed++;

        if (node->estimated_cost() >= global_upper_bound) { delete node; continue; }

        if (node->is_solution(n)) {
            float tour_cost = node->hamilton_cost(&city_graph);
            if (tour_cost < global_upper_bound) {
                global_upper_bound = tour_cost;
                memcpy(best_path, node->path, sizeof(int) * n);
                solutions_found++;
            }
            delete node; continue;
        }

        for (int c = 0; c < n; c++) {
            if (!node->visited[c]) {
                BBNode* child = node->visit_city(c, &city_graph);
                if (child && child->estimated_cost() < global_upper_bound)
                    local_pq.push(child);
                else if (child) delete child;
            }
        }
        delete node;
    }

    TimePoint end_time = Clock::now();
    double exec_time = elapsed_seconds(start_time, end_time);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Optimal tour cost: " << global_upper_bound << std::endl;
    std::cout << "Execution time:    " << exec_time << " s" << std::endl;
    std::cout << "Nodes processed:   " << nodes_processed << std::endl;
    std::cout << "Solutions found:   " << solutions_found << std::endl;
    std::cout << "Best tour: ";
    for (int i = 0; i < n; i++) { std::cout << best_path[i]; if (i < n-1) std::cout << " -> "; }
    std::cout << " -> " << best_path[0] << std::endl;
    std::cout << "\nCSV: " << n << "," << exec_time << "," << nodes_processed << "," << global_upper_bound << std::endl;
    return 0;
}
