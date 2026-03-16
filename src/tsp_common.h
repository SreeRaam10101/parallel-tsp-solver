#ifndef TSP_COMMON_H
#define TSP_COMMON_H

#include <cmath>
#include <cstring>
#include <climits>
#include <cfloat>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <cassert>
#include <iomanip>

#define MAXSIZE 50
#define INF 1e18f
#define GUB_UPDATE_INTERVAL 500
#define LB_CHECK_INTERVAL 200
#define MAX_PQ_SIZE_LIMIT 500000
#define QE_SPAN_S 3
#define WORK_TRANSFER_COUNT 2
#define WEIGHT_PROGRESS_PENALTY 0.15f

#define TAG_INITIAL_WORK      100
#define TAG_FINAL_GUB         101
#define TAG_QE_THRESHOLD      200
#define TAG_QE_WORK_REQUEST   201
#define TAG_QE_WORK_TRANSFER  202
#define TAG_TERM_CHECK        300
#define TAG_TERM_WAKEUP       301
#define TAG_TERM_CONFIRM      302
#define TAG_GLOBAL_TERM       303

#ifdef DEBUG_ENABLED
#define DEBUG_PRINT(rank, fmt, ...) fprintf(stderr, "[Rank %d] " fmt "\n", rank, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(rank, fmt, ...) ((void)0)
#endif

enum ProcessState { ACTIVE, IDLE, TERM_SENT };

struct CityNode { int x; int y; };

struct CityGraph {
    int n_city;
    CityNode city[MAXSIZE];
    float dis[MAXSIZE][MAXSIZE];

    void compute_distances() {
        for (int i = 0; i < n_city; i++)
            for (int j = 0; j < n_city; j++) {
                if (i == j) { dis[i][j] = 0.0f; continue; }
                float dx = (float)(city[i].x - city[j].x);
                float dy = (float)(city[i].y - city[j].y);
                dis[i][j] = sqrtf(dx * dx + dy * dy);
            }
    }
};

struct BBNode {
    int number_visit_city;
    float cost_so_far;
    float mst_cost_val;
    int path[MAXSIZE];
    int visited[MAXSIZE];

    float estimated_cost() const { return cost_so_far + mst_cost_val; }

    float weighted_heuristic_value(int total_cities) const {
        float base = estimated_cost();
        float progress_ratio = (total_cities > 1)
            ? (float)(number_visit_city - 1) / (float)(total_cities - 1) : 0.0f;
        return base + (base * WEIGHT_PROGRESS_PENALTY * progress_ratio);
    }

    bool is_solution(int n_city) const { return number_visit_city == n_city; }

    float hamilton_cost(const CityGraph* g) const {
        return cost_so_far + g->dis[path[number_visit_city - 1]][path[0]];
    }

    void compute_mst_cost(const CityGraph* g) {
        int n = g->n_city;
        if (number_visit_city >= n) { mst_cost_val = 0.0f; return; }

        std::vector<int> unvisited;
        for (int i = 0; i < n; i++) if (!visited[i]) unvisited.push_back(i);

        if (unvisited.empty()) {
            mst_cost_val = g->dis[path[number_visit_city - 1]][path[0]];
            return;
        }

        int m = (int)unvisited.size();
        std::vector<float> key(m, INF);
        std::vector<bool> in_mst(m, false);
        key[0] = 0.0f;
        float mst_total = 0.0f;

        for (int count = 0; count < m; count++) {
            int u = -1; float min_key = INF;
            for (int i = 0; i < m; i++)
                if (!in_mst[i] && key[i] < min_key) { min_key = key[i]; u = i; }
            if (u == -1) break;
            in_mst[u] = true;
            mst_total += min_key;
            for (int i = 0; i < m; i++)
                if (!in_mst[i]) {
                    float w = g->dis[unvisited[u]][unvisited[i]];
                    if (w < key[i]) key[i] = w;
                }
        }

        int last_city = path[number_visit_city - 1], first_city = path[0];
        float min_last = INF, min_first = INF;
        for (int i = 0; i < m; i++) {
            float dl = g->dis[last_city][unvisited[i]];
            float df = g->dis[first_city][unvisited[i]];
            if (dl < min_last) min_last = dl;
            if (df < min_first) min_first = df;
        }
        mst_cost_val = mst_total + min_last + min_first;
    }

    BBNode* visit_city(int c, const CityGraph* g) const {
        if (visited[c]) return nullptr;
        BBNode* child = new BBNode();
        memcpy(child->path, path, sizeof(path));
        memcpy(child->visited, visited, sizeof(visited));
        child->number_visit_city = number_visit_city + 1;
        child->path[number_visit_city] = c;
        child->visited[c] = 1;
        child->cost_so_far = cost_so_far + g->dis[path[number_visit_city - 1]][c];
        child->compute_mst_cost(g);
        return child;
    }

    struct Compare {
        bool operator()(const BBNode* a, const BBNode* b) const {
            return a->estimated_cost() > b->estimated_cost();
        }
    };
};

using PriorityQueue = std::priority_queue<BBNode*, std::vector<BBNode*>, BBNode::Compare>;

inline bool parse_tsp_input(const char* filename, CityGraph* graph) {
    std::ifstream infile(filename);
    if (!infile.is_open()) { std::cerr << "Error: Cannot open " << filename << std::endl; return false; }
    std::string line; bool reading_coords = false; int idx = 0;
    while (std::getline(infile, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line.find("DIMENSION") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) graph->n_city = std::stoi(line.substr(colon + 1));
            else { std::istringstream iss(line); std::string t; iss >> t >> graph->n_city; }
        } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            reading_coords = true;
        } else if (line.find("EOF") != std::string::npos) { break;
        } else if (reading_coords) {
            std::istringstream iss(line); int id; float fx, fy;
            if (iss >> id >> fx >> fy && idx < MAXSIZE) {
                graph->city[idx].x = (int)fx; graph->city[idx].y = (int)fy; idx++;
            }
        }
    }
    if (idx > 0 && graph->n_city == 0) graph->n_city = idx;
    infile.close(); graph->compute_distances();
    std::cout << "Parsed " << graph->n_city << " cities from " << filename << std::endl;
    return true;
}

inline BBNode* create_root_node(const CityGraph* g) {
    BBNode* root = new BBNode();
    memset(root, 0, sizeof(BBNode));
    root->number_visit_city = 1; root->path[0] = 0; root->visited[0] = 1;
    root->compute_mst_cost(g);
    return root;
}

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
inline double elapsed_seconds(TimePoint s, TimePoint e) {
    return std::chrono::duration<double>(e - s).count();
}

#endif
