/**
 * parallel_tsp_bb.cpp
 * Basic parallel B&B TSP solver using MPI (no dynamic load balancing).
 * Uses MPI_Pack/MPI_Unpack for BBNode serialization.
 * Usage: mpirun -np <N> ./parallel_tsp_bb [input_file]
 */
#include "tsp_common.h"
#include <mpi.h>

#ifndef FILENAME
#define FILENAME "Input.txt"
#endif

static int my_rank, psize;

// ---------- BBNode serialization ----------
static int pack_bbnode(BBNode* node, char* buf, int bufsize) {
    int pos = 0;
    MPI_Pack(&node->number_visit_city, 1, MPI_INT, buf, bufsize, &pos, MPI_COMM_WORLD);
    MPI_Pack(&node->cost_so_far, 1, MPI_FLOAT, buf, bufsize, &pos, MPI_COMM_WORLD);
    MPI_Pack(&node->mst_cost_val, 1, MPI_FLOAT, buf, bufsize, &pos, MPI_COMM_WORLD);
    MPI_Pack(node->path, MAXSIZE, MPI_INT, buf, bufsize, &pos, MPI_COMM_WORLD);
    MPI_Pack(node->visited, MAXSIZE, MPI_INT, buf, bufsize, &pos, MPI_COMM_WORLD);
    return pos;
}

static BBNode* unpack_bbnode(char* buf, int bufsize) {
    BBNode* node = new BBNode();
    int pos = 0;
    MPI_Unpack(buf, bufsize, &pos, &node->number_visit_city, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bufsize, &pos, &node->cost_so_far, 1, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bufsize, &pos, &node->mst_cost_val, 1, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bufsize, &pos, node->path, MAXSIZE, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bufsize, &pos, node->visited, MAXSIZE, MPI_INT, MPI_COMM_WORLD);
    return node;
}

static const int PACK_BUF_SIZE = 2048;

// ---------- GUB synchronization ----------
static void broadcast_global_upper_bound(float& gub) {
    float local_gub = gub;
    MPI_Allreduce(&local_gub, &gub, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    const char* input_file = (argc > 1) ? argv[1] : FILENAME;
    CityGraph city_graph;
    memset(&city_graph, 0, sizeof(CityGraph));

    // Rank 0 parses and broadcasts
    if (my_rank == 0) {
        if (!parse_tsp_input(input_file, &city_graph)) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Bcast(&city_graph.n_city, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(city_graph.city, MAXSIZE * sizeof(CityNode), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(city_graph.dis, MAXSIZE * MAXSIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int n = city_graph.n_city;
    float global_upper_bound = INF;
    long long nodes_processed = 0;
    PriorityQueue local_pq;
    ProcessState current_state = IDLE;

    // Rank 0: generate initial subproblems and distribute
    if (my_rank == 0) {
        std::cout << "=== Parallel TSP B&B (Basic) | " << psize << " processes ===" << std::endl;

        PriorityQueue init_pq;
        BBNode* root = create_root_node(&city_graph);
        init_pq.push(root);

        // Expand root to create subproblems
        std::vector<BBNode*> subproblems;
        int expand_target = psize * 2;
        while (!init_pq.empty() && (int)subproblems.size() < expand_target) {
            BBNode* node = init_pq.top(); init_pq.pop();
            nodes_processed++;
            if (node->is_solution(n)) {
                float cost = node->hamilton_cost(&city_graph);
                if (cost < global_upper_bound) global_upper_bound = cost;
                delete node; continue;
            }
            for (int c = 0; c < n; c++) {
                if (!node->visited[c]) {
                    BBNode* child = node->visit_city(c, &city_graph);
                    if (child) { subproblems.push_back(child); }
                }
            }
            delete node;
        }
        // Drain remaining
        while (!init_pq.empty()) { subproblems.push_back(init_pq.top()); init_pq.pop(); }

        // Distribute: round-robin to workers, keep some for rank 0
        for (int i = 0; i < (int)subproblems.size(); i++) {
            int dest = i % psize;
            if (dest == 0) {
                local_pq.push(subproblems[i]);
            } else {
                char buf[PACK_BUF_SIZE];
                pack_bbnode(subproblems[i], buf, PACK_BUF_SIZE);
                MPI_Send(buf, PACK_BUF_SIZE, MPI_CHAR, dest, TAG_INITIAL_WORK, MPI_COMM_WORLD);
                delete subproblems[i];
            }
        }
        // Send empty message to workers that got nothing
        for (int r = 1; r < psize; r++) {
            bool got_work = false;
            for (int i = r; i < (int)subproblems.size(); i += psize) { got_work = true; break; }
            if (!got_work) MPI_Send(nullptr, 0, MPI_CHAR, r, TAG_INITIAL_WORK, MPI_COMM_WORLD);
        }
        current_state = local_pq.empty() ? IDLE : ACTIVE;
    } else {
        // Workers receive initial work
        MPI_Status status;
        char buf[PACK_BUF_SIZE];
        MPI_Recv(buf, PACK_BUF_SIZE, MPI_CHAR, 0, TAG_INITIAL_WORK, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);
        if (count > 0) {
            BBNode* node = unpack_bbnode(buf, PACK_BUF_SIZE);
            local_pq.push(node);
            current_state = ACTIVE;
            // Receive additional subproblems
            int flag = 1;
            while (flag) {
                MPI_Iprobe(0, TAG_INITIAL_WORK, MPI_COMM_WORLD, &flag, &status);
                if (flag) {
                    MPI_Recv(buf, PACK_BUF_SIZE, MPI_CHAR, 0, TAG_INITIAL_WORK, MPI_COMM_WORLD, &status);
                    MPI_Get_count(&status, MPI_CHAR, &count);
                    if (count > 0) { local_pq.push(unpack_bbnode(buf, PACK_BUF_SIZE)); }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TimePoint start_time = Clock::now();

    // ---------- Main B&B Loop ----------
    int terminated_count = 0;
    bool globally_terminated = false;
    bool local_termination_signal_sent = false;
    long long iter = 0;

    while (!globally_terminated) {
        // Process local work
        if (!local_pq.empty()) {
            current_state = ACTIVE;
            BBNode* node = local_pq.top(); local_pq.pop();
            nodes_processed++;

            if (node->estimated_cost() >= global_upper_bound) { delete node; }
            else if (node->is_solution(n)) {
                float cost = node->hamilton_cost(&city_graph);
                if (cost < global_upper_bound) global_upper_bound = cost;
                delete node;
            } else {
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
        } else {
            current_state = IDLE;
        }

        iter++;

        // Periodic GUB sync
        if (iter % GUB_UPDATE_INTERVAL == 0) {
            broadcast_global_upper_bound(global_upper_bound);
        }

        // Termination detection
        if (current_state == IDLE && !local_termination_signal_sent) {
            if (my_rank == 0) terminated_count++;
            else {
                int dummy = my_rank;
                MPI_Send(&dummy, 1, MPI_INT, 0, TAG_TERM_CHECK, MPI_COMM_WORLD);
            }
            local_termination_signal_sent = true;
        }

        // Rank 0: check for termination messages
        if (my_rank == 0) {
            int flag = 1;
            while (flag) {
                MPI_Status status;
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_TERM_CHECK, MPI_COMM_WORLD, &flag, &status);
                if (flag) {
                    int dummy;
                    MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, TAG_TERM_CHECK, MPI_COMM_WORLD, &status);
                    terminated_count++;
                }
            }
            if (terminated_count >= psize) globally_terminated = true;
        }
        MPI_Bcast(&globally_terminated, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    TimePoint end_time = Clock::now();

    // ---------- Gather results ----------
    float final_gub = global_upper_bound;
    MPI_Allreduce(MPI_IN_PLACE, &final_gub, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

    long long all_nodes[64]; memset(all_nodes, 0, sizeof(all_nodes));
    MPI_Gather(&nodes_processed, 1, MPI_LONG_LONG, all_nodes, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double exec_time = elapsed_seconds(start_time, end_time);
        long long total = 0, max_n = 0, min_n = LLONG_MAX;
        for (int i = 0; i < psize; i++) {
            total += all_nodes[i];
            if (all_nodes[i] > max_n) max_n = all_nodes[i];
            if (all_nodes[i] < min_n) min_n = all_nodes[i];
        }
        double avg = (double)total / psize;
        double var = 0;
        for (int i = 0; i < psize; i++) var += (all_nodes[i] - avg) * (all_nodes[i] - avg);
        double stddev = sqrt(var / psize);

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Optimal cost:      " << final_gub << std::endl;
        std::cout << "Execution time:    " << exec_time << " s" << std::endl;
        std::cout << "Total nodes:       " << total << std::endl;
        std::cout << "Avg/Max/Min nodes: " << avg << " / " << max_n << " / " << min_n << std::endl;
        std::cout << "Max-Min diff:      " << (max_n - min_n) << std::endl;
        std::cout << "Std dev:           " << stddev << std::endl;
        std::cout << "\nCSV: " << n << "," << exec_time << "," << total << ","
                  << avg << "," << max_n << "," << min_n << "," << stddev << "," << final_gub << std::endl;
    }

    MPI_Finalize();
    return 0;
}
