/**
 * parallel_tsp_bb_cutting.cpp
 * Parallel B&B with priority queue size limiting to prevent OOM.
 * Identical to parallel_tsp_bb.cpp but caps local_pq size at MAX_PQ_SIZE_LIMIT.
 * Note: May produce suboptimal results due to discarded nodes.
 * Usage: mpirun -np <N> ./parallel_tsp_bb_cutting [input_file]
 */
#include "tsp_common.h"
#include <mpi.h>

#ifndef FILENAME
#define FILENAME "Input.txt"
#endif

static int my_rank, psize;

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

static void broadcast_global_upper_bound(float& gub) {
    float local_gub = gub;
    MPI_Allreduce(&local_gub, &gub, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    const char* input_file = (argc > 1) ? argv[1] : FILENAME;
    CityGraph city_graph; memset(&city_graph, 0, sizeof(CityGraph));

    if (my_rank == 0) {
        if (!parse_tsp_input(input_file, &city_graph)) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Bcast(&city_graph.n_city, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(city_graph.city, MAXSIZE * sizeof(CityNode), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(city_graph.dis, MAXSIZE * MAXSIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int n = city_graph.n_city;
    float global_upper_bound = INF;
    long long nodes_processed = 0, nodes_discarded_pq_full = 0;
    PriorityQueue local_pq;
    ProcessState current_state = IDLE;

    // Same initial distribution as parallel_tsp_bb.cpp
    if (my_rank == 0) {
        std::cout << "=== Parallel TSP B&B (PQ Cutting) | " << psize << " procs | PQ limit: "
                  << MAX_PQ_SIZE_LIMIT << " ===" << std::endl;
        PriorityQueue init_pq;
        BBNode* root = create_root_node(&city_graph); init_pq.push(root);
        std::vector<BBNode*> subs;
        int target = psize * 2;
        while (!init_pq.empty() && (int)subs.size() < target) {
            BBNode* nd = init_pq.top(); init_pq.pop(); nodes_processed++;
            if (nd->is_solution(n)) {
                float c = nd->hamilton_cost(&city_graph);
                if (c < global_upper_bound) global_upper_bound = c;
                delete nd; continue;
            }
            for (int c = 0; c < n; c++) {
                if (!nd->visited[c]) {
                    BBNode* ch = nd->visit_city(c, &city_graph);
                    if (ch) subs.push_back(ch);
                }
            }
            delete nd;
        }
        while (!init_pq.empty()) { subs.push_back(init_pq.top()); init_pq.pop(); }

        for (int i = 0; i < (int)subs.size(); i++) {
            int dest = i % psize;
            if (dest == 0) {
                if ((int)local_pq.size() < MAX_PQ_SIZE_LIMIT) local_pq.push(subs[i]);
                else { delete subs[i]; nodes_discarded_pq_full++; }
            } else {
                char buf[PACK_BUF_SIZE]; pack_bbnode(subs[i], buf, PACK_BUF_SIZE);
                MPI_Send(buf, PACK_BUF_SIZE, MPI_CHAR, dest, TAG_INITIAL_WORK, MPI_COMM_WORLD);
                delete subs[i];
            }
        }
        for (int r = 1; r < psize; r++) {
            bool got = false;
            for (int i = r; i < (int)subs.size(); i += psize) { got = true; break; }
            if (!got) MPI_Send(nullptr, 0, MPI_CHAR, r, TAG_INITIAL_WORK, MPI_COMM_WORLD);
        }
        current_state = local_pq.empty() ? IDLE : ACTIVE;
    } else {
        MPI_Status status; char buf[PACK_BUF_SIZE];
        MPI_Recv(buf, PACK_BUF_SIZE, MPI_CHAR, 0, TAG_INITIAL_WORK, MPI_COMM_WORLD, &status);
        int count; MPI_Get_count(&status, MPI_CHAR, &count);
        if (count > 0) {
            local_pq.push(unpack_bbnode(buf, PACK_BUF_SIZE));
            current_state = ACTIVE;
            int flag = 1;
            while (flag) {
                MPI_Iprobe(0, TAG_INITIAL_WORK, MPI_COMM_WORLD, &flag, &status);
                if (flag) {
                    MPI_Recv(buf, PACK_BUF_SIZE, MPI_CHAR, 0, TAG_INITIAL_WORK, MPI_COMM_WORLD, &status);
                    MPI_Get_count(&status, MPI_CHAR, &count);
                    if (count > 0 && (int)local_pq.size() < MAX_PQ_SIZE_LIMIT)
                        local_pq.push(unpack_bbnode(buf, PACK_BUF_SIZE));
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TimePoint start_time = Clock::now();

    int terminated_count = 0;
    bool globally_terminated = false, local_termination_signal_sent = false;
    long long iter = 0;

    while (!globally_terminated) {
        if (!local_pq.empty()) {
            current_state = ACTIVE;
            BBNode* node = local_pq.top(); local_pq.pop(); nodes_processed++;
            if (node->estimated_cost() >= global_upper_bound) { delete node; }
            else if (node->is_solution(n)) {
                float cost = node->hamilton_cost(&city_graph);
                if (cost < global_upper_bound) global_upper_bound = cost;
                delete node;
            } else {
                for (int c = 0; c < n; c++) {
                    if (!node->visited[c]) {
                        BBNode* child = node->visit_city(c, &city_graph);
                        if (child && child->estimated_cost() < global_upper_bound) {
                            // KEY DIFFERENCE: cap PQ size
                            if ((int)local_pq.size() < MAX_PQ_SIZE_LIMIT)
                                local_pq.push(child);
                            else { delete child; nodes_discarded_pq_full++; }
                        } else if (child) delete child;
                    }
                }
                delete node;
            }
        } else { current_state = IDLE; }

        iter++;
        if (iter % GUB_UPDATE_INTERVAL == 0) broadcast_global_upper_bound(global_upper_bound);

        if (current_state == IDLE && !local_termination_signal_sent) {
            if (my_rank == 0) terminated_count++;
            else { int d = my_rank; MPI_Send(&d, 1, MPI_INT, 0, TAG_TERM_CHECK, MPI_COMM_WORLD); }
            local_termination_signal_sent = true;
        }
        if (my_rank == 0) {
            int flag = 1;
            while (flag) {
                MPI_Status st; MPI_Iprobe(MPI_ANY_SOURCE, TAG_TERM_CHECK, MPI_COMM_WORLD, &flag, &st);
                if (flag) { int d; MPI_Recv(&d, 1, MPI_INT, st.MPI_SOURCE, TAG_TERM_CHECK, MPI_COMM_WORLD, &st); terminated_count++; }
            }
            if (terminated_count >= psize) globally_terminated = true;
        }
        MPI_Bcast(&globally_terminated, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    TimePoint end_time = Clock::now();
    float final_gub = global_upper_bound;
    MPI_Allreduce(MPI_IN_PLACE, &final_gub, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

    long long all_nodes[64] = {0};
    MPI_Gather(&nodes_processed, 1, MPI_LONG_LONG, all_nodes, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    long long all_discarded[64] = {0};
    MPI_Gather(&nodes_discarded_pq_full, 1, MPI_LONG_LONG, all_discarded, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double exec_time = elapsed_seconds(start_time, end_time);
        long long total = 0, max_n = 0, min_n = LLONG_MAX, total_disc = 0;
        for (int i = 0; i < psize; i++) {
            total += all_nodes[i]; total_disc += all_discarded[i];
            if (all_nodes[i] > max_n) max_n = all_nodes[i];
            if (all_nodes[i] < min_n) min_n = all_nodes[i];
        }
        double avg = (double)total / psize;
        double var = 0; for (int i = 0; i < psize; i++) var += (all_nodes[i]-avg)*(all_nodes[i]-avg);
        double stddev = sqrt(var / psize);

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Optimal cost:       " << final_gub << std::endl;
        std::cout << "Execution time:     " << exec_time << " s" << std::endl;
        std::cout << "Total nodes:        " << total << std::endl;
        std::cout << "Total discarded:    " << total_disc << std::endl;
        std::cout << "Std dev:            " << stddev << std::endl;
    }
    MPI_Finalize();
    return 0;
}
