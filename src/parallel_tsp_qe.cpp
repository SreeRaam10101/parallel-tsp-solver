/**
 * parallel_tsp_qe.cpp
 * Parallel B&B with Quality-Equalizing (QE) dynamic load balancing.
 * Uses direct struct send for BBNode (homogeneous cluster assumption).
 * Safer component-wise broadcast for CityGraph.
 * Usage: mpirun -np <N> ./parallel_tsp_qe [input_file]
 */
#include "tsp_common.h"
#include <mpi.h>

#ifndef FILENAME
#define FILENAME "Input.txt"
#endif

static int my_rank, psize;
static int neighbor_ranks[2];           // [0]=prev, [1]=next in ring
static float neighbor_thresholds[2];    // Received QE thresholds from neighbors
static bool requested_work_from[2];     // Whether we've requested work from each neighbor

static ProcessState current_state;
static bool local_termination_signal_sent;
static bool received_term_confirm[2];
static int terminated_count;
static bool globally_terminated;

static CityGraph* city_graph_ptr;
static float global_upper_bound;
static PriorityQueue local_pq;
static long long nodes_processed;

// ---------- QE: Send threshold to neighbors ----------
static void send_threshold_cost() {
    float threshold = INF;
    if (!local_pq.empty()) {
        // Get the S-th best node's estimated cost as threshold
        std::vector<BBNode*> temp;
        for (int i = 0; i < QE_SPAN_S && !local_pq.empty(); i++) {
            temp.push_back(local_pq.top()); local_pq.pop();
        }
        if (!temp.empty()) threshold = temp.back()->estimated_cost();
        for (auto* nd : temp) local_pq.push(nd);
    }
    for (int i = 0; i < 2; i++)
        MPI_Send(&threshold, 1, MPI_FLOAT, neighbor_ranks[i], TAG_QE_THRESHOLD, MPI_COMM_WORLD);
}

// ---------- QE: Request work from a neighbor ----------
static void request_work(int neighbor_rank) {
    int my_r = my_rank;
    MPI_Send(&my_r, 1, MPI_INT, neighbor_rank, TAG_QE_WORK_REQUEST, MPI_COMM_WORLD);
}

// ---------- QE: Send work to a requester ----------
static void send_work(int requester_rank) {
    if ((int)local_pq.size() <= WORK_TRANSFER_COUNT) {
        // Not enough surplus; send 0-count message
        int count = 0;
        MPI_Send(&count, 1, MPI_INT, requester_rank, TAG_QE_WORK_TRANSFER, MPI_COMM_WORLD);
        return;
    }

    // Keep the best node, send 2nd through S-th best
    BBNode* best = local_pq.top(); local_pq.pop();
    std::vector<BBNode*> to_send;
    for (int i = 0; i < WORK_TRANSFER_COUNT && !local_pq.empty(); i++) {
        to_send.push_back(local_pq.top()); local_pq.pop();
    }
    local_pq.push(best);

    int count = (int)to_send.size();
    MPI_Send(&count, 1, MPI_INT, requester_rank, TAG_QE_WORK_TRANSFER, MPI_COMM_WORLD);
    for (auto* nd : to_send) {
        MPI_Send(nd, sizeof(BBNode), MPI_BYTE, requester_rank, TAG_QE_WORK_TRANSFER, MPI_COMM_WORLD);
        delete nd;
    }
}

// ---------- QE: Load balancing check ----------
static void perform_load_balancing_check() {
    send_threshold_cost();

    float my_best = local_pq.empty() ? INF : local_pq.top()->estimated_cost();
    for (int i = 0; i < 2; i++) {
        if ((local_pq.empty() || my_best > neighbor_thresholds[i]) && !requested_work_from[i]) {
            request_work(neighbor_ranks[i]);
            requested_work_from[i] = true;
        }
    }
}

// ---------- Termination protocol ----------
static void send_termination_signal() {
    current_state = TERM_SENT;
    local_termination_signal_sent = true;
    for (int r = 0; r < psize; r++) {
        if (r != my_rank) {
            int dummy = my_rank;
            MPI_Send(&dummy, 1, MPI_INT, r, TAG_TERM_CHECK, MPI_COMM_WORLD);
        }
    }
    if (my_rank == 0) terminated_count++;
}

static void send_wakeup_signal() {
    local_termination_signal_sent = false;
    current_state = ACTIVE;
    for (int r = 0; r < psize; r++) {
        if (r != my_rank) {
            int dummy = my_rank;
            MPI_Send(&dummy, 1, MPI_INT, r, TAG_TERM_WAKEUP, MPI_COMM_WORLD);
        }
    }
    // Send pending confirmations
    for (int i = 0; i < 2; i++) {
        if (!received_term_confirm[i]) {
            int dummy = my_rank;
            MPI_Send(&dummy, 1, MPI_INT, neighbor_ranks[i], TAG_TERM_CONFIRM, MPI_COMM_WORLD);
            received_term_confirm[i] = true;
        }
    }
}

static void check_termination_condition() {
    if (current_state != IDLE || !local_pq.empty()) return;
    if (local_termination_signal_sent) return;
    // Check neighbors don't have useful work
    bool neighbors_done = true;
    for (int k = 0; k < 2; k++) {
        if (neighbor_thresholds[k] < global_upper_bound) { neighbors_done = false; break; }
    }
    if (!neighbors_done) return;
    if (!received_term_confirm[0] || !received_term_confirm[1]) return;
    send_termination_signal();
}

// ---------- Handle incoming MPI messages ----------
static void handle_incoming_messages() {
    int flag = 1;
    MPI_Status status;
    while (flag) {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        if (!flag) break;

        int src = status.MPI_SOURCE, tag = status.MPI_TAG;

        if (tag == TAG_QE_THRESHOLD) {
            float thresh;
            MPI_Recv(&thresh, 1, MPI_FLOAT, src, tag, MPI_COMM_WORLD, &status);
            int idx = (src == neighbor_ranks[0]) ? 0 : 1;
            neighbor_thresholds[idx] = thresh;
            requested_work_from[idx] = false;
        }
        else if (tag == TAG_QE_WORK_REQUEST) {
            int req_rank;
            MPI_Recv(&req_rank, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
            send_work(req_rank);
        }
        else if (tag == TAG_QE_WORK_TRANSFER) {
            int count;
            MPI_Recv(&count, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
            for (int i = 0; i < count; i++) {
                BBNode* nd = new BBNode();
                MPI_Recv(nd, sizeof(BBNode), MPI_BYTE, src, TAG_QE_WORK_TRANSFER, MPI_COMM_WORLD, &status);
                if (nd->estimated_cost() < global_upper_bound) {
                    local_pq.push(nd);
                } else { delete nd; }
            }
            if (count > 0 && !local_pq.empty() && (current_state == IDLE || current_state == TERM_SENT)) {
                send_wakeup_signal();
            }
        }
        else if (tag == TAG_TERM_CHECK) {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
            if (my_rank == 0) terminated_count++;
        }
        else if (tag == TAG_TERM_WAKEUP) {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
            if (my_rank == 0 && terminated_count > 0) terminated_count--;
            if (local_termination_signal_sent) {
                local_termination_signal_sent = false;
                if (current_state == TERM_SENT) current_state = IDLE;
            }
        }
        else if (tag == TAG_TERM_CONFIRM) {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
            int idx = (src == neighbor_ranks[0]) ? 0 : 1;
            received_term_confirm[idx] = true;
        }
        else {
            // Consume unknown message
            char junk[4096];
            MPI_Recv(junk, 4096, MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    const char* input_file = (argc > 1) ? argv[1] : FILENAME;
    CityGraph city_graph; memset(&city_graph, 0, sizeof(CityGraph));
    city_graph_ptr = &city_graph;

    // Safer component-wise broadcast
    if (my_rank == 0) {
        if (!parse_tsp_input(input_file, &city_graph)) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Bcast(&city_graph.n_city, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(city_graph.city, MAXSIZE * sizeof(CityNode), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(city_graph.dis, MAXSIZE * MAXSIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int n = city_graph.n_city;
    global_upper_bound = INF;
    nodes_processed = 0;
    current_state = IDLE;
    local_termination_signal_sent = false;
    terminated_count = 0;
    globally_terminated = false;

    // Ring topology neighbors
    neighbor_ranks[0] = (my_rank - 1 + psize) % psize;
    neighbor_ranks[1] = (my_rank + 1) % psize;
    neighbor_thresholds[0] = neighbor_thresholds[1] = INF;
    requested_work_from[0] = requested_work_from[1] = false;
    received_term_confirm[0] = received_term_confirm[1] = true;

    // Initial work distribution (same as basic parallel)
    if (my_rank == 0) {
        std::cout << "=== Parallel TSP B&B with QE | " << psize << " procs ===" << std::endl;
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
            if (dest == 0) { local_pq.push(subs[i]); }
            else { MPI_Send(subs[i], sizeof(BBNode), MPI_BYTE, dest, TAG_INITIAL_WORK, MPI_COMM_WORLD); delete subs[i]; }
        }
        for (int r = 1; r < psize; r++) {
            bool got = false;
            for (int i = r; i < (int)subs.size(); i += psize) { got = true; break; }
            if (!got) MPI_Send(nullptr, 0, MPI_BYTE, r, TAG_INITIAL_WORK, MPI_COMM_WORLD);
        }
        current_state = local_pq.empty() ? IDLE : ACTIVE;
    } else {
        MPI_Status st; BBNode tmp;
        MPI_Recv(&tmp, sizeof(BBNode), MPI_BYTE, 0, TAG_INITIAL_WORK, MPI_COMM_WORLD, &st);
        int cnt; MPI_Get_count(&st, MPI_BYTE, &cnt);
        if (cnt >= (int)sizeof(BBNode)) {
            BBNode* nd = new BBNode(tmp); local_pq.push(nd); current_state = ACTIVE;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TimePoint start_time = Clock::now();
    long long iter = 0;

    // ---------- Main B&B + QE Loop ----------
    while (!globally_terminated) {
        handle_incoming_messages();

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
                        if (child && child->estimated_cost() < global_upper_bound) local_pq.push(child);
                        else if (child) delete child;
                    }
                }
                delete node;
            }
        } else {
            current_state = (current_state == TERM_SENT) ? TERM_SENT : IDLE;
        }

        iter++;
        if (iter % GUB_UPDATE_INTERVAL == 0) {
            float local_gub = global_upper_bound;
            MPI_Allreduce(&local_gub, &global_upper_bound, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        }
        if (iter % LB_CHECK_INTERVAL == 0) perform_load_balancing_check();
        if (current_state == IDLE) check_termination_condition();

        // Global termination check
        if (my_rank == 0 && terminated_count >= psize && local_pq.empty()) {
            globally_terminated = true;
        }
        MPI_Bcast(&globally_terminated, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    TimePoint end_time = Clock::now();
    float final_gub = global_upper_bound;
    MPI_Allreduce(MPI_IN_PLACE, &final_gub, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

    long long all_nodes[64] = {0};
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
        double var = 0; for (int i = 0; i < psize; i++) var += (all_nodes[i]-avg)*(all_nodes[i]-avg);
        double stddev = sqrt(var / psize);

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Optimal cost:   " << final_gub << std::endl;
        std::cout << "Execution time: " << exec_time << " s" << std::endl;
        std::cout << "Total nodes:    " << total << std::endl;
        std::cout << "Std dev:        " << stddev << std::endl;
        std::cout << "\nCSV: " << n << "," << exec_time << "," << total << ","
                  << avg << "," << max_n << "," << min_n << "," << stddev << "," << final_gub << std::endl;
    }
    MPI_Finalize();
    return 0;
}
