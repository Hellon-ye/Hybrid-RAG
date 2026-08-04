#include "rag-scheduler.h"

#include <cassert>
#include <cstdio>

// ---- internal helpers ----

static int64_t now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// ---- RagExecutionPlan ----

RagExecutionPlan::RagExecutionPlan() : request_id(0), runtime(nullptr), cancel_requested(false)
{}

RagTaskNode * RagExecutionPlan::find_node(int id)
{
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i].id == id) {
            return &nodes[i];
        }
    }
    return nullptr;
}

const RagTaskNode * RagExecutionPlan::find_node(int id) const
{
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i].id == id) {
            return &nodes[i];
        }
    }
    return nullptr;
}

bool RagExecutionPlan::add_node(int id, RagTaskType type, RagBackend backend)
{
    if (find_node(id) != nullptr) {
        return false; // duplicate id
    }
    RagTaskNode node;
    node.id             = id;
    node.type           = type;
    node.target_backend = backend;
    node.state          = RagTaskState::Pending;
    node.pending_deps   = 0;
    nodes.push_back(node);
    return true;
}

bool RagExecutionPlan::add_dependency(int pred_id, int succ_id)
{
    if (pred_id == succ_id) {
        return false; // self-loop
    }
    RagTaskNode * pred = find_node(pred_id);
    RagTaskNode * succ = find_node(succ_id);
    if (!pred || !succ) {
        return false; // missing node
    }
    // Deduplicate
    for (int i = 0; i < (int)pred->successors.size(); i++) {
        if (pred->successors[i] == succ_id) {
            return true; // already registered
        }
    }
    pred->successors.push_back(succ_id);
    succ->predecessors.push_back(pred_id);
    return true;
}

// Iterative DFS to detect cycles.
// color: 0=white (unvisited), 1=gray (in stack), 2=black (done)
static bool dfs_cycle(int start_id, const std::vector<RagTaskNode> & nodes,
                       std::map<int, int> & color)
{
    struct Frame {
        int id;
        int succ_idx;
    };

    std::vector<Frame> stack;
    stack.push_back({start_id, 0});
    color[start_id] = 1;

    while (!stack.empty()) {
        Frame & f = stack.back();

        const RagTaskNode * node = nullptr;
        for (int i = 0; i < (int)nodes.size(); i++) {
            if (nodes[i].id == f.id) {
                node = &nodes[i];
                break;
            }
        }

        if (!node || f.succ_idx >= (int)node->successors.size()) {
            color[f.id] = 2;
            stack.pop_back();
            continue;
        }

        int succ_id    = node->successors[f.succ_idx];
        f.succ_idx++;

        int succ_color = 0;
        if (color.count(succ_id)) {
            succ_color = color[succ_id];
        }

        if (succ_color == 1) {
            return true; // back edge -> cycle
        }
        if (succ_color == 0) {
            color[succ_id] = 1;
            stack.push_back({succ_id, 0});
        }
    }
    return false;
}

bool RagExecutionPlan::has_cycle() const
{
    std::map<int, int> color;
    for (int i = 0; i < (int)nodes.size(); i++) {
        color[nodes[i].id] = 0;
    }
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (color[nodes[i].id] == 0) {
            if (dfs_cycle(nodes[i].id, nodes, color)) {
                return true;
            }
        }
    }
    return false;
}

std::vector<int> RagExecutionPlan::topological_sort() const
{
    // Kahn's algorithm (BFS-based)
    std::map<int, int> in_degree;
    for (int i = 0; i < (int)nodes.size(); i++) {
        in_degree[nodes[i].id] = 0;
    }
    for (int i = 0; i < (int)nodes.size(); i++) {
        for (int j = 0; j < (int)nodes[i].successors.size(); j++) {
            in_degree[nodes[i].successors[j]]++;
        }
    }

    std::queue<int> q;
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (in_degree[nodes[i].id] == 0) {
            q.push(nodes[i].id);
        }
    }

    std::vector<int> order;
    while (!q.empty()) {
        int id = q.front();
        q.pop();
        order.push_back(id);

        const RagTaskNode * node = find_node(id);
        if (!node) {
            continue;
        }
        for (int j = 0; j < (int)node->successors.size(); j++) {
            int succ = node->successors[j];
            in_degree[succ]--;
            if (in_degree[succ] == 0) {
                q.push(succ);
            }
        }
    }

    if ((int)order.size() != (int)nodes.size()) {
        return std::vector<int>(); // cycle present
    }
    return order;
}

void RagExecutionPlan::init_deps()
{
    for (int i = 0; i < (int)nodes.size(); i++) {
        nodes[i].pending_deps = static_cast<int>(nodes[i].predecessors.size());
        nodes[i].state        = RagTaskState::Pending;
    }
}

std::vector<int> RagExecutionPlan::get_ready_nodes() const
{
    std::vector<int> ready;
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i].state == RagTaskState::Ready) {
            ready.push_back(nodes[i].id);
        }
    }
    return ready;
}

// ---- Mock workers ----

MockCPUWorker::MockCPUWorker(int sleep_ms) : sleep_ms_(sleep_ms)
{}

RagTaskResult MockCPUWorker::execute(const RagTaskContext & ctx)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
    RagTaskResult r;
    r.task_id     = ctx.task_id;
    r.success     = true;
    r.execution_ms = sleep_ms_;
    return r;
}

MockNPUWorker::MockNPUWorker(int sleep_ms) : sleep_ms_(sleep_ms)
{}

RagTaskResult MockNPUWorker::execute(const RagTaskContext & ctx)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
    RagTaskResult r;
    r.task_id     = ctx.task_id;
    r.success     = true;
    r.execution_ms = sleep_ms_;
    return r;
}

// ---- RagScheduler ----

void RagScheduler::register_executor(RagTaskType type, RagBackend backend, RagTaskExecutor * executor)
{
    ExecutorKey key;
    key.type    = type;
    key.backend = backend;
    executors_[key] = executor;
}

void RagScheduler::set_backend_routing(RagTaskType type, RagBackend backend)
{
    routing_[type] = backend;
}

RagBackend RagScheduler::resolve_backend(RagTaskType type, RagBackend requested) const
{
    if (requested != RagBackend::Auto) {
        return requested;
    }
    auto it = routing_.find(type);
    if (it != routing_.end()) {
        return it->second;
    }
    return RagBackend::CPU;
}

RagTaskExecutor * RagScheduler::find_executor(RagTaskType type, RagBackend backend) const
{
    ExecutorKey key;
    key.type    = type;
    key.backend = backend;
    auto it = executors_.find(key);
    if (it != executors_.end()) {
        return it->second;
    }
    return nullptr;
}

void RagScheduler::cancel(RagExecutionPlan & plan)
{
    plan.cancel_requested.store(true);
    plan.cv.notify_all();
}

// BFS propagation: mark all transitive successors of failed_id as Failed.
// Must be called with plan.mtx held.
static void propagate_failure(RagExecutionPlan & plan, int failed_id)
{
    std::queue<int> q;
    const RagTaskNode * src = plan.find_node(failed_id);
    if (!src) {
        return;
    }
    for (int i = 0; i < (int)src->successors.size(); i++) {
        q.push(src->successors[i]);
    }

    while (!q.empty()) {
        int id = q.front();
        q.pop();
        RagTaskNode * node = plan.find_node(id);
        if (!node) {
            continue;
        }
        // Do not override already-terminal states
        if (node->state == RagTaskState::Completed || node->state == RagTaskState::Running) {
            continue;
        }
        node->state                         = RagTaskState::Failed;
        plan.metrics[id].final_state        = RagTaskState::Failed;
        plan.metrics[id].error_message      = "predecessor failed";

        for (int j = 0; j < (int)node->successors.size(); j++) {
            q.push(node->successors[j]);
        }
    }
}

// Check whether every node in the plan has reached a terminal state.
static bool all_terminal(const RagExecutionPlan & plan)
{
    for (int i = 0; i < (int)plan.nodes.size(); i++) {
        RagTaskState s = plan.nodes[i].state;
        if (s != RagTaskState::Completed &&
            s != RagTaskState::Failed    &&
            s != RagTaskState::Cancelled) {
            return false;
        }
    }
    return true;
}

bool RagScheduler::run(RagExecutionPlan & plan)
{
    if (plan.has_cycle()) {
        return false;
    }

    plan.cancel_requested.store(false);
    plan.init_deps();

    // Per-run local state (protected by plan.mtx)
    std::queue<int>       ready_q;
    std::map<int, int64_t> enqueue_time;
    int                   active = 0;

    // Seed the ready queue with nodes that have no predecessors
    {
        std::unique_lock<std::mutex> lk(plan.mtx);
        for (int i = 0; i < (int)plan.nodes.size(); i++) {
            if (plan.nodes[i].pending_deps == 0) {
                plan.nodes[i].state   = RagTaskState::Ready;
                enqueue_time[plan.nodes[i].id] = now_ms();
                ready_q.push(plan.nodes[i].id);
            }
        }
    }

    std::vector<std::thread> threads;

    std::unique_lock<std::mutex> lk(plan.mtx);

    while (true) {
        // -- cancel path: drain ready queue and mark pending nodes --
        if (plan.cancel_requested.load()) {
            while (!ready_q.empty()) {
                int id = ready_q.front();
                ready_q.pop();
                RagTaskNode * n = plan.find_node(id);
                if (n && (n->state == RagTaskState::Ready ||
                          n->state == RagTaskState::Pending)) {
                    n->state                        = RagTaskState::Cancelled;
                    plan.metrics[id].final_state    = RagTaskState::Cancelled;
                }
            }
            for (int i = 0; i < (int)plan.nodes.size(); i++) {
                RagTaskState s = plan.nodes[i].state;
                if (s == RagTaskState::Pending || s == RagTaskState::Ready) {
                    plan.nodes[i].state                         = RagTaskState::Cancelled;
                    plan.metrics[plan.nodes[i].id].final_state  = RagTaskState::Cancelled;
                }
            }
            if (active == 0) {
                break;
            }
            // Wait for still-running tasks to finish
            plan.cv.wait(lk, [&]() { return active == 0; });
            break;
        }

        // -- normal dispatch path --
        while (!ready_q.empty() && !plan.cancel_requested.load()) {
            int           task_id  = ready_q.front();
            ready_q.pop();
            RagTaskNode * node = plan.find_node(task_id);
            if (!node || node->state != RagTaskState::Ready) {
                continue;
            }

            RagTaskType       task_type  = node->type;
            RagBackend        preferred_backend = node->target_backend;
            RagBackend        backend    = resolve_backend(
                    task_type,
                    preferred_backend);
            RagTaskExecutor * exec       = find_executor(
                    task_type,
                    backend);
            int64_t           eq_time    = enqueue_time.count(task_id) ?
                                           enqueue_time[task_id] : now_ms();

            const std::size_t query_index =
                    node->query_index;
            const std::size_t candidate_index =
                    node->candidate_index;

            plan.metrics[task_id].task_type =
                    task_type;
            plan.metrics[task_id].preferred_backend =
                    preferred_backend;
            plan.metrics[task_id].query_index =
                    query_index;
            plan.metrics[task_id].candidate_index =
                    candidate_index;

            node->state = RagTaskState::Running;
            active++;

            // Worker thread: execute task outside the lock, report back under lock
            threads.emplace_back([this, &plan, &lk, &ready_q, &active, &enqueue_time,
                                   task_id, task_type, backend, exec, eq_time,
                                   query_index, candidate_index]()
            {
                int64_t exec_start = now_ms();
                int64_t qwait      = exec_start - eq_time;

                RagTaskResult result;
                result.task_id = task_id;

                if (exec) {
                    RagTaskContext ctx;
                    ctx.task_id         = task_id;
                    ctx.request_id      = plan.request_id;
                    ctx.type            = task_type;
                    ctx.backend         = backend;
                    ctx.query_index     = query_index;
                    ctx.candidate_index = candidate_index;
                    ctx.runtime         = plan.runtime;
                    result = exec->execute(ctx);
                } else {
                    result.success       = false;
                    result.error_message = "no executor registered";
                }

                int64_t exec_end = now_ms();

                // Report completion under the plan mutex
                std::unique_lock<std::mutex> inner(plan.mtx);

                RagTaskNode * n = plan.find_node(task_id);
                if (n) {
                    plan.metrics[task_id].queue_wait_ms =
                            qwait;
                    plan.metrics[task_id].execution_ms =
                            exec_end - exec_start;
                    plan.metrics[task_id].start_ms =
                            exec_start;
                    plan.metrics[task_id].end_ms =
                            exec_end;
                    plan.metrics[task_id].selected_backend =
                            backend;
                    plan.metrics[task_id].error_message =
                            result.error_message;

                    if (!result.success) {
                        n->state                        = RagTaskState::Failed;
                        plan.metrics[task_id].final_state = RagTaskState::Failed;
                        propagate_failure(plan, task_id);
                    } else {
                        n->state                        = RagTaskState::Completed;
                        plan.metrics[task_id].final_state = RagTaskState::Completed;

                        // Release successors only if not cancelled
                        if (!plan.cancel_requested.load()) {
                            for (int j = 0; j < (int)n->successors.size(); j++) {
                                int           succ_id = n->successors[j];
                                RagTaskNode * succ    = plan.find_node(succ_id);
                                if (succ && succ->state == RagTaskState::Pending) {
                                    succ->pending_deps--;
                                    if (succ->pending_deps == 0) {
                                        succ->state         = RagTaskState::Ready;
                                        enqueue_time[succ_id] = now_ms();
                                        ready_q.push(succ_id);
                                    }
                                }
                            }
                        }
                    }
                }

                active--;
                plan.cv.notify_one();
            });
        }

        // -- termination check --
        if (all_terminal(plan) && active == 0) {
            break;
        }
        if (active == 0 && ready_q.empty()) {
            // No runnable tasks and none in flight -- stuck or done
            break;
        }

        // Wait for any completion or cancel signal
        plan.cv.wait(lk, [&]() {
            return active == 0 || !ready_q.empty() || plan.cancel_requested.load();
        });
    }

    lk.unlock();

    for (int i = 0; i < (int)threads.size(); i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }

    // Success only when every node completed
    for (int i = 0; i < (int)plan.nodes.size(); i++) {
        if (plan.nodes[i].state != RagTaskState::Completed) {
            return false;
        }
    }
    return true;
}
