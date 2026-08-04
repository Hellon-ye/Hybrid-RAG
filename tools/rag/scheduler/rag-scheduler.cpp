#include "rag-scheduler.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <deque>
#include <exception>
#include <limits>

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

RagScheduler::RagScheduler()
    : cpu_worker_count_(4)
    , npu_worker_count_(1)
{}

void RagScheduler::set_worker_counts(
        std::size_t cpu_workers,
        std::size_t npu_workers) {
    cpu_worker_count_ = std::max<std::size_t>(
            1,
            cpu_workers);

    npu_worker_count_ = npu_workers;
}

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
    plan.metrics.clear();

    const bool serial_policy =
            plan.policy == RagSchedulePolicy::Sequential ||
            plan.policy == RagSchedulePolicy::CarrierBaseline;

    const bool allow_stealing =
            plan.policy == RagSchedulePolicy::HeteroParallel ||
            plan.policy == RagSchedulePolicy::NpuCpuCriticalScore;

    const bool critical_score_policy =
            plan.policy == RagSchedulePolicy::NpuCpuCriticalScore;

    const std::size_t cpu_workers =
            serial_policy
                    ? 1
                    : std::max<std::size_t>(
                              1,
                              cpu_worker_count_);

    const std::size_t npu_workers =
            serial_policy
                    ? std::min<std::size_t>(
                              1,
                              npu_worker_count_)
                    : npu_worker_count_;

    std::deque<int> cpu_ready;
    std::deque<int> npu_ready;
    std::map<int, int64_t> enqueue_time;

    int active = 0;

    const auto & queue_for =
            [&](RagBackend backend) -> std::deque<int> & {
        return backend == RagBackend::NPU
                ? npu_ready
                : cpu_ready;
    };

    const auto other_backend =
            [](RagBackend backend) {
        return backend == RagBackend::NPU
                ? RagBackend::CPU
                : RagBackend::NPU;
    };

    const auto worker_available =
            [&](RagBackend backend) {
        if (backend == RagBackend::CPU) {
            return cpu_workers > 0;
        }

        if (backend == RagBackend::NPU) {
            return npu_workers > 0;
        }

        return false;
    };

    const auto backend_allowed =
            [](const RagTaskNode & node,
               RagBackend backend) {
        if (backend != RagBackend::CPU &&
            backend != RagBackend::NPU) {
            return false;
        }

        if (!node.allowed_backends.empty()) {
            return std::find(
                           node.allowed_backends.begin(),
                           node.allowed_backends.end(),
                           backend) !=
                    node.allowed_backends.end();
        }

        if (node.target_backend == RagBackend::Auto) {
            return true;
        }

        return node.target_backend == backend;
    };

    const auto can_run =
            [&](const RagTaskNode & node,
                RagBackend backend) {
        return worker_available(backend) &&
               backend_allowed(node, backend) &&
               find_executor(node.type, backend) != nullptr;
    };

    const auto preferred_backend =
            [&](const RagTaskNode & node) {
        return resolve_backend(
                node.type,
                node.target_backend);
    };

    const auto mark_unrunnable_locked =
            [&](RagTaskNode & node) {
        node.state = RagTaskState::Failed;

        auto & metric = plan.metrics[node.id];

        metric.task_type = node.type;
        metric.preferred_backend =
                preferred_backend(node);
        metric.query_index = node.query_index;
        metric.candidate_index =
                node.candidate_index;
        metric.final_state =
                RagTaskState::Failed;
        metric.error_message =
                "no allowed worker/executor for task";

        propagate_failure(plan, node.id);
    };

    const auto enqueue_ready_locked =
            [&](int task_id) {
        RagTaskNode * node =
                plan.find_node(task_id);

        if (!node ||
            node->state == RagTaskState::Running ||
            node->state == RagTaskState::Completed ||
            node->state == RagTaskState::Failed ||
            node->state == RagTaskState::Cancelled) {
            return;
        }

        const RagBackend preferred =
                preferred_backend(*node);

        RagBackend queue_backend = preferred;

        if (!can_run(*node, queue_backend)) {
            const RagBackend alternate =
                    other_backend(queue_backend);

            if (!can_run(*node, alternate)) {
                mark_unrunnable_locked(*node);
                return;
            }

            queue_backend = alternate;
        }

        node->state = RagTaskState::Ready;

        auto & metric = plan.metrics[task_id];

        metric.task_type = node->type;
        metric.preferred_backend = preferred;
        metric.query_index = node->query_index;
        metric.candidate_index =
                node->candidate_index;
        metric.final_state =
                RagTaskState::Ready;

        enqueue_time[task_id] = now_ms();
        queue_for(queue_backend).push_back(task_id);
    };

    const auto eligible =
            [&](int task_id,
                RagBackend worker_backend,
                bool stealing) {
        const RagTaskNode * node =
                plan.find_node(task_id);

        if (!node ||
            node->state != RagTaskState::Ready) {
            return false;
        }

        if (stealing && !node->stealable) {
            return false;
        }

        return can_run(*node, worker_backend);
    };

    const auto has_eligible =
            [&](const std::deque<int> & queue,
                RagBackend worker_backend,
                bool stealing) {
        for (const int task_id : queue) {
            if (eligible(
                        task_id,
                        worker_backend,
                        stealing)) {
                return true;
            }
        }

        return false;
    };

    const auto pop_queue_locked =
            [&](std::deque<int> & queue,
                RagBackend worker_backend,
                bool stealing) {
        std::size_t selected =
                std::numeric_limits<std::size_t>::max();

        double selected_score =
                std::numeric_limits<double>::lowest();

        for (std::size_t index = 0;
             index < queue.size();) {
            const int task_id = queue[index];
            const RagTaskNode * node =
                    plan.find_node(task_id);

            if (!node ||
                node->state != RagTaskState::Ready) {
                queue.erase(
                        queue.begin() +
                        static_cast<std::ptrdiff_t>(
                                index));
                continue;
            }

            if (!eligible(
                        task_id,
                        worker_backend,
                        stealing)) {
                ++index;
                continue;
            }

            if (!critical_score_policy) {
                selected = index;
                break;
            }

            if (selected ==
                        std::numeric_limits<
                                std::size_t>::max() ||
                node->critical_score >
                        selected_score) {
                selected = index;
                selected_score =
                        node->critical_score;
            }

            ++index;
        }

        if (selected ==
            std::numeric_limits<std::size_t>::max()) {
            return -1;
        }

        const int task_id = queue[selected];

        queue.erase(
                queue.begin() +
                static_cast<std::ptrdiff_t>(
                        selected));

        return task_id;
    };

    const auto cancel_pending_locked = [&]() {
        cpu_ready.clear();
        npu_ready.clear();

        for (auto & node : plan.nodes) {
            if (node.state == RagTaskState::Pending ||
                node.state == RagTaskState::Ready) {
                node.state =
                        RagTaskState::Cancelled;

                auto & metric =
                        plan.metrics[node.id];

                metric.task_type = node.type;
                metric.query_index =
                        node.query_index;
                metric.candidate_index =
                        node.candidate_index;
                metric.final_state =
                        RagTaskState::Cancelled;
            }
        }
    };

    {
        std::lock_guard<std::mutex> lock(plan.mtx);

        for (auto & node : plan.nodes) {
            if (node.pending_deps == 0) {
                enqueue_ready_locked(node.id);
            }
        }
    }

    std::vector<std::thread> workers;
    workers.reserve(cpu_workers + npu_workers);

    const auto worker_loop =
            [&](RagBackend worker_backend,
                int worker_id) {
        while (true) {
            int task_id = -1;
            bool stolen = false;

            RagTaskType task_type =
                    RagTaskType::DocumentEmbedding;

            std::size_t query_index =
                    std::numeric_limits<
                            std::size_t>::max();

            std::size_t candidate_index =
                    std::numeric_limits<
                            std::size_t>::max();

            int64_t queued_at = 0;

            {
                std::unique_lock<std::mutex> lock(
                        plan.mtx);

                plan.cv.wait(
                        lock,
                        [&]() {
                            if (plan.cancel_requested.load()) {
                                return true;
                            }

                            if (all_terminal(plan) &&
                                active == 0) {
                                return true;
                            }

                            if (serial_policy &&
                                active != 0) {
                                return false;
                            }

                            const auto & own_queue =
                                    queue_for(
                                            worker_backend);

                            if (has_eligible(
                                        own_queue,
                                        worker_backend,
                                        false)) {
                                return true;
                            }

                            if (allow_stealing) {
                                const auto & other_queue =
                                        queue_for(
                                                other_backend(
                                                        worker_backend));

                                if (has_eligible(
                                            other_queue,
                                            worker_backend,
                                            true)) {
                                    return true;
                                }
                            }

                            return false;
                        });

                if (plan.cancel_requested.load()) {
                    cancel_pending_locked();
                    return;
                }

                if (all_terminal(plan) &&
                    active == 0) {
                    return;
                }

                if (serial_policy &&
                    active != 0) {
                    continue;
                }

                auto & own_queue =
                        queue_for(worker_backend);

                task_id = pop_queue_locked(
                        own_queue,
                        worker_backend,
                        false);

                if (task_id < 0 &&
                    allow_stealing) {
                    auto & steal_queue =
                            queue_for(
                                    other_backend(
                                            worker_backend));

                    task_id = pop_queue_locked(
                            steal_queue,
                            worker_backend,
                            true);

                    stolen = task_id >= 0;
                }

                if (task_id < 0) {
                    continue;
                }

                RagTaskNode * node =
                        plan.find_node(task_id);

                if (!node ||
                    node->state !=
                            RagTaskState::Ready) {
                    continue;
                }

                task_type = node->type;
                query_index = node->query_index;
                candidate_index =
                        node->candidate_index;

                queued_at =
                        enqueue_time.count(task_id)
                                ? enqueue_time[task_id]
                                : now_ms();

                node->state = RagTaskState::Running;
                ++active;

                auto & metric =
                        plan.metrics[task_id];

                metric.task_type = task_type;
                metric.selected_backend =
                        worker_backend;
                metric.query_index =
                        query_index;
                metric.candidate_index =
                        candidate_index;
                metric.worker_id = worker_id;
                metric.stolen = stolen;
                metric.final_state =
                        RagTaskState::Running;
            }

            const int64_t exec_start = now_ms();

            RagTaskResult result;
            result.task_id = task_id;

            try {
                RagTaskExecutor * executor =
                        find_executor(
                                task_type,
                                worker_backend);

                if (!executor) {
                    result.success = false;
                    result.error_message =
                            "no executor registered";
                } else {
                    RagTaskContext context;

                    context.task_id = task_id;
                    context.request_id =
                            plan.request_id;
                    context.type = task_type;
                    context.backend =
                            worker_backend;
                    context.query_index =
                            query_index;
                    context.candidate_index =
                            candidate_index;
                    context.worker_id =
                            worker_id;
                    context.stolen = stolen;
                    context.runtime =
                            plan.runtime;

                    result =
                            executor->execute(context);
                }
            } catch (const std::exception & error) {
                result.success = false;
                result.error_message =
                        error.what();
            } catch (...) {
                result.success = false;
                result.error_message =
                        "executor threw an unknown exception";
            }

            const int64_t exec_end = now_ms();

            {
                std::lock_guard<std::mutex> lock(
                        plan.mtx);

                RagTaskNode * node =
                        plan.find_node(task_id);

                if (node) {
                    auto & metric =
                            plan.metrics[task_id];

                    metric.queue_wait_ms =
                            exec_start - queued_at;
                    metric.execution_ms =
                            exec_end - exec_start;
                    metric.start_ms = exec_start;
                    metric.end_ms = exec_end;
                    metric.selected_backend =
                            worker_backend;
                    metric.worker_id = worker_id;
                    metric.stolen = stolen;
                    metric.error_message =
                            result.error_message;

                    if (!result.success) {
                        node->state =
                                RagTaskState::Failed;

                        metric.final_state =
                                RagTaskState::Failed;

                        propagate_failure(
                                plan,
                                task_id);
                    } else {
                        node->state =
                                RagTaskState::Completed;

                        metric.final_state =
                                RagTaskState::Completed;

                        if (!plan.cancel_requested.load()) {
                            for (const int successor_id :
                                 node->successors) {
                                RagTaskNode * successor =
                                        plan.find_node(
                                                successor_id);

                                if (!successor ||
                                    successor->state !=
                                            RagTaskState::Pending) {
                                    continue;
                                }

                                --successor->pending_deps;

                                if (successor->pending_deps ==
                                    0) {
                                    enqueue_ready_locked(
                                            successor_id);
                                }
                            }
                        }
                    }
                }

                --active;
            }

            plan.cv.notify_all();
        }
    };

    for (std::size_t index = 0;
         index < cpu_workers;
         ++index) {
        workers.emplace_back(
                worker_loop,
                RagBackend::CPU,
                static_cast<int>(index));
    }

    for (std::size_t index = 0;
         index < npu_workers;
         ++index) {
        workers.emplace_back(
                worker_loop,
                RagBackend::NPU,
                1000 + static_cast<int>(index));
    }

    plan.cv.notify_all();

    for (auto & worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    for (const auto & node : plan.nodes) {
        if (node.state != RagTaskState::Completed) {
            return false;
        }
    }

    return true;
}
