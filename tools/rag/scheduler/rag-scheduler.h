#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Task types for the HybridRAG pipeline
enum class RagTaskType {
    DocumentEmbedding = 0,
    QueryExpansion,
    QueryEmbedding,
    VectorSearch,
    RetrievalMerge,
    Reranking,
    Generation,
    GenerationPrefill,
    GenerationDecode,
    CandidateMerge,
    Finalize,
};

// Compute backend selection
enum class RagBackend {
    CPU = 0,
    NPU,
    Auto,
};

// Task lifecycle states
enum class RagTaskState {
    Pending = 0,
    Ready,
    Running,
    Completed,
    Failed,
    Cancelled,
};

// Single node in the DAG
struct RagTaskNode {
    int              id;
    RagTaskType      type;
    RagBackend       target_backend;
    std::vector<int> predecessors;
    std::vector<int> successors;
    RagTaskState     state;
    int              pending_deps;

    RagTaskNode()
        : id(-1)
        , type(RagTaskType::DocumentEmbedding)
        , target_backend(RagBackend::Auto)
        , state(RagTaskState::Pending)
        , pending_deps(0)
    {}
};

// Per-task metrics recorded during execution
struct RagTaskMetrics {
    int64_t      queue_wait_ms;
    int64_t      execution_ms;
    RagBackend   selected_backend;
    RagTaskState final_state;
    std::string  error_message;

    RagTaskMetrics()
        : queue_wait_ms(0)
        , execution_ms(0)
        , selected_backend(RagBackend::CPU)
        , final_state(RagTaskState::Pending)
    {}
};

// Execution plan for one RAG request -- holds the full task graph.
// Non-copyable because of internal mutex/cv.
struct RagExecutionPlan {
    int                           request_id;
    void *                        runtime;
    std::vector<RagTaskNode>      nodes;
    std::map<int, RagTaskMetrics> metrics;

    // Scheduler signals / cancellation flag
    std::atomic<bool>        cancel_requested;
    std::mutex               mtx;
    std::condition_variable  cv;

    RagExecutionPlan();

    // Disable copy (mutex is non-copyable)
    RagExecutionPlan(const RagExecutionPlan &)            = delete;
    RagExecutionPlan & operator=(const RagExecutionPlan &) = delete;

    // Add a node; returns false if id already exists
    bool add_node(int id, RagTaskType type, RagBackend backend = RagBackend::Auto);

    // Add directed dependency edge: pred must finish before succ
    bool add_dependency(int pred_id, int succ_id);

    // Cycle detection (DFS); returns true if a cycle exists
    bool has_cycle() const;

    // Topological order; returns empty vector if cycle detected
    std::vector<int> topological_sort() const;

    // Reset node states and initialise pending_deps from predecessor counts
    void init_deps();

    // Return ids of nodes currently in Ready state
    std::vector<int> get_ready_nodes() const;

    RagTaskNode *       find_node(int id);
    const RagTaskNode * find_node(int id) const;
};

// Context passed to an executor when a task is dispatched
struct RagTaskContext {
    int         task_id;
    int         request_id;
    RagTaskType type;
    RagBackend  backend;
    void *      runtime;
};

// Result returned by an executor after running a task
struct RagTaskResult {
    int         task_id;
    bool        success;
    std::string error_message;
    int64_t     execution_ms;

    RagTaskResult() : task_id(-1), success(false), execution_ms(0) {}
};

// Abstract executor interface -- implement one per (task_type, backend) pair
class RagTaskExecutor {
public:
    virtual ~RagTaskExecutor() = default;
    virtual RagTaskResult execute(const RagTaskContext & ctx) = 0;
};

// Mock CPU executor: simulates work with a configurable sleep
class MockCPUWorker : public RagTaskExecutor {
public:
    explicit MockCPUWorker(int sleep_ms = 10);
    RagTaskResult execute(const RagTaskContext & ctx) override;

private:
    int sleep_ms_;
};

// Mock NPU executor: simulates work with a configurable sleep
class MockNPUWorker : public RagTaskExecutor {
public:
    explicit MockNPUWorker(int sleep_ms = 20);
    RagTaskResult execute(const RagTaskContext & ctx) override;

private:
    int sleep_ms_;
};

// Lookup key for the executor registry
struct ExecutorKey {
    RagTaskType type;
    RagBackend  backend;

    bool operator<(const ExecutorKey & o) const
    {
        if (static_cast<int>(type) != static_cast<int>(o.type)) {
            return static_cast<int>(type) < static_cast<int>(o.type);
        }
        return static_cast<int>(backend) < static_cast<int>(o.backend);
    }
};

// The main scheduler: builds and drives execution of a RagExecutionPlan.
class RagScheduler {
public:
    // Register an executor for a specific (task_type, backend) pair.
    // Pointer must remain valid for the lifetime of the scheduler.
    void register_executor(RagTaskType type, RagBackend backend, RagTaskExecutor * executor);

    // Set static backend assignment for a task type
    void set_backend_routing(RagTaskType type, RagBackend backend);

    // Execute the plan, blocking until all tasks are terminal or the plan is cancelled.
    // Returns true only when every task reached Completed.
    bool run(RagExecutionPlan & plan);

    // Request cancellation of a running plan (thread-safe, callable from any thread).
    void cancel(RagExecutionPlan & plan);

private:
    std::map<ExecutorKey, RagTaskExecutor *> executors_;
    std::map<RagTaskType, RagBackend>        routing_;

    RagBackend        resolve_backend(RagTaskType type, RagBackend requested) const;
    RagTaskExecutor * find_executor(RagTaskType type, RagBackend backend) const;
};
