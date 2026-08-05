#include "rag-scheduler.h"

#include <cassert>
#include <vector>

struct Runtime {
    int value = 0;
    std::vector<int> order;
};

struct Worker : RagTaskExecutor {
    bool fail = false;
    RagTaskResult execute(const RagTaskContext & ctx) override {
        auto * runtime = static_cast<Runtime *>(ctx.runtime);
        assert(runtime != nullptr);
        runtime->value++;
        runtime->order.push_back(ctx.task_id);
        RagTaskResult result;
        result.task_id = ctx.task_id;
        result.success = !fail;
        if (fail) result.error_message = "forced failure";
        return result;
    }
};

int main() {
    Runtime first, second;
    Worker worker;
    RagScheduler scheduler;
    scheduler.register_executor(RagTaskType::DocumentEmbedding, RagBackend::CPU, &worker);

    RagExecutionPlan plan1;
    plan1.runtime = &first;
    plan1.add_node(1, RagTaskType::DocumentEmbedding, RagBackend::CPU);
    assert(scheduler.run(plan1));
    assert(first.value == 1 && second.value == 0);

    RagExecutionPlan plan2;
    plan2.runtime = &second;
    plan2.add_node(1, RagTaskType::DocumentEmbedding, RagBackend::CPU);
    assert(scheduler.run(plan2));
    assert(first.value == 1 && second.value == 1);

    worker.fail = true;
    RagExecutionPlan failed;
    failed.runtime = &first;
    failed.add_node(1, RagTaskType::DocumentEmbedding, RagBackend::CPU);
    assert(!scheduler.run(failed));

    worker.fail = false;

    Runtime critical_priority;
    RagScheduler priority_scheduler;
    priority_scheduler.set_worker_counts(1, 0);
    priority_scheduler.register_executor(
            RagTaskType::DocumentEmbedding,
            RagBackend::CPU,
            &worker);

    RagExecutionPlan priority_plan;
    priority_plan.runtime = &critical_priority;
    priority_plan.policy =
            RagSchedulePolicy::NpuCpuCriticalScore;

    priority_plan.add_node(
            1,
            RagTaskType::DocumentEmbedding,
            RagBackend::CPU);
    priority_plan.add_node(
            2,
            RagTaskType::DocumentEmbedding,
            RagBackend::CPU);
    priority_plan.add_node(
            3,
            RagTaskType::DocumentEmbedding,
            RagBackend::CPU);

    priority_plan.find_node(1)->critical_score = 1.0;
    priority_plan.find_node(2)->critical_score = 100.0;
    priority_plan.find_node(3)->critical_score = 10.0;

    assert(priority_scheduler.run(priority_plan));
    assert((
            critical_priority.order ==
            std::vector<int>{2, 3, 1}));

    return 0;
}
