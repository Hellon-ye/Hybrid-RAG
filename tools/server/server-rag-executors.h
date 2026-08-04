#pragma once

#include "rag-scheduler.h"
#include "server-rag.h"

struct server_models;

class RagServerExecutor : public RagTaskExecutor {
public:
    RagServerExecutor(server_models * models, RagTaskType type);
    RagTaskResult execute(const RagTaskContext & ctx) override;

private:
    server_models * models_;
    RagTaskType type_;
};

using DocumentEmbeddingExecutor = RagServerExecutor;
using QueryExpansionExecutor = RagServerExecutor;
using QueryEmbeddingExecutor = RagServerExecutor;
using VectorSearchExecutor = RagServerExecutor;
using RetrievalMergeExecutor = RagServerExecutor;
using RerankingExecutor = RagServerExecutor;
using GenerationExecutor = RagServerExecutor;
using FinalizeExecutor = RagServerExecutor;
