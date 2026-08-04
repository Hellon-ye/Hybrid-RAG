#include "server-rag-executors.h"
#include "server-models.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

RagServerExecutor::RagServerExecutor(server_models * models, RagTaskType type)
    : models_(models), type_(type) {}

RagTaskResult RagServerExecutor::execute(const RagTaskContext & ctx) {
    RagTaskResult result;
    result.task_id = ctx.task_id;
    auto * runtime = static_cast<RagRequestRuntime *>(ctx.runtime);
    if (!models_ || !runtime) {
        result.error_message = "missing RAG runtime context";
        return result;
    }

    try {
        const RagRequest & req = runtime->request;
        if (type_ == RagTaskType::DocumentEmbedding) {
            runtime->chunks = rag_split_document(req.doc);
            if (runtime->chunks.empty()) throw std::runtime_error("document produced no chunks");
            runtime->document_embeddings = models_->request_model_embeddings(req.embedding_model, runtime->chunks);
        } else if (type_ == RagTaskType::QueryExpansion) {
            runtime->expanded_queries = { req.query };
            if (req.enable_query_expansion) {
                const std::string model = req.expansion_model.empty() ? req.generation_model : req.expansion_model;
                const json response = models_->request_model_json(model, "/completion", {
                    {"prompt", build_query_expansion_prompt(req.query)},
                    {"n_predict", std::min<std::size_t>(req.max_tokens, 96)},
                    {"temperature", 0.2F}, {"stream", false}});
                if (!response.contains("content") || !response["content"].is_string()) throw std::runtime_error("invalid query expansion response");
                const auto queries = rag_split_sub_queries(response["content"].get<std::string>());
                if (!queries.empty()) runtime->expanded_queries = queries;
            }
        } else if (type_ == RagTaskType::QueryEmbedding) {
            runtime->query_embeddings = models_->request_model_embeddings(req.embedding_model, runtime->expanded_queries);
        } else if (type_ == RagTaskType::VectorSearch) {
            std::vector<std::size_t> indices(runtime->chunks.size());
            std::iota(indices.begin(), indices.end(), 0);
            runtime->retrieval_results.clear();
            for (const auto & embedding : runtime->query_embeddings) runtime->retrieval_results.push_back(rag_search_inner_product(runtime->document_embeddings, indices, embedding, req.top_k));
        } else if (type_ == RagTaskType::RetrievalMerge || type_ == RagTaskType::CandidateMerge) {
            runtime->retrieved_indices = rag_merge_subquery_hits(runtime->retrieval_results, req.top_k);
        } else if (type_ == RagTaskType::Reranking) {
            std::vector<std::string> candidates;
            for (auto index : runtime->retrieved_indices) if (index < runtime->chunks.size()) candidates.push_back(runtime->chunks[index]);
            runtime->reranked_chunks = candidates;
            if (!req.rerank_model.empty()) {
                const json ranked = models_->request_model_rerank(req.rerank_model, req.query, candidates, req.top_n);
                runtime->reranked_chunks.clear();
                for (const auto & item : ranked) { const auto i = item["index"].get<std::size_t>(); if (i < candidates.size()) runtime->reranked_chunks.push_back(candidates[i]); }
            }
        } else if (type_ == RagTaskType::Generation) {
            runtime->generation_prompt =
                    build_generation_prompt(
                            req.query,
                            runtime->reranked_chunks);

            const json response =
                    models_->request_model_json(
                            req.generation_model,
                            "/completion",
                            {
                                {"model", req.generation_model},
                                {"prompt", runtime->generation_prompt},
                                {"n_predict", req.max_tokens},
                                {"temperature", req.temperature},
                                {"stream", false},
                                {"generation_handoff", true},
                                {"generation_prefill_backend",
                                        req.generation_prefill_backend},
                                {"generation_decode_backend",
                                        req.generation_decode_backend},
                                {"backend_sampling", false},
                            });

            if (!response.contains("content") ||
                    !response["content"].is_string()) {
                throw std::runtime_error(
                        "generation child returned an invalid "
                        "completion response");
            }

            runtime->generation_result =
                    response["content"].get<std::string>();
        } else if (type_ == RagTaskType::Finalize) {
            runtime->final_answer = runtime->generation_result;

            if (runtime->final_answer.empty()) {
                throw std::runtime_error(
                        "Finalize has no CPU Decode answer");
            }
        }
        result.success = true;
    } catch (const std::exception & e) {
        runtime->error = e.what();
        result.error_message = e.what();
    }
    return result;
}
