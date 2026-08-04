#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <mutex>
#include <memory>

struct RagRequest {
    std::string doc;
    std::string query;

    std::string mode = "sequential";
    bool enable_query_expansion = false;

    std::string generation_model;
    std::string embedding_model;
    std::string rerank_model;
    std::string expansion_model;

    std::size_t top_k = 20;
    std::size_t top_n = 5;
    std::size_t max_tokens = 128;
    std::size_t generation_decode_steps = 64;
    std::size_t generation_subquery_decode_steps = 64;
    std::size_t generation_candidate_repeats = 1;

    std::string generation_prefill_backend = "auto";
    std::string generation_decode_backend = "auto";

    float temperature = 0.2F;
};

struct RagStageMetrics {
    std::size_t document_embedding_ms = 0;
    std::size_t query_expansion_ms = 0;
    std::size_t query_embedding_ms = 0;
    std::size_t vector_search_ms = 0;
    std::size_t reranking_ms = 0;
    std::size_t generation_ms = 0;
    std::size_t total_ms = 0;
};

struct RagRequestRuntime {
    RagRequest request;
    std::vector<std::string> chunks;
    std::vector<std::vector<float>> document_embeddings;
    std::vector<std::string> expanded_queries;
    std::vector<std::vector<float>> query_embeddings;
    std::vector<std::vector<std::size_t>> retrieval_results;
    std::vector<std::size_t> retrieved_indices;
    std::vector<std::string> reranked_chunks;
    std::string generation_prompt;
    std::string generation_result;
    std::string final_answer;
    RagStageMetrics stage_metrics;
    std::string error;
    std::mutex mutex;
};

std::string rag_trim(std::string value);

std::vector<std::string> rag_split_sub_queries(
        const std::string & expanded,
        std::size_t max_subqueries = 3,
        std::size_t max_subquery_chars = 128);

std::string build_query_expansion_prompt(
        const std::string & query);

std::vector<std::string> rag_split_document(
        const std::string & document);

std::vector<std::size_t> rag_search_inner_product(
        const std::vector<std::vector<float>> & doc_embeddings,
        const std::vector<std::size_t> & doc_embedding_source_indices,
        const std::vector<float> & query_embedding,
        std::size_t top_k);

std::vector<std::size_t> rag_merge_subquery_hits(
        const std::vector<std::vector<std::size_t>> & per_query_hits,
        std::size_t top_k);

std::string build_generation_prompt(
        const std::string & query,
        const std::vector<std::string> & context_chunks);

std::vector<std::string> build_generation_segments(
        const std::string & query,
        const std::vector<std::string> & sub_queries,
        const std::vector<std::string> & context_chunks);
