#pragma once

#include <cstddef>
#include <string>
#include <vector>

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
    std::size_t indexing_ms = 0;
    std::size_t query_expand_ms = 0;
    std::size_t query_embedding_ms = 0;
    std::size_t embedding_ms = 0;
    std::size_t searching_ms = 0;
    std::size_t reranking_ms = 0;
    std::size_t generation_ms = 0;
    std::size_t total_ms = 0;
};

std::string rag_trim(std::string value);

std::vector<std::string> rag_split_document(
        const std::string & document);

std::string build_generation_prompt(
        const std::string & query,
        const std::vector<std::string> & context_chunks);

std::vector<std::string> build_generation_segments(
        const std::string & query,
        const std::vector<std::string> & sub_queries,
        const std::vector<std::string> & context_chunks);