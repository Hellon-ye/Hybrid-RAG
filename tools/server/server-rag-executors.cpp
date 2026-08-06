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
            runtime->expanded_queries.clear();
            runtime->expanded_queries.push_back(req.query);

            if (req.enable_query_expansion &&
                req.max_expanded_queries > 1) {
                const std::string model =
                        req.expansion_model.empty()
                                ? req.generation_model
                                : req.expansion_model;

                const json response =
                        models_->request_model_json(
                                model,
                                "/completion",
                                {
                                    {
                                        "prompt",
                                        build_query_expansion_prompt(
                                                req.query),
                                    },
                                    {
                                        "n_predict",
                                        std::min<std::size_t>(
                                                req.max_tokens,
                                                96),
                                    },
                                    {"temperature", 0.2F},
                                    {"stream", false},
                                });

                if (!response.contains("content") ||
                    !response["content"].is_string()) {
                    throw std::runtime_error(
                            "invalid query expansion response");
                }

                const auto queries =
                        rag_split_sub_queries(
                                response["content"]
                                        .get<std::string>(),
                                req.max_expanded_queries - 1);

                for (const auto & query : queries) {
                    if (query.empty() || query == req.query) {
                        continue;
                    }

                    if (std::find(
                                runtime->expanded_queries.begin(),
                                runtime->expanded_queries.end(),
                                query) !=
                        runtime->expanded_queries.end()) {
                        continue;
                    }

                    runtime->expanded_queries.push_back(query);

                    if (runtime->expanded_queries.size() >=
                        req.max_expanded_queries) {
                        break;
                    }
                }
            }
        } else if (type_ == RagTaskType::QueryEmbedding) {
            if (ctx.query_index == RAG_INVALID_INDEX ||
                ctx.query_index >=
                        runtime->expanded_queries.size() ||
                ctx.query_index >=
                        runtime->query_branches.size()) {
                throw std::runtime_error(
                        "QueryEmbedding received an invalid "
                        "query branch index");
            }

            const std::string & query =
                    runtime->expanded_queries.at(
                            ctx.query_index);

            auto embeddings =
                    models_->request_model_embeddings(
                            req.embedding_model,
                            {query});

            if (embeddings.size() != 1 ||
                embeddings.front().empty()) {
                throw std::runtime_error(
                        "QueryEmbedding expected exactly one "
                        "non-empty embedding");
            }

            auto & branch =
                    runtime->query_branches.at(
                            ctx.query_index);

            branch.query_index = ctx.query_index;
            branch.query = query;
            branch.embedding =
                    std::move(embeddings.front());
            branch.error.clear();
            branch.success = false;
        } else if (type_ == RagTaskType::VectorSearch) {
            if (ctx.query_index == RAG_INVALID_INDEX ||
                ctx.query_index >=
                        runtime->query_branches.size()) {
                throw std::runtime_error(
                        "VectorSearch received an invalid "
                        "query branch index");
            }

            auto & branch =
                    runtime->query_branches.at(
                            ctx.query_index);

            if (branch.embedding.empty()) {
                throw std::runtime_error(
                        "VectorSearch has no query embedding");
            }

            std::vector<std::size_t> indices(
                    runtime->chunks.size());

            std::iota(
                    indices.begin(),
                    indices.end(),
                    0);

            branch.retrieval_indices =
                    rag_search_inner_product(
                            runtime->document_embeddings,
                            indices,
                            branch.embedding,
                            req.top_k);

            branch.success = true;
            branch.error.clear();
        } else if (
                type_ == RagTaskType::RetrievalMerge) {
            runtime->query_embeddings.clear();
            runtime->retrieval_results.clear();

            runtime->query_embeddings.reserve(
                    runtime->query_branches.size());

            runtime->retrieval_results.reserve(
                    runtime->query_branches.size());

            for (const auto & branch :
                 runtime->query_branches) {
                if (!branch.success) {
                    throw std::runtime_error(
                            "RetrievalMerge received an "
                            "incomplete query branch");
                }

                runtime->query_embeddings.push_back(
                        branch.embedding);

                runtime->retrieval_results.push_back(
                        branch.retrieval_indices);
            }

            runtime->retrieved_indices =
                    rag_merge_subquery_hits(
                            runtime->retrieval_results,
                            req.top_k);
        } else if (type_ == RagTaskType::Reranking) {
            std::vector<std::string> candidates;
            for (auto index : runtime->retrieved_indices) if (index < runtime->chunks.size()) candidates.push_back(runtime->chunks[index]);
            runtime->reranked_chunks = candidates;
            if (!req.rerank_model.empty()) {
                const json ranked = models_->request_model_rerank(req.rerank_model, req.query, candidates, req.top_n);
                runtime->reranked_chunks.clear();
                for (const auto & item : ranked) { const auto i = item["index"].get<std::size_t>(); if (i < candidates.size()) runtime->reranked_chunks.push_back(candidates[i]); }
            }
        } else if (
                type_ == RagTaskType::GenerationPrefill) {
            if (ctx.candidate_index == RAG_INVALID_INDEX ||
                ctx.candidate_index >=
                        runtime->generation_candidates.size()) {
                throw std::runtime_error(
                        "GenerationPrefill received an invalid "
                        "candidate index");
            }

            auto & candidate =
                    runtime->generation_candidates.at(
                            ctx.candidate_index);

            candidate.candidate_index =
                    ctx.candidate_index;

            candidate.source_query_index =
                    RAG_INVALID_INDEX;

            candidate.seed =
                    req.seed +
                    static_cast<std::uint32_t>(
                            ctx.candidate_index);

            candidate.prompt =
                    build_generation_prompt(
                            req.query,
                            runtime->reranked_chunks);

            candidate.content.clear();
            candidate.stop_reason.clear();
            candidate.error.clear();
            candidate.generated_tokens = 0;
            candidate.generation_handoff_handle = 0;
            candidate.generation_handoff_state_bytes = 0;
            candidate.generation_handoff_logits_count = 0;
            candidate
                    .generation_handoff_producer_backend
                    .clear();
            candidate.prefill_success = false;
            candidate.success = false;

            try {
                const json response =
                        models_->request_model_json(
                                req.generation_model,
                                "/completion",
                                {
                                    {
                                        "model",
                                        req.generation_model,
                                    },
                                    {
                                        "prompt",
                                        candidate.prompt,
                                    },
                                    {
                                        "n_predict",
                                        req.max_tokens,
                                    },
                                    {"n_cmpl", 1},
                                    {
                                        "temperature",
                                        req.temperature,
                                    },
                                    {
                                        "seed",
                                        candidate.seed,
                                    },
                                    {"stream", false},
                                    {"cache_prompt", false},
                                    {
                                        "generation_handoff",
                                        true,
                                    },
                                    {
                                        "generation_prefill_only",
                                        true,
                                    },
                                    {
                                        "generation_prefill_backend",
                                        req
                                                .generation_prefill_backend,
                                    },
                                    {
                                        "generation_decode_backend",
                                        req
                                                .generation_decode_backend,
                                    },
                                    {
                                        "backend_sampling",
                                        false,
                                    },
                                });

                if (!response.contains(
                            "generation_handoff_handle")) {
                    throw std::runtime_error(
                            "GenerationPrefill response is "
                            "missing generation_handoff_handle");
                }

                const json & handle_value =
                        response[
                            "generation_handoff_handle"];

                if (!handle_value.is_number_unsigned() &&
                    !handle_value.is_number_integer()) {
                    throw std::runtime_error(
                            "GenerationPrefill returned an "
                            "invalid generation_handoff_handle");
                }

                candidate.generation_handoff_handle =
                        handle_value.get<std::uint64_t>();

                if (candidate.generation_handoff_handle == 0) {
                    throw std::runtime_error(
                            "GenerationPrefill returned handle 0");
                }

                if (response.contains(
                            "generation_handoff_state_bytes") &&
                    response[
                        "generation_handoff_state_bytes"]
                            .is_number_integer()) {
                    const long long value =
                            response[
                                "generation_handoff_state_bytes"]
                                    .get<long long>();

                    if (value > 0) {
                        candidate
                                .generation_handoff_state_bytes =
                                static_cast<std::size_t>(
                                        value);
                    }
                }

                if (response.contains(
                            "generation_handoff_logits_count") &&
                    response[
                        "generation_handoff_logits_count"]
                            .is_number_integer()) {
                    const long long value =
                            response[
                                "generation_handoff_logits_count"]
                                    .get<long long>();

                    if (value > 0) {
                        candidate
                                .generation_handoff_logits_count =
                                static_cast<std::size_t>(
                                        value);
                    }
                }

                if (response.contains(
                            "generation_handoff_producer_backend") &&
                    response[
                        "generation_handoff_producer_backend"]
                            .is_string()) {
                    candidate
                            .generation_handoff_producer_backend =
                            response[
                                "generation_handoff_producer_backend"]
                                    .get<std::string>();
                }

                candidate.prefill_success = true;
            } catch (const std::exception & error) {
                // A failed candidate must not fail the complete DAG.
                // Its Decode node will observe prefill_success=false,
                // skip the HTTP request, and let CandidateMerge choose
                // another successful candidate.
                candidate.error = error.what();
                candidate.prefill_success = false;
                candidate.success = false;
            }
        } else if (
                type_ == RagTaskType::GenerationDecode) {
            if (ctx.candidate_index == RAG_INVALID_INDEX ||
                ctx.candidate_index >=
                        runtime->generation_candidates.size()) {
                throw std::runtime_error(
                        "GenerationDecode received an invalid "
                        "candidate index");
            }

            auto & candidate =
                    runtime->generation_candidates.at(
                            ctx.candidate_index);

            candidate.success = false;

            if (!candidate.prefill_success ||
                candidate.generation_handoff_handle == 0) {
                if (candidate.error.empty()) {
                    candidate.error =
                            "GenerationDecode has no valid "
                            "Prefill handoff";
                }
            } else {
                try {
                    const json response =
                            models_->request_model_json(
                                    req.generation_model,
                                    "/completion",
                                    {
                                        {
                                            "model",
                                            req.generation_model,
                                        },
                                        {
                                            "prompt",
                                            candidate.prompt,
                                        },
                                        {
                                            "n_predict",
                                            req.max_tokens,
                                        },
                                        {"n_cmpl", 1},
                                        {"stream", false},
                                        {
                                            "generation_handoff",
                                            true,
                                        },
                                        {
                                            "generation_handoff_handle",
                                            candidate
                                                    .generation_handoff_handle,
                                        },
                                        {
                                            "generation_decode_backend",
                                            req
                                                    .generation_decode_backend,
                                        },
                                        {
                                            "backend_sampling",
                                            false,
                                        },
                                    });

                    if (!response.contains("content") ||
                        !response["content"].is_string()) {
                        throw std::runtime_error(
                                "GenerationDecode returned an "
                                "invalid completion response");
                    }

                    candidate.content =
                            response["content"]
                                    .get<std::string>();

                    if (response.contains(
                                "tokens_predicted") &&
                        response["tokens_predicted"]
                                .is_number_integer()) {
                        const long long value =
                                response["tokens_predicted"]
                                        .get<long long>();

                        if (value > 0) {
                            candidate.generated_tokens =
                                    static_cast<std::size_t>(
                                            value);
                        }
                    } else if (
                        response.contains("timings") &&
                        response["timings"].is_object() &&
                        response["timings"].contains(
                                "predicted_n") &&
                        response["timings"]["predicted_n"]
                                .is_number_integer()) {
                        const long long value =
                                response["timings"]
                                        ["predicted_n"]
                                        .get<long long>();

                        if (value > 0) {
                            candidate.generated_tokens =
                                    static_cast<std::size_t>(
                                            value);
                        }
                    }

                    if (response.contains("stop_type") &&
                        response["stop_type"].is_string()) {
                        candidate.stop_reason =
                                response["stop_type"]
                                        .get<std::string>();
                    } else if (
                        response.contains("stopped_eos") &&
                        response["stopped_eos"]
                                .is_boolean() &&
                        response["stopped_eos"]
                                .get<bool>()) {
                        candidate.stop_reason = "eos";
                    } else if (
                        response.contains("stopped_limit") &&
                        response["stopped_limit"]
                                .is_boolean() &&
                        response["stopped_limit"]
                                .get<bool>()) {
                        candidate.stop_reason = "limit";
                    }

                    if (candidate.content.empty()) {
                        throw std::runtime_error(
                                "GenerationDecode returned "
                                "empty content");
                    }

                    candidate.error.clear();
                    candidate.success = true;
                } catch (const std::exception & error) {
                    candidate.error = error.what();
                    candidate.success = false;
                }
            }
        } else if (type_ == RagTaskType::Generation) {
            if (ctx.candidate_index == RAG_INVALID_INDEX ||
                ctx.candidate_index >=
                        runtime->generation_candidates.size()) {
                throw std::runtime_error(
                        "Generation received an invalid "
                        "candidate index");
            }

            auto & candidate =
                    runtime->generation_candidates.at(
                            ctx.candidate_index);

            candidate.candidate_index =
                    ctx.candidate_index;

            candidate.source_query_index =
                    RAG_INVALID_INDEX;

            candidate.seed =
                    req.seed +
                    static_cast<std::uint32_t>(
                            ctx.candidate_index);

            candidate.prompt =
                    build_generation_prompt(
                            req.query,
                            runtime->reranked_chunks);

            candidate.content.clear();
            candidate.stop_reason.clear();
            candidate.error.clear();
            candidate.generated_tokens = 0;
            candidate.success = false;

            try {
                const json response =
                        models_->request_model_json(
                                req.generation_model,
                                "/completion",
                                {
                                    {
                                        "model",
                                        req.generation_model,
                                    },
                                    {
                                        "prompt",
                                        candidate.prompt,
                                    },
                                    {
                                        "n_predict",
                                        req.max_tokens,
                                    },
                                    {
                                        "temperature",
                                        req.temperature,
                                    },
                                    {
                                        "seed",
                                        candidate.seed,
                                    },
                                    {"stream", false},
                                    {
                                        "generation_handoff",
                                        true,
                                    },
                                    {
                                        "generation_prefill_backend",
                                        req.generation_prefill_backend,
                                    },
                                    {
                                        "generation_decode_backend",
                                        req.generation_decode_backend,
                                    },
                                    {
                                        "backend_sampling",
                                        false,
                                    },
                                });

                if (!response.contains("content") ||
                    !response["content"].is_string()) {
                    throw std::runtime_error(
                            "generation child returned an "
                            "invalid completion response");
                }

                candidate.content =
                        response["content"]
                                .get<std::string>();

                if (response.contains(
                            "tokens_predicted") &&
                    response["tokens_predicted"]
                            .is_number_integer()) {
                    const long long value =
                            response["tokens_predicted"]
                                    .get<long long>();

                    if (value > 0) {
                        candidate.generated_tokens =
                                static_cast<std::size_t>(
                                        value);
                    }
                } else if (
                        response.contains("timings") &&
                        response["timings"].is_object() &&
                        response["timings"].contains(
                                "predicted_n") &&
                        response["timings"]["predicted_n"]
                                .is_number_integer()) {
                    const long long value =
                            response["timings"]
                                    ["predicted_n"]
                                    .get<long long>();

                    if (value > 0) {
                        candidate.generated_tokens =
                                static_cast<std::size_t>(
                                        value);
                    }
                }

                if (response.contains("stop_type") &&
                    response["stop_type"].is_string()) {
                    candidate.stop_reason =
                            response["stop_type"]
                                    .get<std::string>();
                } else if (
                        response.contains("stopped_eos") &&
                        response["stopped_eos"]
                                .is_boolean() &&
                        response["stopped_eos"]
                                .get<bool>()) {
                    candidate.stop_reason = "eos";
                } else if (
                        response.contains("stopped_limit") &&
                        response["stopped_limit"]
                                .is_boolean() &&
                        response["stopped_limit"]
                                .get<bool>()) {
                    candidate.stop_reason = "limit";
                }

                if (candidate.content.empty()) {
                    throw std::runtime_error(
                            "generation candidate returned "
                            "empty content");
                }

                candidate.success = true;
            } catch (const std::exception & error) {
                // A failed candidate does not immediately fail the
                // entire DAG. CandidateMerge succeeds when at least
                // one independent candidate completed successfully.
                candidate.error = error.what();
                candidate.success = false;
            }
        } else if (type_ == RagTaskType::CandidateMerge) {
            std::size_t selected =
                    RAG_INVALID_INDEX;

            std::size_t best_generated_tokens = 0;
            std::size_t best_content_size = 0;

            for (std::size_t index = 0;
                 index <
                         runtime->generation_candidates.size();
                 ++index) {
                const auto & candidate =
                        runtime->generation_candidates[index];

                if (!candidate.success ||
                    candidate.content.empty()) {
                    continue;
                }

                const bool better =
                        selected == RAG_INVALID_INDEX ||
                        candidate.generated_tokens >
                                best_generated_tokens ||
                        (
                            candidate.generated_tokens ==
                                    best_generated_tokens &&
                            candidate.content.size() >
                                    best_content_size
                        );

                if (!better) {
                    continue;
                }

                selected = index;
                best_generated_tokens =
                        candidate.generated_tokens;
                best_content_size =
                        candidate.content.size();
            }

            if (selected == RAG_INVALID_INDEX) {
                throw std::runtime_error(
                        "all generation candidates failed");
            }

            const auto & candidate =
                    runtime->generation_candidates.at(
                            selected);

            runtime->selected_generation_candidate_index =
                    selected;

            runtime->generation_prompt =
                    candidate.prompt;

            runtime->generation_result =
                    candidate.content;
        } else if (type_ == RagTaskType::Finalize) {
            runtime->final_answer = runtime->generation_result;

            if (runtime->final_answer.empty()) {
                throw std::runtime_error(
                        "Finalize has no CPU Decode answer");
            }
        }
        result.success = true;
    } catch (const std::exception & e) {
        {
            std::lock_guard<std::mutex> lock(runtime->mutex);
            if (runtime->error.empty()) {
                runtime->error = e.what();
            }
        }
        result.error_message = e.what();
    }
    return result;
}
