#include "server-rag.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

std::string rag_trim(std::string value) {
    const auto is_space = [](unsigned char ch) {
        return std::isspace(ch) != 0;
    };

    while (!value.empty() &&
           is_space(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }

    while (!value.empty() &&
           is_space(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }

    return value;
}

std::vector<std::string> rag_split_document(
        const std::string & document) {
    std::vector<std::string> chunks;
    std::string current;
    current.reserve(document.size());

    for (const char ch : document) {
        current.push_back(ch);

        if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') {
            std::string trimmed = rag_trim(current);

            if (!trimmed.empty()) {
                chunks.push_back(std::move(trimmed));
            }

            current.clear();
        }
    }

    std::string tail = rag_trim(current);

    if (!tail.empty()) {
        chunks.push_back(std::move(tail));
    }

    return chunks;
}

std::vector<std::size_t> rag_search_inner_product(
        const std::vector<std::vector<float>> & doc_embeddings,
        const std::vector<std::size_t> & doc_embedding_source_indices,
        const std::vector<float> & query_embedding,
        std::size_t top_k) {
    if (doc_embeddings.empty()) {
        throw std::runtime_error("document embeddings are empty");
    }

    if (query_embedding.empty()) {
        throw std::runtime_error("query embedding is empty");
    }

    if (doc_embeddings.size() != doc_embedding_source_indices.size()) {
        throw std::runtime_error(
                "doc embedding index mapping size mismatch");
    }

    const std::size_t dimension = query_embedding.size();

    struct scored_embedding {
        std::size_t dense_index;
        float score;
    };

    std::vector<scored_embedding> scored;
    scored.reserve(doc_embeddings.size());

    for (std::size_t dense_index = 0;
         dense_index < doc_embeddings.size();
         ++dense_index) {
        const auto & embedding = doc_embeddings[dense_index];

        if (embedding.size() != dimension) {
            throw std::runtime_error("embedding dimension mismatch");
        }

        float score = 0.0F;

        for (std::size_t dim = 0; dim < dimension; ++dim) {
            score += embedding[dim] * query_embedding[dim];
        }

        scored.push_back({
            dense_index,
            score,
        });
    }

    std::sort(
            scored.begin(),
            scored.end(),
            [](const scored_embedding & lhs,
               const scored_embedding & rhs) {
                if (lhs.score != rhs.score) {
                    return lhs.score > rhs.score;
                }

                return lhs.dense_index < rhs.dense_index;
            });

    const std::size_t keep_n = std::min(top_k, scored.size());

    std::vector<std::size_t> top_indices;
    top_indices.reserve(keep_n);

    for (std::size_t rank = 0; rank < keep_n; ++rank) {
        const std::size_t dense_index = scored[rank].dense_index;
        top_indices.push_back(
                doc_embedding_source_indices[dense_index]);
    }

    return top_indices;
}

std::vector<std::size_t> rag_merge_subquery_hits(
        const std::vector<std::vector<std::size_t>> & per_query_hits,
        std::size_t top_k) {
    std::unordered_map<std::size_t, float> score_by_index;

    for (const auto & hits : per_query_hits) {
        for (std::size_t rank = 0; rank < hits.size(); ++rank) {
            const std::size_t index = hits[rank];

            score_by_index[index] +=
                    1.0F / (1.0F + static_cast<float>(rank));
        }
    }

    std::vector<std::pair<std::size_t, float>> scored_hits;
    scored_hits.reserve(score_by_index.size());

    for (const auto & [index, score] : score_by_index) {
        scored_hits.emplace_back(index, score);
    }

    std::sort(
            scored_hits.begin(),
            scored_hits.end(),
            [](const auto & lhs, const auto & rhs) {
                if (lhs.second != rhs.second) {
                    return lhs.second > rhs.second;
                }

                return lhs.first < rhs.first;
            });

    const std::size_t keep_n =
            std::min(top_k, scored_hits.size());

    std::vector<std::size_t> merged_indices;
    merged_indices.reserve(keep_n);

    for (std::size_t index = 0; index < keep_n; ++index) {
        merged_indices.push_back(scored_hits[index].first);
    }

    return merged_indices;
}

std::string build_generation_prompt(
        const std::string & query,
        const std::vector<std::string> & context_chunks) {
    std::ostringstream oss;

    oss << "You are a concise and faithful QA assistant. "
           "Use only the provided context.\n";
    oss << "Context:\n";

    for (const auto & chunk : context_chunks) {
        oss << "- " << chunk << "\n";
    }

    oss << "Question: " << query << "\n";
    oss << "Answer:";

    return oss.str();
}

std::vector<std::string> build_generation_segments(
        const std::string & query,
        const std::vector<std::string> & sub_queries,
        const std::vector<std::string> & context_chunks) {
    std::vector<std::string> segments;
    segments.reserve(3);

    std::ostringstream segment_1;
    segment_1 << "You are a concise and faithful QA assistant. "
                 "Use only the provided context.\n";
    segment_1 << "Question: " << query << "\n";
    segments.push_back(segment_1.str());

    for (const auto & sub_query : sub_queries) {
        std::ostringstream segment_2_item;
        segment_2_item << "Sub-query hint:\n";
        segment_2_item << "- " << sub_query << "\n";
        segments.push_back(segment_2_item.str());
    }

    std::ostringstream segment_3;
    segment_3 << "Context:\n";

    for (const auto & chunk : context_chunks) {
        segment_3 << "- " << chunk << "\n";
    }

    segment_3 << "Answer:";
    segments.push_back(segment_3.str());

    return segments;
}