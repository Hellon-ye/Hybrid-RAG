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


namespace {

std::string rag_replace_fullwidth_semicolon(std::string value) {
    static const std::string fullwidth_semicolon = "；";

    std::size_t position = 0;

    while ((position = value.find(
                    fullwidth_semicolon,
                    position)) != std::string::npos) {
        value.replace(
                position,
                fullwidth_semicolon.size(),
                ";");

        ++position;
    }

    return value;
}

std::string rag_strip_subquery_marker(std::string token) {
    token = rag_trim(std::move(token));

    // Remove common bullet markers.
    while (!token.empty() &&
           (token.front() == '-' ||
            token.front() == '*' ||
            token.front() == ';')) {
        token.erase(token.begin());
        token = rag_trim(std::move(token));
    }

    // Remove numbered-list markers such as:
    // "1. query", "2) query", "3、query", "4）query".
    std::size_t digit_end = 0;

    while (digit_end < token.size() &&
           std::isdigit(
                   static_cast<unsigned char>(
                           token[digit_end])) != 0) {
        ++digit_end;
    }

    if (digit_end > 0) {
        std::size_t marker_end = digit_end;
        bool has_number_marker = false;

        if (marker_end < token.size() &&
            (token[marker_end] == '.' ||
             token[marker_end] == ')' ||
             token[marker_end] == ':')) {
            ++marker_end;
            has_number_marker = true;
        } else {
            static const std::string chinese_separator = "、";
            static const std::string fullwidth_right_parenthesis = "）";
            static const std::string fullwidth_period = "．";

            if (token.compare(
                        marker_end,
                        chinese_separator.size(),
                        chinese_separator) == 0) {
                marker_end += chinese_separator.size();
                has_number_marker = true;
            } else if (token.compare(
                               marker_end,
                               fullwidth_right_parenthesis.size(),
                               fullwidth_right_parenthesis) == 0) {
                marker_end += fullwidth_right_parenthesis.size();
                has_number_marker = true;
            } else if (token.compare(
                               marker_end,
                               fullwidth_period.size(),
                               fullwidth_period) == 0) {
                marker_end += fullwidth_period.size();
                has_number_marker = true;
            }
        }

        if (has_number_marker) {
            token.erase(0, marker_end);
            token = rag_trim(std::move(token));
        }
    }

    return token;
}

} // namespace

std::vector<std::string> rag_split_sub_queries(
        const std::string & expanded,
        std::size_t max_subqueries,
        std::size_t max_subquery_chars) {
    std::vector<std::string> sub_queries;

    if (max_subqueries == 0 || max_subquery_chars == 0) {
        return sub_queries;
    }

    const std::string normalized =
            rag_replace_fullwidth_semicolon(expanded);

    std::string current;

    const auto push_token = [&](std::string token) {
        token = rag_strip_subquery_marker(std::move(token));

        if (token.empty()) {
            return;
        }

        if (token.size() > max_subquery_chars) {
            token.resize(max_subquery_chars);
            token = rag_trim(std::move(token));
        }

        if (token.empty()) {
            return;
        }

        if (std::find(
                    sub_queries.begin(),
                    sub_queries.end(),
                    token) != sub_queries.end()) {
            return;
        }

        sub_queries.push_back(std::move(token));
    };

    for (const char ch : normalized) {
        // Accept both the original semicolon format and multiline lists.
        if (ch == ';' || ch == '\n' || ch == '\r') {
            push_token(std::move(current));
            current.clear();

            if (sub_queries.size() >= max_subqueries) {
                break;
            }

            continue;
        }

        current.push_back(ch);
    }

    if (sub_queries.size() < max_subqueries) {
        push_token(std::move(current));
    }

    return sub_queries;
}

std::string build_query_expansion_prompt(
        const std::string & query) {
    return
            "You are a query rewriter for a retrieval system. "
            "Given the user's query, generate three different sub-queries. "
            "Each sub-query should focus on a distinct aspect or phrasing "
            "of the original query. "
            "Return the three sub-queries on a single line, separated by "
            "semicolon(;). "
            "Do NOT use numbers. "
            "Do NOT answer the query. "
            "Query: " + query + "\n"
            "Your output:";
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