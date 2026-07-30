#include "server-rag.h"

#include <cctype>
#include <sstream>
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