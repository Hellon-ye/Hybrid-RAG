#pragma once

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Backend-independent boundary between Generation Prefill and Decode.
//
// The producer evaluates the complete Prompt and exports:
//   - canonical host sequence state;
//   - final-position logits;
//   - Prompt tokens and sampling configuration.
//
// No generated token is sampled by the producer. The Decode consumer owns
// the sampler and RNG from the first generated token.
struct server_generation_handoff {
    static constexpr std::uint32_t FORMAT_VERSION = 2;

    std::uint32_t format_version = FORMAT_VERSION;
    std::uint64_t request_id = 0;

    std::string model_identity;
    std::string runtime_identity;
    std::string producer_backend;

    llama_seq_id source_seq_id = -1;
    llama_state_seq_flags state_flags = LLAMA_STATE_SEQ_FLAGS_NONE;

    std::vector<llama_token> prompt_tokens;
    llama_pos next_position = -1;

    common_params_sampling sampling_params;

    std::vector<std::uint8_t> sequence_state;
    std::uint64_t sequence_state_checksum = 0;

    std::vector<float> final_logits;
    std::uint64_t final_logits_checksum = 0;
};

std::uint64_t server_generation_state_checksum(
        const std::uint8_t * data,
        std::size_t size);

std::uint64_t server_generation_state_checksum(
        const std::vector<std::uint8_t> & data);

std::uint64_t server_generation_logits_checksum(
        const float * logits,
        std::size_t count);

std::uint64_t server_generation_logits_checksum(
        const std::vector<float> & logits);

std::string server_generation_model_identity(
        const llama_context * ctx);

std::string server_generation_runtime_identity(
        const llama_context * ctx);

void server_generation_handoff_validate(
        const server_generation_handoff & handoff);

server_generation_handoff server_generation_handoff_export(
        llama_context * ctx,
        llama_seq_id source_seq_id,
        std::uint64_t request_id,
        const std::string & model_identity,
        const std::string & runtime_identity,
        const std::string & producer_backend,
        const std::vector<llama_token> & prompt_tokens,
        const float * final_logits,
        std::size_t n_logits,
        llama_pos next_position,
        const common_params_sampling & sampling_params);

std::size_t server_generation_handoff_restore(
        llama_context * ctx,
        llama_seq_id dest_seq_id,
        const server_generation_handoff & handoff);

void server_generation_handoff_prepare_sampler(
        common_sampler * sampler,
        const server_generation_handoff & handoff);

llama_token server_generation_handoff_sample_first_token(
        common_sampler * sampler,
        const server_generation_handoff & handoff,
        bool grammar_first = false);
