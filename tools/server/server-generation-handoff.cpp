#include "server-generation-handoff.h"

#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr std::uint64_t FNV1A_OFFSET_BASIS =
        14695981039346656037ULL;
constexpr std::uint64_t FNV1A_PRIME =
        1099511628211ULL;

} // namespace

std::uint64_t server_generation_state_checksum(
        const std::uint8_t * data,
        std::size_t size) {
    if (size > 0 && data == nullptr) {
        throw std::invalid_argument(
                "cannot checksum a null byte buffer");
    }

    std::uint64_t value = FNV1A_OFFSET_BASIS;

    for (std::size_t i = 0; i < size; ++i) {
        value ^= static_cast<std::uint64_t>(data[i]);
        value *= FNV1A_PRIME;
    }

    return value;
}

std::uint64_t server_generation_state_checksum(
        const std::vector<std::uint8_t> & data) {
    return server_generation_state_checksum(
            data.data(),
            data.size());
}

std::uint64_t server_generation_logits_checksum(
        const float * logits,
        std::size_t count) {
    if (count > 0 && logits == nullptr) {
        throw std::invalid_argument(
                "cannot checksum a null logits buffer");
    }

    return server_generation_state_checksum(
            reinterpret_cast<const std::uint8_t *>(logits),
            count * sizeof(float));
}

std::uint64_t server_generation_logits_checksum(
        const std::vector<float> & logits) {
    return server_generation_logits_checksum(
            logits.data(),
            logits.size());
}

std::string server_generation_model_identity(
        const llama_context * ctx) {
    if (ctx == nullptr) {
        throw std::invalid_argument(
                "cannot build model identity from a null context");
    }

    const llama_model * model = llama_get_model(ctx);
    if (model == nullptr) {
        throw std::runtime_error(
                "generation context does not contain a model");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
        throw std::runtime_error(
                "generation model does not contain a vocabulary");
    }

    std::array<char, 512> description = {};

    const int description_size = llama_model_desc(
            model,
            description.data(),
            description.size());

    if (description_size < 0) {
        throw std::runtime_error(
                "failed to query generation model description");
    }

    std::ostringstream out;
    out << "desc=" << description.data()
        << ";size=" << llama_model_size(model)
        << ";params=" << llama_model_n_params(model)
        << ";ftype="
        << static_cast<int>(llama_model_ftype(model))
        << ";vocab=" << llama_vocab_n_tokens(vocab);

    return out.str();
}

std::string server_generation_runtime_identity(
        const llama_context * ctx) {
    if (ctx == nullptr) {
        throw std::invalid_argument(
                "cannot build runtime identity from a null context");
    }

    std::ostringstream out;
    out << "n_ctx=" << llama_n_ctx(ctx)
        << ";n_ctx_seq=" << llama_n_ctx_seq(ctx)
        << ";n_batch=" << llama_n_batch(ctx)
        << ";n_ubatch=" << llama_n_ubatch(ctx)
        << ";n_seq_max=" << llama_n_seq_max(ctx);

    return out.str();
}

void server_generation_handoff_validate(
        const server_generation_handoff & handoff) {
    if (handoff.format_version !=
            server_generation_handoff::FORMAT_VERSION) {
        throw std::invalid_argument(
                "unsupported generation handoff format version");
    }

    if (handoff.model_identity.empty()) {
        throw std::invalid_argument(
                "generation handoff is missing model identity");
    }

    if (handoff.runtime_identity.empty()) {
        throw std::invalid_argument(
                "generation handoff is missing producer runtime identity");
    }

    if (handoff.producer_backend.empty()) {
        throw std::invalid_argument(
                "generation handoff is missing producer backend");
    }

    if (handoff.source_seq_id < 0) {
        throw std::invalid_argument(
                "generation handoff has an invalid source sequence ID");
    }

    if (handoff.state_flags != LLAMA_STATE_SEQ_FLAGS_NONE) {
        throw std::invalid_argument(
                "generation handoff must use host sequence-state format");
    }

    if (handoff.prompt_tokens.empty()) {
        throw std::invalid_argument(
                "generation handoff has no Prompt tokens");
    }

    if (handoff.next_position < 0) {
        throw std::invalid_argument(
                "generation handoff has an invalid next position");
    }

    if (handoff.sequence_state.empty()) {
        throw std::invalid_argument(
                "generation handoff has no sequence state");
    }

    if (handoff.final_logits.empty()) {
        throw std::invalid_argument(
                "generation handoff has no final Prefill logits");
    }

    if (server_generation_state_checksum(handoff.sequence_state) !=
            handoff.sequence_state_checksum) {
        throw std::invalid_argument(
                "generation handoff sequence-state checksum mismatch");
    }

    if (server_generation_logits_checksum(handoff.final_logits) !=
            handoff.final_logits_checksum) {
        throw std::invalid_argument(
                "generation handoff final-logits checksum mismatch");
    }
}

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
        const common_params_sampling & sampling_params) {
    if (ctx == nullptr) {
        throw std::invalid_argument(
                "cannot export generation handoff from a null context");
    }

    if (source_seq_id < 0) {
        throw std::invalid_argument(
                "cannot export generation handoff from an invalid sequence");
    }

    if (final_logits == nullptr || n_logits == 0) {
        throw std::invalid_argument(
                "cannot export generation handoff without final logits");
    }

    const llama_state_seq_flags flags =
            LLAMA_STATE_SEQ_FLAGS_NONE;

    const std::size_t state_size =
            llama_state_seq_get_size_ext(
                    ctx,
                    source_seq_id,
                    flags);

    if (state_size == 0) {
        throw std::runtime_error(
                "failed to determine generation sequence-state size");
    }

    std::vector<std::uint8_t> state(state_size);

    const std::size_t written =
            llama_state_seq_get_data_ext(
                    ctx,
                    state.data(),
                    state.size(),
                    source_seq_id,
                    flags);

    if (written == 0) {
        throw std::runtime_error(
                "failed to export generation sequence state");
    }

    if (written != state.size()) {
        throw std::runtime_error(
                "generation sequence-state export size mismatch");
    }

    server_generation_handoff handoff;
    handoff.request_id = request_id;
    handoff.model_identity = model_identity;
    handoff.runtime_identity = runtime_identity;
    handoff.producer_backend = producer_backend;
    handoff.source_seq_id = source_seq_id;
    handoff.state_flags = flags;
    handoff.prompt_tokens = prompt_tokens;
    handoff.next_position = next_position;
    handoff.sampling_params = sampling_params;

    handoff.sequence_state = std::move(state);
    handoff.sequence_state_checksum =
            server_generation_state_checksum(
                    handoff.sequence_state);

    handoff.final_logits.assign(
            final_logits,
            final_logits + n_logits);

    handoff.final_logits_checksum =
            server_generation_logits_checksum(
                    handoff.final_logits);

    server_generation_handoff_validate(handoff);
    return handoff;
}

std::size_t server_generation_handoff_restore(
        llama_context * ctx,
        llama_seq_id dest_seq_id,
        const server_generation_handoff & handoff) {
    if (ctx == nullptr) {
        throw std::invalid_argument(
                "cannot restore generation handoff into a null context");
    }

    if (dest_seq_id < 0) {
        throw std::invalid_argument(
                "cannot restore generation handoff into an invalid sequence");
    }

    server_generation_handoff_validate(handoff);

    if (server_generation_model_identity(ctx) !=
            handoff.model_identity) {
        throw std::invalid_argument(
                "generation handoff model identity mismatch");
    }

    if (handoff.next_position >=
            static_cast<llama_pos>(llama_n_ctx_seq(ctx))) {
        throw std::invalid_argument(
                "generation handoff exceeds destination context capacity");
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (vocab == nullptr) {
        throw std::runtime_error(
                "destination context has no vocabulary");
    }

    if (handoff.final_logits.size() !=
            static_cast<std::size_t>(
                    llama_vocab_n_tokens(vocab))) {
        throw std::invalid_argument(
                "generation handoff vocabulary size mismatch");
    }

    const std::size_t restored =
            llama_state_seq_set_data_ext(
                    ctx,
                    handoff.sequence_state.data(),
                    handoff.sequence_state.size(),
                    dest_seq_id,
                    handoff.state_flags);

    if (restored == 0) {
        throw std::runtime_error(
                "failed to restore generation sequence state");
    }

    if (restored != handoff.sequence_state.size()) {
        throw std::runtime_error(
                "generation sequence-state restore size mismatch");
    }

    return restored;
}

void server_generation_handoff_prepare_sampler(
        common_sampler * sampler,
        const server_generation_handoff & handoff) {
    if (sampler == nullptr) {
        throw std::invalid_argument(
                "cannot prepare a null generation sampler");
    }

    server_generation_handoff_validate(handoff);

    common_sampler_reset(sampler);

    for (const llama_token token : handoff.prompt_tokens) {
        if (token != LLAMA_TOKEN_NULL) {
            common_sampler_accept(sampler, token, false);
        }
    }
}

llama_token server_generation_handoff_sample_first_token(
        common_sampler * sampler,
        const server_generation_handoff & handoff,
        bool grammar_first) {
    if (sampler == nullptr) {
        throw std::invalid_argument(
                "cannot sample with a null generation sampler");
    }

    server_generation_handoff_validate(handoff);

    if (handoff.final_logits.size() >
            static_cast<std::size_t>(
                    std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error(
                "generation handoff logits count exceeds int32 range");
    }

    return common_sampler_sample_from_logits(
            sampler,
            handoff.final_logits.data(),
            static_cast<int32_t>(
                    handoff.final_logits.size()),
            grammar_first);
}
