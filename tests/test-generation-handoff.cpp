#include "arg.h"
#include "common.h"
#include "log.h"
#include "server-generation-handoff.h"
#include "server-generation-executor.h"
#include "server-schema.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

using common_sampler_unique_ptr =
        std::unique_ptr<common_sampler, decltype(&common_sampler_free)>;

bool decode_prompt(
        llama_context * ctx,
        const llama_tokens & tokens) {
    if (ctx == nullptr || tokens.empty()) {
        LOG_ERR("%s: invalid context or empty Prompt\n", __func__);
        return false;
    }

    // llama_batch_get_one() currently accepts a mutable token pointer.
    // Keep a local copy alive until llama_decode() completes.
    llama_tokens mutable_tokens = tokens;

    llama_batch batch = llama_batch_get_one(
            mutable_tokens.data(),
            static_cast<int32_t>(mutable_tokens.size()));

    const int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        LOG_ERR(
                "%s: Prompt decode failed, ret=%d\n",
                __func__,
                ret);
        return false;
    }

    llama_synchronize(ctx);
    return true;
}

bool decode_generated_token(
        llama_context * ctx,
        llama_token token,
        llama_pos position,
        llama_seq_id sequence_id) {
    if (ctx == nullptr) {
        LOG_ERR("%s: null context\n", __func__);
        return false;
    }

    llama_batch batch = llama_batch_init(1, 0, 1);

    common_batch_add(
            batch,
            token,
            position,
            { sequence_id },
            true);

    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        LOG_ERR(
                "%s: generated-token decode failed, "
                "token=%d, pos=%d, seq=%d, ret=%d\n",
                __func__,
                token,
                position,
                sequence_id,
                ret);
        return false;
    }

    llama_synchronize(ctx);
    return true;
}

template<typename Function>
bool expect_exception(
        const char * test_name,
        Function && function) {
    try {
        function();
    } catch (const std::exception & error) {
        LOG_INF(
                "PASS: %s rejected: %s\n",
                test_name,
                error.what());
        return true;
    }

    LOG_ERR(
            "FAIL: %s was not rejected\n",
            test_name);
    return false;
}

bool run_integrity_tests(
        llama_context * destination_ctx,
        const server_generation_handoff & handoff) {
    {
        auto corrupted = handoff;
        corrupted.sequence_state[0] ^= 0x01U;

        if (!expect_exception(
                    "corrupted sequence-state checksum",
                    [&]() {
                        server_generation_handoff_validate(corrupted);
                    })) {
            return false;
        }
    }

    {
        auto corrupted = handoff;

        auto * bytes = reinterpret_cast<std::uint8_t *>(
                corrupted.final_logits.data());
        bytes[0] ^= 0x01U;

        if (!expect_exception(
                    "corrupted final-logits checksum",
                    [&]() {
                        server_generation_handoff_validate(corrupted);
                    })) {
            return false;
        }
    }

    {
        auto incompatible = handoff;
        incompatible.model_identity += ";tampered=true";

        if (!expect_exception(
                    "incompatible model identity",
                    [&]() {
                        server_generation_handoff_restore(
                                destination_ctx,
                                0,
                                incompatible);
                    })) {
            return false;
        }
    }

    {
        auto incompatible = handoff;
        incompatible.final_logits.pop_back();
        incompatible.final_logits_checksum =
                server_generation_logits_checksum(
                        incompatible.final_logits);

        if (!expect_exception(
                    "incompatible vocabulary size",
                    [&]() {
                        server_generation_handoff_restore(
                                destination_ctx,
                                0,
                                incompatible);
                    })) {
            return false;
        }
    }

    return true;
}

bool run_generation_schema_tests(
        llama_model * model,
        const common_params & params_base) {
    const llama_vocab * vocab =
            llama_model_get_vocab(model);

    if (vocab == nullptr) {
        LOG_ERR("%s: model has no vocabulary\n", __func__);
        return false;
    }

    const std::vector<llama_logit_bias> logit_bias_eog;

    auto parse_request =
            [&](const nlohmann::ordered_json & request) {
                return server_schema::eval_llama_cmpl_schema(
                        vocab,
                        params_base,
                        256,
                        logit_bias_eog,
                        request);
            };

    auto check_serialized_route =
            [&](const nlohmann::ordered_json & serialized,
                    const char * mode) {
                if (!serialized.contains("generation_handoff") ||
                        !serialized.contains(
                                "generation_prefill_backend") ||
                        !serialized.contains(
                                "generation_decode_backend")) {
                    LOG_ERR(
                            "%s: Generation route missing from %s JSON\n",
                            __func__,
                            mode);
                    return false;
                }

                if (!serialized.at(
                            "generation_handoff").get<bool>() ||
                        serialized.at(
                            "generation_prefill_backend")
                                .get<std::string>() != "cpu" ||
                        serialized.at(
                            "generation_decode_backend")
                                .get<std::string>() != "cpu") {
                    LOG_ERR(
                            "%s: incorrect Generation route in %s JSON\n",
                            __func__,
                            mode);
                    return false;
                }

                return true;
            };

    {
        const nlohmann::ordered_json request = {
            { "generation_handoff", true },
            { "generation_prefill_backend", "cpu" },
            { "generation_decode_backend", "cpu" },
            { "backend_sampling", false },
        };

        const task_params parsed =
                parse_request(request);

        if (!parsed.generation_handoff ||
                parsed.generation_prefill_backend != "cpu" ||
                parsed.generation_decode_backend != "cpu") {
            LOG_ERR(
                    "%s: CPU Generation route parsed incorrectly\n",
                    __func__);
            return false;
        }

        if (!check_serialized_route(
                    parsed.to_json(false),
                    "full") ||
                !check_serialized_route(
                    parsed.to_json(true),
                    "metrics")) {
            return false;
        }

        LOG_INF(
                "PASS: CPU Prefill -> CPU Decode schema route\n");
    }

    {
        const nlohmann::ordered_json request = {
            { "generation_handoff", true },
            { "generation_prefill_backend", "auto" },
            { "generation_decode_backend", "auto" },
            { "backend_sampling", false },
        };

        const task_params parsed =
                parse_request(request);

        if (parsed.generation_prefill_backend != "cpu" ||
                parsed.generation_decode_backend != "cpu") {
            LOG_ERR(
                    "%s: AUTO route did not resolve to CPU\n",
                    __func__);
            return false;
        }

        LOG_INF(
                "PASS: AUTO route resolved to registered CPU backend\n");
    }

    if (!expect_exception(
                "unknown Generation backend",
                [&]() {
                    const nlohmann::ordered_json request = {
                        { "generation_handoff", true },
                        { "generation_prefill_backend", "cuda" },
                        { "generation_decode_backend", "cpu" },
                    };

                    (void) parse_request(request);
                })) {
        return false;
    }

    {
        const nlohmann::ordered_json request = {
            { "generation_handoff", true },
            { "generation_prefill_backend", "npu" },
            { "generation_decode_backend", "cpu" },
            { "backend_sampling", false },
        };

        const task_params parsed =
                parse_request(request);

        if (!parsed.generation_handoff ||
                parsed.generation_prefill_backend != "npu" ||
                parsed.generation_decode_backend != "cpu") {
            LOG_ERR(
                    "%s: NPU Prefill -> CPU Decode route parsed "
                    "incorrectly\n",
                    __func__);
            return false;
        }

        LOG_INF(
                "PASS: NPU Prefill -> CPU Decode schema route\n");
    }

    if (!expect_exception(
                "unregistered NPU Decode backend",
                [&]() {
                    const nlohmann::ordered_json request = {
                        { "generation_handoff", true },
                        { "generation_prefill_backend", "cpu" },
                        { "generation_decode_backend", "npu" },
                    };

                    (void) parse_request(request);
                })) {
        return false;
    }

    if (!expect_exception(
                "backend sampling with Generation handoff",
                [&]() {
                    const nlohmann::ordered_json request = {
                        { "generation_handoff", true },
                        { "generation_prefill_backend", "cpu" },
                        { "generation_decode_backend", "cpu" },
                        { "backend_sampling", true },
                    };

                    (void) parse_request(request);
                })) {
        return false;
    }

    if (!expect_exception(
                "token probabilities with Generation handoff",
                [&]() {
                    const nlohmann::ordered_json request = {
                        { "generation_handoff", true },
                        { "generation_prefill_backend", "cpu" },
                        { "generation_decode_backend", "cpu" },
                        { "backend_sampling", false },
                        { "n_probs", 5 },
                    };

                    (void) parse_request(request);
                })) {
        return false;
    }

    return true;
}

bool run_reference_handoff_test(
        llama_model * model,
        const common_params & params,
        const llama_tokens & prompt_tokens) {
    auto context_params =
            common_context_params_to_llama(params);

    context_params.n_seq_max = 1;

    llama_context_ptr baseline_ctx {
        llama_init_from_model(model, context_params)
    };
    llama_context_ptr producer_ctx {
        llama_init_from_model(model, context_params)
    };

    if (!baseline_ctx || !producer_ctx) {
        LOG_ERR("%s: failed to create baseline or producer Context\n", __func__);
        return false;
    }

    std::unique_ptr<server_generation_decode_executor>
            consumer_executor;

    try {
        consumer_executor =
                std::make_unique<
                        server_generation_decode_executor>(
                            model,
                            context_params,
                            0);
    } catch (const std::exception & error) {
        LOG_ERR(
                "%s: failed to create Decode executor: %s\n",
                __func__,
                error.what());
        return false;
    }

    LOG_INF(
            "Context A: uninterrupted CPU baseline\n"
            "Context B: CPU Prefill producer\n"
            "Context C: CPU Decode consumer\n");

    if (!decode_prompt(baseline_ctx.get(), prompt_tokens) ||
            !decode_prompt(producer_ctx.get(), prompt_tokens)) {
        return false;
    }

    const llama_vocab * vocab =
            llama_model_get_vocab(model);

    if (vocab == nullptr) {
        LOG_ERR("%s: model has no vocabulary\n", __func__);
        return false;
    }

    const int32_t n_vocab =
            llama_vocab_n_tokens(vocab);

    const int32_t final_output_index =
            static_cast<int32_t>(prompt_tokens.size()) - 1;

    const float * final_logits =
            llama_get_logits_ith(
                    producer_ctx.get(),
                    final_output_index);

    if (final_logits == nullptr) {
        LOG_ERR("%s: failed to obtain producer final logits\n", __func__);
        return false;
    }

    const llama_pos next_position =
            static_cast<llama_pos>(prompt_tokens.size());

    server_generation_handoff handoff =
            server_generation_handoff_export(
                    producer_ctx.get(),
                    0,
                    1,
                    server_generation_model_identity(
                            producer_ctx.get()),
                    server_generation_runtime_identity(
                            producer_ctx.get()),
                    "cpu-reference",
                    prompt_tokens,
                    final_logits,
                    static_cast<std::size_t>(n_vocab),
                    next_position,
                    params.sampling);

    LOG_INF(
            "Handoff exported: state=%zu bytes, "
            "logits=%zu, prompt=%zu tokens, next_pos=%d\n",
            handoff.sequence_state.size(),
            handoff.final_logits.size(),
            handoff.prompt_tokens.size(),
            handoff.next_position);

    if (!run_integrity_tests(
                consumer_executor->context(),
                handoff)) {
        return false;
    }

    common_params_sampling baseline_sampling =
            params.sampling;

    common_sampler_unique_ptr baseline_sampler(
            common_sampler_init(
                    model,
                    baseline_sampling),
            &common_sampler_free);

    if (!baseline_sampler) {
        LOG_ERR("%s: failed to initialize baseline sampler\n", __func__);
        return false;
    }

    // The baseline and executor both begin with the same Prompt-only
    // sampler history, but the executor owns its sampler internally.
    server_generation_handoff_prepare_sampler(
            baseline_sampler.get(),
            handoff);

    llama_token baseline_token =
            common_sampler_sample(
                    baseline_sampler.get(),
                    baseline_ctx.get(),
                    final_output_index);

    llama_token consumer_token;

    try {
        consumer_token =
                consumer_executor->begin(handoff);
    } catch (const std::exception & error) {
        LOG_ERR(
                "%s: Decode executor begin failed: %s\n",
                __func__,
                error.what());
        return false;
    }

    common_sampler_accept(
            baseline_sampler.get(),
            baseline_token,
            true);

    if (baseline_token != consumer_token) {
        LOG_ERR(
                "first-token mismatch: baseline=%d handoff=%d\n",
                baseline_token,
                consumer_token);
        return false;
    }

    llama_tokens baseline_output { baseline_token };
    llama_tokens consumer_output { consumer_token };

    LOG_INF(
            "token[0] matches: %d\n",
            baseline_token);

    const int32_t n_predict =
            std::max<int32_t>(2, params.n_predict);

    llama_pos position = next_position;

    for (int32_t step = 1; step < n_predict; ++step) {
        if (llama_vocab_is_eog(vocab, baseline_token)) {
            LOG_INF(
                    "both paths reached EOG after %d token(s)\n",
                    step);
            break;
        }

        if (!decode_generated_token(
                    baseline_ctx.get(),
                    baseline_token,
                    position,
                    0)) {
            return false;
        }

        baseline_token =
                common_sampler_sample(
                        baseline_sampler.get(),
                        baseline_ctx.get(),
                        0);

        common_sampler_accept(
                baseline_sampler.get(),
                baseline_token,
                true);

        try {
            consumer_token =
                    consumer_executor->decode_next();
        } catch (const std::exception & error) {
            LOG_ERR(
                    "%s: Decode executor step failed: %s\n",
                    __func__,
                    error.what());
            return false;
        }

        baseline_output.push_back(baseline_token);
        consumer_output.push_back(consumer_token);

        if (baseline_token != consumer_token) {
            LOG_ERR(
                    "token mismatch at step %d: "
                    "baseline=%d handoff=%d\n",
                    step,
                    baseline_token,
                    consumer_token);
            return false;
        }

        LOG_INF(
                "token[%d] matches: %d\n",
                step,
                baseline_token);

        ++position;
    }

    if (baseline_output != consumer_output) {
        LOG_ERR("%s: final output vectors differ\n", __func__);
        return false;
    }

    LOG_INF(
            "PASS: uninterrupted CPU generation and "
            "CPU Prefill->CPU Decode Handoff produced "
            "%zu identical token(s)\n",
            baseline_output.size());

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt =
            "A deterministic generation handoff test:";
    params.n_batch = 128;
    params.n_predict = 8;
    params.n_ctx = 256;
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(
                argc,
                argv,
                params,
                LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_predict < 2) {
        params.n_predict = 8;
    }

    ggml_backend_load_all();

    auto llama_init =
            common_init_from_params(params, true);

    if (!llama_init || llama_init->model() == nullptr) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    llama_model * model = llama_init->model();

    GGML_ASSERT(
            llama_init->context() == nullptr);

    if (!run_generation_schema_tests(
                model,
                params)) {
        return 1;
    }

    auto tokenizer_params =
            common_context_params_to_llama(params);

    llama_context_ptr tokenizer_ctx {
        llama_init_from_model(model, tokenizer_params)
    };

    if (!tokenizer_ctx) {
        LOG_ERR("%s: failed to create tokenizer context\n", __func__);
        return 1;
    }

    const llama_tokens prompt_tokens =
            common_tokenize(
                    tokenizer_ctx.get(),
                    params.prompt,
                    true);

    tokenizer_ctx.reset();

    if (prompt_tokens.empty()) {
        LOG_ERR("%s: Prompt tokenization returned no tokens\n", __func__);
        return 1;
    }

    LOG_INF(
            "Prompt contains %zu token(s), seed=%u, n_predict=%d\n",
            prompt_tokens.size(),
            params.sampling.seed,
            params.n_predict);

    return run_reference_handoff_test(
            model,
            params,
            prompt_tokens)
        ? 0
        : 1;
}
