#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef CUSTOM_MOE
// ----------------------------------------------------------------------------
// custom MoE offload options
//
// self-contained: defaults, parsing and usage text all live in this block, so
// that the rest of the example stays identical to upstream when CUSTOM_MOE is
// disabled
// ----------------------------------------------------------------------------

struct custom_moe_params {
    float    memory_utilization   = 0.46f; // fraction of the device memory used as the expert pool, 0 < F < 1
    bool     enable_optimizations = false; // LRU-based expert replacement instead of random
    uint32_t n_prefetch_layers    = 1;     // number of layers to prefetch ahead
};

enum custom_moe_arg {
    CUSTOM_MOE_ARG_IGNORED, // not a custom MoE option - the caller keeps parsing
    CUSTOM_MOE_ARG_OK,      // parsed and consumed
    CUSTOM_MOE_ARG_INVALID, // malformed value - the caller should print the usage
};

// try to consume argv[i] (and its value, if any) as a custom MoE option
static custom_moe_arg custom_moe_parse_arg(int argc, char ** argv, int & i, custom_moe_params & moe) {
    const char * const arg = argv[i];

    const bool is_mu = strcmp(arg, "-mu") == 0 || strcmp(arg, "--moe-memory-utilization") == 0;
    const bool is_mp = strcmp(arg, "-mp") == 0 || strcmp(arg, "--moe-prefetch-layers")   == 0;
    const bool is_mo = strcmp(arg, "-mo") == 0 || strcmp(arg, "--moe-optimizations")     == 0;

    if (!is_mu && !is_mp && !is_mo) {
        return CUSTOM_MOE_ARG_IGNORED;
    }
    if (i + 1 >= argc) {
        return CUSTOM_MOE_ARG_INVALID; // missing value
    }

    const char * const value = argv[++i];

    try {
        if (is_mu) {
            moe.memory_utilization = std::stof(value);
            // the model loader requires 0 < utilization < 1
            if (!(moe.memory_utilization > 0.0f && moe.memory_utilization < 1.0f)) {
                return CUSTOM_MOE_ARG_INVALID;
            }
        } else if (is_mp) {
            const int n_layer = std::stoi(value);
            if (n_layer < 0) {
                return CUSTOM_MOE_ARG_INVALID;
            }
            moe.n_prefetch_layers = (uint32_t) n_layer;
        } else {
            moe.enable_optimizations = std::stoi(value) != 0;
        }
    } catch (...) {
        return CUSTOM_MOE_ARG_INVALID;
    }

    return CUSTOM_MOE_ARG_OK;
}

static void custom_moe_print_usage(void) {
    printf("\ncustom MoE offload options:\n");
    printf("\n    -mu, --moe-memory-utilization F   fraction of the device memory used as the expert pool, 0 < F < 1 (default: 0.46)\n");
    printf("\n    -mp, --moe-prefetch-layers N      number of layers to prefetch ahead (default: 1)\n");
    printf("\n    -mo, --moe-optimizations 0|1      use LRU-based expert replacement (default: 0)\n");
}

// write the custom options into the model parameters
static void custom_moe_apply(const custom_moe_params & moe, llama_model_params & model_params) {
    model_params.moe_memory_utilization   = moe.memory_utilization;
    model_params.moe_enable_optimizations = moe.enable_optimizations;
    model_params.n_prefetch_layers        = moe.n_prefetch_layers;

    // the on-demand expert loading reads the weights straight from the file,
    // so mmap has to be off - this is not a tunable, it is a requirement
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
}
#endif // CUSTOM_MOE

// fill the batch with tokens at consecutive positions starting from pos_0, output logits only for the last one
static void batch_set_tokens(llama_batch_ext * batch, const llama_token * tokens, int32_t n_tokens, llama_pos pos_0) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = pos_0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n_tokens - 1, true);
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
#ifdef CUSTOM_MOE
    custom_moe_print_usage();
#endif
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;
#ifdef CUSTOM_MOE
    // custom MoE offload options, filled in by custom_moe_parse_arg() below
    custom_moe_params moe_params;
#endif

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
#ifdef CUSTOM_MOE
            const custom_moe_arg moe_arg = custom_moe_parse_arg(argc, argv, i, moe_params);
            if (moe_arg == CUSTOM_MOE_ARG_INVALID) {
                print_usage(argc, argv);
                return 1;
            }
            if (moe_arg == CUSTOM_MOE_ARG_OK) {
                continue;
            }
#endif
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    llama_backend_init();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

#ifdef CUSTOM_MOE
    // custom MoE offload: apply the options parsed from the command line
    custom_moe_apply(moe_params, model_params);
#endif

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    int n_tokens = n_prompt; // number of tokens in the current batch

    batch_set_tokens(batch, prompt_tokens.data(), n_prompt, 0);

    if (llama_model_has_encoder(model)) {
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch_set_tokens(batch, &decoder_start_token_id, 1, 0);
        n_tokens = 1;
    }

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch_set_tokens(batch, &new_token_id, 1, n_pos);
            n_tokens = 1;

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
