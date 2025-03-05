#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <list>  
#include <fstream> // 添加文件流头文件


///////////////////////////
/* Perf_mode list */
//  0:     close getting tensor info 
//  1:     get select_moe expert_id info 
//  2:     get select gate_in logist info
#define Perf_mode  2
////////////////////////

#if Perf_mode == 1
    const char * path = "/home/yxk/workspace/test_log/moe_info_6.log";
    typedef struct save_data
    {
        int32_t * data_ptr;
        int32_t   len;
        // 构造函数
        save_data(int32_t* ptr, int32_t length) : data_ptr(ptr), len(length) {
            GGML_ASSERT((ptr != nullptr)&&(length > 0));
            data_ptr = ptr;
            len = length;
        }
        
        //析构函数
        ~save_data(){
            // free(data_ptr);
        }
    }save_data;
    std::vector<save_data> moe_info_buffer;

    // 用户自定义数据结构（用于传递目标节点名称）
    typedef struct {
        const char * target_node_name; // 要观察的节点名称（如 "ffn_moe_topk-1"）
        bool data_ready;              // 标记数据是否已准备好
        std::vector<save_data>& save_buffer;           // 保存数据的指针
    } my_callback_data;

    // 初始化用户数据
    my_callback_data cb_data = {
        "ffn_moe_topk-",
        false,
        moe_info_buffer
    };
#elif Perf_mode == 2
    const char * path = "/home/yxk/workspace/test_log/gate_info_1.log";
    typedef struct save_data
    {
        float * data_ptr;
        int32_t   len;
        
        // 构造函数
        save_data(float* ptr, int32_t length) : data_ptr(ptr), len(length) {
            GGML_ASSERT((ptr != nullptr)&&(length > 0));
            data_ptr = ptr;
            len = length;
        }
        
        //析构函数
        ~save_data(){
            // free(data_ptr);
        }
    }save_data;
    std::vector<save_data> gate_info_buffer;

    // 用户自定义数据结构（用于传递目标节点名称）
    typedef struct {
        const char * target_node_name; // 要观察的节点名称（如 "ffn_moe_topk-1"）
        bool data_ready;              // 标记数据是否已准备好
        std::vector<save_data>& save_buffer;           // 保存数据的指针
    } my_callback_data;

    // 初始化用户数据
    my_callback_data cb_data = {
        "ffn_norm-",
        false,
        gate_info_buffer
    };
#endif



static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}



int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
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

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    model_params.use_mmap = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }
    /////////////////////////////////////////////////////////
    // initialize the context
    /*custom callback by yxl;*/
    ////////////////////////////////////////////////////
    #if Perf_mode == 1
        auto my_callback = [](ggml_tensor *t, bool ask, void * user_data) -> bool {
            my_callback_data * data = (my_callback_data *)user_data;

            if (ask) {
                // 阶段1：调度器询问是否需要观察此节点
                // 检查节点名称是否匹配目标
                size_t prefix_len = strlen(data->target_node_name);
                if (strncmp(t->name, data->target_node_name, prefix_len) == 0) {
                    data->data_ready = false;
                    return true; // 需要观察此节点
                }
                return false;    // 其他节点不观察
            } else {
                // 阶段2：调度器传递计算后的数据
                // 确保是目标节点
                size_t prefix_len = strlen(data->target_node_name);
                if (strncmp(t->name, data->target_node_name, prefix_len) == 0) {
                    // 数据已计算完成，保存到独立内存
                    
                    size_t num_elements = ggml_nelements(t);
                    int32_t * data_local = (int32_t *) malloc(num_elements*sizeof(int32_t));
                    save_data cur(data_local,num_elements);              
                    // GGML_ASSERT()
                    size_t offsize = 0;;
                    if(ggml_n_dims(t) == 1){
                        for(int y = 0; y < t->ne[0]; y++){\
                            GGML_ASSERT(offsize < num_elements && "erro in reading the selec_moe_info");
                            ggml_backend_tensor_get(t, data_local++, y * t->nb[0], sizeof(int32_t));
                            offsize++;
                        }
                    }else{
                        for(int x = 0; x < t->ne[1]; x++){
                            for(int y = 0; y < t->ne[0]; y++){
                                GGML_ASSERT(offsize < num_elements && "erro in reading the selec_moe_info");
                                ggml_backend_tensor_get(t, data_local++, y * t->nb[0] + x* t->nb[1], sizeof(int32_t));
                                offsize++;
                            }
                        }
                    }

                    data->save_buffer.emplace_back(cur);
                    data->data_ready = true;
                }
                return true; // 继续计算
            }
        };
    #elif Perf_mode ==2
        auto my_callback = [](ggml_tensor *t, bool ask, void * user_data) -> bool {
            static int count = 0;
            my_callback_data * data = (my_callback_data *)user_data;

            if (ask) {
                // 阶段1：调度器询问是否需要观察此节点
                // 检查节点名称是否匹配目标

                size_t prefix_len = strlen(data->target_node_name);
                if (strncmp(t->name, data->target_node_name, prefix_len) == 0 && strlen(t->name) == 10) {
                    data->data_ready = false;
                    count++;
                    if(count > 3 && count <= 5){
                        return true;// 需要观察此节点
                    }
                    return false; 
                }
                return false;
            } else {
                // 阶段2：调度器传递计算后的数据
                // 确保是目标节点
                size_t prefix_len = strlen(data->target_node_name);
                if (strncmp(t->name, data->target_node_name, prefix_len) == 0) {
                    // 数据已计算完成，保存到独立内存
                    
                    size_t num_elements = ggml_nelements(t);
                    float * data_local = (float *) malloc(num_elements*sizeof(float));
                    save_data cur(data_local,num_elements);              
                    // GGML_ASSERT()
                    size_t offsize = 0;;
                    if(ggml_n_dims(t) == 1){
                        for(int y = 0; y < t->ne[0]; y++){\
                            GGML_ASSERT(offsize < num_elements && "erro in reading the selec_moe_info");
                            ggml_backend_tensor_get(t, data_local++, y * t->nb[0], sizeof(float));
                            offsize++;
                        }
                    }else{
                        for(int x = 0; x < t->ne[1]; x++){
                            for(int y = 0; y < t->ne[0]; y++){
                                GGML_ASSERT(offsize < num_elements && "erro in reading the selec_moe_info");
                                ggml_backend_tensor_get(t, data_local++, y * t->nb[0] + x* t->nb[1], sizeof(float));
                                offsize++;
                            }
                        }
                    }

                    data->save_buffer.emplace_back(cur);
                    data->data_ready = true;
                }
                return true; // 继续计算
            }
        };
    #endif

    llama_context_params ctx_params = llama_context_default_params();
    //register user function
    #if Perf_mode != 0
        ctx_params.cb_eval = my_callback;
        ctx_params.cb_eval_user_data = &cb_data;
    #endif

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

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

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
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
 
        
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

        // extract the moe info
        #if Perf_mode == 1
        {
            int count = 0;
            int n_itera = 0;
            std::ofstream outfile(path, std::ios::out);
            if (!outfile.is_open()) {
                fprintf(stderr, "%s: error: Failed to open log file!\n", __func__);
            }
            for (const auto& sd : moe_info_buffer) {
                // 打印数组长度
                int n_batch = sd.len/6;
                if(count == 0){
                    outfile << "Bath "<<n_itera<<" has "<<n_batch<<" tokens"<< "\n";
                    outfile << "--------------------------------------------------\n";
                }
                outfile<<"the "<<count++<<" layers:\n";
                // outfile << "Array length: " << sd.len << "\n";
                for(int i =0; i < n_batch; i++){
                    outfile << "token "<< i<<" 's expert id select: ";
                    for (int32_t j = 0; j < 6; ++j) {
                        outfile << sd.data_ptr[j + i * 6] << " ";
                    }
                    outfile << ",\n";
                }
                    
                outfile << "----------------\n";
                free(sd.data_ptr);
                if (count == 26)
                {
                    count = 0;
                    n_itera++;
                }
            }
        outfile.close(); // 显式关闭文件（非必须，RAII 会自动处理）
        printf("%s \n",path);
        }
        #elif Perf_mode == 2
        {//batch 2, len = 32768
            int count = 0;
            std::ofstream outfile(path, std::ios::out);
            if (!outfile.is_open()) {
                fprintf(stderr, "%s: error: Failed to open log file!\n", __func__);
            }
            for (const auto& sd : gate_info_buffer) {
                // 打印数组长度

                outfile << "Layer "<<count++<<": has "<<sd.len<<" num"<< "\n";
                outfile << "--------------------------------------------------\n";
                
                // outfile<<"the "<<count++<<" layers:\n";
                // outfile << "Array length: " << sd.len << "\n";
                for(size_t i =0; i < 16; i++){
                    outfile << "token "<< i<<" 's gate logist info: ";
                    for (int32_t j = 0; j < 2048; ++j) {
                        outfile << sd.data_ptr[j + i * 2048] << " ";
                    }
                    outfile << "\n";
                }
                    
                outfile << "----------------\n";
                free(sd.data_ptr);
            }
            outfile.close(); // 显式关闭文件（非必须，RAII 会自动处理）
            printf("%s \n",path);
        }
        #endif
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
