#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"
#include "ggml-cpp.h"

#include <set>
#include <vector>

struct llama_hparams;
struct llama_model;
struct llama_context;


struct ffn_expert_group{
    struct ggml_tensor * up;
    struct ggml_tensor * gate;
    struct ggml_tensor * down;
};

enum expert_state{
    InMemory,
    OnDisk
};

struct custom_tensor_mmap{
    uint16_t idx;
    size_t offs;
};

using tensor_mmap_entry = std::pair<expert_state, custom_tensor_mmap>;

struct custom_expert_group{
    struct custom_tensor_mmap up;
    struct custom_tensor_mmap gate;
    struct custom_tensor_mmap down;
    expert_state state;
    int pool_id;
};

class custom_expert_table{
public:

    custom_expert_table(int n_moe_layer, int n_expert);

    //members
    std::vector<std::vector<custom_expert_group>> experts;
    llama_files files;

    // func
    custom_expert_group& at(int row, int col){
        check_indices(row,col);
        return experts[row][col];
    }
private:
    int n_row;
    int n_col;

    void check_indices(int row, int col) const {
        GGML_ASSERT((row < n_row)&& (col < n_col) && "expert_table index out of bounds");
    }
};


// struct custom_moe_pool: public llama_memory_i{
// //TBD:...

// };


class custom_moe_unified: public llama_memory_i{
public:
    custom_moe_unified(const llama_model & model,ggml_backend_buffer_type_t buft,float utilization);

    ~custom_moe_unified() = default;
    
    ggml_type                               type_expert;
    size_t                                  nbyte_expert;
    size_t                                  nbyte_layer_experts;
    //global expert table
    class custom_expert_table               table;
    //available expert pool
    std::vector<struct ffn_expert_group>    groups;

    //////////////////////
    //llama_memory_i
    //TBD:
    void clear() override;
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    virtual bool get_can_edit() const override;
    ///////////////////
    
private:
    const llama_model                       &model;
    const llama_hparams                     &hparams;


    // ctxs_bufs for pool
    std::vector<ggml_context_ptr>           ctxs;
    std::vector<ggml_backend_buffer_ptr>    bufs;

    //func
    //TBD
    size_t total_size() const;

    void prefill_load_init();
    
    llm_graph_result_ptr build_moe_predic();
};