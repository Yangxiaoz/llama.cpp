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

#define M_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#ifndef NDEBUG
    #define  CUSTOM_ASSERT(x)  GGML_ASSERT(x)
#else
    #define  CUSTOM_ASSERT(x)  ((void)(x))
#endif
struct llama_hparams;
struct llama_model;
struct llama_context;


struct ffn_expert_pool{
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
    llama_pos pos;
};

class custom_expert_table{
public:

    custom_expert_table(int n_moe_layer, int n_expert);

    //members
    llama_files files;
    std::vector<std::vector<custom_expert_group>> experts;

    // func
    custom_expert_group& at(int row, int col){
        check_indices(row,col);

        return experts[row][col];
    }

    llama_pos get_pos(int row, int col){
        check_indices(row,col);
        return experts[row][col].pos; 
    }

    void set_pos(int row, int col, llama_pos id){
        check_indices(row,col);
        experts[row][col].pos = id;
    }


private:
    int n_row;
    int n_col;
    //func:
    void check_indices(int row, int col) const {
        GGML_ASSERT((row < n_row)&& (col < n_col) && "expert_table index out of bounds");
    }
};


class custom_moe_unified{
public:
    custom_moe_unified(const llama_model & model,float utilization,const std::string & fname,llama_model_loader & ml);
    ~custom_moe_unified() = default;
    
    ggml_type                               type_expert;
    uint32_t                                nbyte_expert;
    class custom_expert_table               table;       //global expert table

//func:
    //
    // custom_moe_unified  API
    //
    uint32_t total_size() const;

    void init_full();
    void init_prefill();

    void expert_check();
    void expert_fetch();
    void sync();
    // void defrag_sched();

    //
    // graph_build API
    //  
    ggml_tensor * get_up() const;
    ggml_tensor * get_gate() const;
    ggml_tensor * get_down() const;

    
private:
    const llama_model                       &model;
    const llama_hparams                     &hparams;

    //available expert pool
    // custom_moe_cells_unified                cells;
    std::vector<struct ffn_expert_pool>     pools;       

    // ctxs_bufs for pool
    std::vector<ggml_context_ptr>           ctxs;
    std::vector<ggml_backend_buffer_ptr>    bufs;

    //func
    //TBD
    void table_init(const llama_model & model,const std::string & fname,llama_model_loader & ml);
    uint32_t get_padding() const;

    // llm_graph_result_ptr build_moe_predic();
};


// class custom_moe_cells_unified {
// public:
//     void reset(){
//         for (uint32_t i = 0; i < pos.size(); ++i) {
//             pos[i]   = -1;
//             shift[i] =  0;
//         }

//         used.clear();
//     }

//     uint32_t size() const {
//         return 0;
//     }

//     void resize(uint32_t n) {
//         pos.resize(n);
//         shift.resize(n);

//         reset();
//     }

// private:
//     std::set<uint32_t> used;

//     std::vector<llama_pos> pos;

//     std::vector<llama_pos> shift;
// };