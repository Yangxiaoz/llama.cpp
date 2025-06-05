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
#include <stdexcept>

#define M_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#ifndef NDEBUG
    #define  CUSTOM_ASSERT(x)  GGML_ASSERT(x)
#else
    #define  CUSTOM_ASSERT(x)  ((void)(x))
#endif

struct llama_hparams;
struct llama_model;
struct llama_context;


enum expert_state{
    OnDisk,
    InMemory
};

enum alloc_type{
    Layer   = 1,
    Singel  = 2
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
    custom_expert_table(uint32_t n_moe_layer, uint32_t n_expert);
//members
    // used for loading in prefill step
    llama_files                                     files;
    std::vector<std::vector<custom_expert_group>>   experts;
    // std::vector<table_layer_state>                  layer_state;

// func
    custom_expert_group& at(uint32_t row, uint32_t col){
        check_indices(row,col);
        return experts[row][col];
    }

    llama_pos get_pos(uint32_t row, uint32_t col){
        check_indices(row,col);
        return experts[row][col].pos; 
    }

    void set_pos(uint32_t row, uint32_t col, llama_pos id){
        check_indices(row,col);
        experts[row][col].pos = id;
    }

    void mark(uint32_t row, uint32_t col, expert_state state, llama_pos pos){
        check_indices(row,col);
        if(state == expert_state::InMemory){
            experts[row][col].pos   = pos;
        } else{
            experts[row][col].pos   = -1;
        }
        experts[row][col].state = state;
    }
    void mark_layer(uint32_t row, expert_state state, llama_pos head_pos){
        check_indices(row,0);
        if(state == expert_state::InMemory){
            for(uint32_t i =0; i < n_col; i++){
                experts[row][i].pos = head_pos + i;
                experts[row][i].state = state;
            }
        } else{
            for(uint32_t i =0; i < n_col; i++){
                experts[row][i].pos = -1;
                experts[row][i].state = state;
            }
        }
    }
private:
    uint32_t n_row;
    uint32_t n_col;
    //func:
    void check_indices(uint32_t row, uint32_t col) const {
        GGML_ASSERT((row < n_row)&& (col < n_col) && "expert_table index out of bounds");
    }
};

struct ffn_expert_pool{
    struct ggml_tensor *        up;
    struct ggml_tensor *        gate;
    struct ggml_tensor *        down;
    int32_t                    n_slots;
    std::map<llama_pos, int32_t>    free_slots;
};
class custom_moe_unified{
public:
    custom_moe_unified(const llama_model & model,float utilization,const std::string & fname,llama_model_loader & ml);
    ~custom_moe_unified() = default;
    
    ggml_type                   type_up_gate;
    ggml_type                   type_down;

    uint32_t                    nbyte_up_gate;
    uint32_t                    nbyte_down;
    uint32_t                    nbyte_group;

    class custom_expert_table   table;       //global expert table

//func:
    //
    // custom_moe_unified  API
    //
    uint32_t total_size() const;

    void init_full();//for simulate in worst-case graph_build

    void prefill_init();
    void prefill_step();

    void decode_step();
    

    void expert_check() const;
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

    uint32_t                                n_prefill_loaded;
    uint32_t                                i_cur_layer = 0;
    struct ffn_expert_pool                  pools;       

    // ctxs_bufs for pool
    std::vector<ggml_context_ptr>           ctxs;
    std::vector<ggml_backend_buffer_ptr>    bufs;

    //func

    void        table_init(const llama_model & model,const std::string & fname,llama_model_loader & ml);
    //
    // pools API
    // 
    llama_pos   pool_alloc(int32_t n);
    void        pool_free(llama_pos pos, int32_t n);

    void        load_data(ggml_tensor * dst,llama_pos offset,size_t n_size,uint32_t table_row, uint32_t table_col);
    void        load_expert(uint32_t il, uint32_t id);
    void        load_layer(uint32_t il, llama_pos target_pos);
    
    uint32_t    get_padding() const;
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