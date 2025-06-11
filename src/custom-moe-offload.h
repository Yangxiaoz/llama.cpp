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
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#ifndef NDEBUG
    #define  CUSTOM_ASSERT(x)  GGML_ASSERT(x)
#else
    #define  CUSTOM_ASSERT(x)  ((void)(x))
#endif

#define NUM_EXPERT          3   //up gate down
#define LAYER_HEAD          0  //mean the id of layer_head in table


struct llama_hparams;
struct llama_model;
struct llama_context;


enum expert_state{
    OnDisk,
    InMemory
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
    
    llama_files                                     files;
    std::vector<std::vector<custom_expert_group>>   experts;
    std::vector<std::array<ggml_type, NUM_EXPERT>>  layer_type; //up gate down
    std::vector<expert_state>                       layer_state;//used for layer_check when prefill

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
        layer_state[row] = state;
    }
private:
    uint32_t n_row;
    uint32_t n_col;
    //func:
    void check_indices(uint32_t row, uint32_t col) const {
        GGML_ASSERT((row < n_row)&& (col < n_col) && "expert_table index out of bounds");
    }
};


class custom_moe_unified{
public:
    custom_moe_unified(const llama_model & model,float utilization,const std::string & fname,llama_model_loader & ml);
    ~custom_moe_unified() = default;

    int                                         ne[2];
    uint32_t                                    nbyte_slot_up;
    uint32_t                                    nbyte_slot_gate;
    uint32_t                                    nbyte_slot_down;
    uint32_t                                    n_slots_layer;    // number of slots in one layer
    std::unordered_map<std::string,uint32_t>    name_layer_map;   //mapping when runtime

//func:
    //
    // custom_moe_unified  API
    //
    uint32_t                                total_size() const;
    void                                    prefill_init();
    void                                    check_layer(uint32_t il);
    void                                    check_expert(uint32_t il,uint32_t id);
    llama_pos                               id_map(uint32_t il, int32_t selec_id);
    //struct ggml_context * ctx
    ggml_tensor *                           get_ups(struct ggml_context * ctx, uint32_t il) const;
    ggml_tensor *                           get_gates(struct ggml_context * ctx, uint32_t il) const;
    ggml_tensor *                           get_downs(struct ggml_context * ctx, uint32_t il) const;    
    
private:
    const llama_model                   &model;
    const llama_hparams                 &hparams;
    struct custom_expert_pool{
        struct ggml_tensor *            up;
        struct ggml_tensor *            gate;
        struct ggml_tensor *            down;
        int32_t                         n_slots;
        std::map<llama_pos, int32_t>    free_slots;
    };
    struct custom_pool_type{
        ggml_type                       up;
        ggml_type                       gate;
        ggml_type                       down;
    };
    //menbers
    struct  custom_pool_type            pool_type;
    struct  custom_expert_pool          pool;       
    class   custom_expert_table         table;       //global expert table
    std::vector<ggml_context_ptr>       ctxs;
    std::vector<ggml_backend_buffer_ptr>bufs;
    //
    //common
    //
    void        table_init(const llama_model & model,const std::string & fname,llama_model_loader & ml);
    uint32_t    get_padding() const;
    //
    // pool API
    // 
    llama_pos   pool_alloc(int32_t n);
    void        pool_free(llama_pos pos, int32_t n);

    void        load_data(llama_pos offset,uint32_t table_row, uint32_t table_col);
    void        load_expert(uint32_t il, uint32_t id,llama_pos target_pos);
    void        load_layer(uint32_t il, llama_pos target_pos);
    
};



void id_pos_map(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth,  void * userdata);

