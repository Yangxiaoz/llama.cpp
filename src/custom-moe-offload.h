#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"

#include <set>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <cstring>  
#include <map>
#include <random>
#include <fcntl.h>  
#include <unistd.h>  
#include <liburing.h> 


#ifndef NDEBUG
    #define  CUSTOM_ASSERT(x)  GGML_ASSERT(x)
#else
    #define  CUSTOM_ASSERT(x)  ((void)(x))
#endif
#define M_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//general
#define NUM_EXPERT          3   //up、gate、down
#define LAYER_HEAD          0  //mean the id of layer_head in table
#define TABLE_SHIFT         0  //the id of table shift in table
#define INVALID_POS        -1
struct llama_hparams;
struct llama_model;
struct llama_context;

enum class io_mode{
    block,
    async
};


// enum class request_status {
//     NOT_FOUND, 
//     LOADING,   
//     COMPLETED, 
//     FAILED    
// };

enum class expert_state{
    OnDisk,
    Loading,
    InMemory
};

enum class active_type{
    Layer,
    Expert
};

struct perf_time{
    int il;
    std::array<int64_t, 27>         load_time;
    std::array<int64_t, 27>         compute_time;
};


//used for async read request
// struct ReadRequest {
//     char* buffer;      
//     uint32_t size;     
//     size_t offset;     
// };

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
struct custom_expert_pool{
    struct ggml_tensor *            up;
    struct ggml_tensor *            gate;
    struct ggml_tensor *            down;
    int32_t                         n_slots;
    std::map<llama_pos, int32_t>    free_slots;
};

class custom_async_io{
    private:
    int                 fd;
    io_uring            ring;
    io_uring_params     params{};
    // 状态表：key 是 idx (user_data), value 是当前状态
    // std::unordered_map<int, request_status> request_status_map_;

    //update the signal of cqe to request status map
    uint32_t process_cqe(struct io_uring_cqe* cqe) {
        uint64_t id = cqe->user_data;
        if (cqe->res < 0) {
            int err = -cqe->res;
            throw std::runtime_error(
                std::string("async I/O error: ") + strerror(err)
            );
        }
        // 标记该 CQE 已处理
        io_uring_cqe_seen(&ring, cqe);
        return static_cast<uint32_t>(id);
        }

    public:

    custom_async_io(const std::string & fname,int queue_depth = 640){
        //no params is no used yet
        int ret = io_uring_queue_init_params(queue_depth, &ring, &params);
        if (ret < 0) {
            throw std::runtime_error("Failed to initialize io_uring: " + std::string(strerror(-ret)));
        }
        fd = open(fname.c_str(), O_RDONLY);// note: O_DIRECT is currently not be support
        if (fd < 0) {
            throw std::runtime_error("Failed to open file: " + std::string(strerror(errno)));
        }
        //register the fd 
        int fixed_fds[] = {fd};
        io_uring_register_files(&ring,fixed_fds,1);

    }
    ~custom_async_io(){
        io_uring_queue_exit(&ring);
        close(fd);
    }

    //func
    void block_read(char * buf, size_t size, size_t offset);
    bool async_submit(int idx,char * buf, size_t size, size_t offset);
    std::vector<uint32_t> refresh_completions();
    // void wait_for_request(int target_idx); //no used now
    // void block_check();
};


struct LoadContext {
    bool is_finished;
    bool is_started;
    const size_t n_expert; // 动态的 expert 数量
    size_t finished_expert_count; // 已完成的 expert 数量
    std::vector<uint8_t> pending_io; // 每个 expert 剩余的 IO 请求数


    explicit LoadContext(size_t n_expert) 
        : is_finished(false),
          is_started(false),
          n_expert(n_expert),
          finished_expert_count(0),
          pending_io(n_expert) // 初始化 vector 大小
    {
        for (size_t i = 0; i < n_expert; ++i) {
            pending_io[i] = 3; // 每个 expert 有 3 个权重块
        }
    }

    bool is_layer_started() const {
        return is_started;
    }
    bool is_layer_finished() const {
        return is_finished;
    }

    /* return true if the expert is finished */
    bool transfor_complete(size_t expert_idx) {
        GGML_ASSERT(expert_idx < n_expert && "expert_idx out of bounds");
  
        // 原子减少 pending_io 计数
        pending_io[expert_idx] = pending_io[expert_idx] - 1;
        uint8_t cur_count = pending_io[expert_idx];
        
        if (cur_count == 0) {
            finished_expert_count += 1;
            pending_io[expert_idx] = 3; // 重置为 3，准备下一次加载
            return true; // 该 expert 刚完成
        }
        if (finished_expert_count == n_expert) {
            is_finished = true;
        }
        return false;
    }
};

class custom_expert_table{
private:
    uint32_t n_row;
    int32_t n_col;
    std::vector<struct LoadContext>                 load_ctx;//for async load management
    //func:
    void check_indices(uint32_t row, int32_t col) const {
        GGML_ASSERT((row < n_row)&& (0 <= col)&& (col < n_col) && "expert_table index out of bounds");
    }

public:
    custom_expert_table(uint32_t n_moe_layer, uint32_t n_expert);
//members
    std::vector<std::vector<custom_expert_group>>   experts;
    std::vector<std::array<ggml_type, NUM_EXPERT>>  layer_type; //up gate down
    
// func
    custom_expert_group& at(uint32_t row, int32_t col){
        check_indices(row,col);
        return experts[row][col];
    }

    llama_pos get_pos(uint32_t row, int32_t col){
        check_indices(row,col);
        return experts[row][col].pos; 
    }

    void mark(uint32_t row, int32_t col, expert_state state, llama_pos pos){
    //note: when the state is InMemory, the pos is no used
        check_indices(row,col);
        if(state == expert_state::OnDisk){
            experts[row][col].pos   = -1;
        } else if(state == expert_state::Loading){
            experts[row][col].pos   = pos;
        }else if(state == expert_state::InMemory){
            // Do nothing with pos
        }
        experts[row][col].state = state;
    }

    void mark_layer(uint32_t row, expert_state state, llama_pos head_pos){
        check_indices(row,0);
        if(state == expert_state::OnDisk){
            for(int32_t i =0; i < n_col; i++){
                experts[row][i].pos = -1;
                experts[row][i].state = expert_state::OnDisk;
            }
        } else if(state == expert_state::Loading){
            for(int32_t i =0; i < n_col; i++){
                experts[row][i].pos = head_pos + i;
                //layer load no need to set the pos of each expert for loading
                // experts[row][i].state = expert_state::Loading;
            }
            load_ctx.at(row).is_started = true; 
        }
        else if(state == expert_state::InMemory){
            for(int32_t i =0; i < n_col; i++){
                experts[row][i].state = expert_state::InMemory;
            }
            load_ctx.at(row).is_finished = true;
            load_ctx.at(row).is_started = false;
        }
        
    }

    int map(int row, int col) const{
        check_indices(row,col);
        return row * n_col + col + TABLE_SHIFT;
    }

    std::pair<int, int> unmap(int id) const {
        const int x = id - TABLE_SHIFT; 
        const int row = x / n_col;
        const int col = x % n_col;

        check_indices(row, col); 
        return {row, col};
    }

    bool layer_check(int il) const{
        GGML_ASSERT((il >= 0)&& (il < static_cast<int>(n_row)) && "expert_table layer_check index out of bounds");
        return load_ctx.at(il).is_layer_finished();
    }

    /* return true if the layer is started */
    bool is_layer_trigger(int il) const{
        GGML_ASSERT((il >= 0)&& (il < static_cast<int>(n_row)) && "expert_table layer_check index out of bounds");
        return load_ctx.at(il).is_layer_started();
    }
    
    void refresh(class custom_async_io& async_io){
        auto compelete_ids = async_io.refresh_completions();
        for (auto data_id : compelete_ids) {
            //get il and expert_id from id
            auto [layer_id, expert_id] = unmap(data_id);
            auto is_complete = load_ctx.at(layer_id).transfor_complete(expert_id);
            if(is_complete){
                //layer load complete
                //mark all expert in layer as expertstate::InMemory
                mark_layer(layer_id,expert_state::InMemory,INVALID_POS);
            }
        }
    }
    
    uint32_t row_size(){
        return n_row;
    }
    int32_t col_size(){
        return n_col;
    }

};

class custom_expert_manage {

private:
    struct ActiveUnit {
        uint32_t layer_id;
        int32_t expert_id;
        uint32_t hit_count;

        bool operator<(const ActiveUnit& rhs) const {
            return std::tie(hit_count, layer_id, expert_id) < 
                   std::tie(rhs.hit_count, rhs.layer_id, rhs.expert_id);
        }
    };
    custom_expert_table&                                            table;
    custom_expert_pool&                                             pool;
    std::vector<std::set<ActiveUnit>>                               active_set;// a set for indicate the active experts in each layer
    std::unordered_map<int, std::unordered_map<int, ActiveUnit>>    active_map;//for quick access the unit according to (layer_id,expert_id)                

public: 
    custom_expert_manage(custom_expert_table& table,custom_expert_pool& pool);
    
    void active(uint32_t layer_id,int32_t expert_id){
        GGML_ASSERT(layer_id < table.row_size() && 0 <= expert_id && expert_id < table.col_size());  
        auto& unit = active_map[layer_id][expert_id]; 
        active_set[layer_id].insert(unit);
    }
    void erase(uint32_t layer_id,int32_t expert_id){
        GGML_ASSERT(layer_id < table.row_size() && 0 <= expert_id && expert_id < table.col_size());  
        auto& unit = active_map[layer_id][expert_id]; 
        active_set[layer_id].erase(unit);
    }
    void erase_layer(uint32_t layer_id){
        GGML_ASSERT(layer_id < table.row_size() );  
        for(int i = 0; i < table.col_size(); i++){
            auto& unit = active_map[layer_id][i]; 
            active_set[layer_id].erase(unit);
        }
    }
    void active_layer(uint32_t layer_id){
        GGML_ASSERT(layer_id < table.row_size());  
        for(int i = 0; i < table.col_size(); i++){
            auto& unit = active_map[layer_id][i]; 
            active_set[layer_id].insert(unit);
        }
    }
    void hit(uint32_t layer_id,int32_t expert_id){
        GGML_ASSERT(layer_id < table.row_size() && 0 <= expert_id && expert_id < table.col_size());      

        auto& unit = active_map[layer_id][expert_id];
        active_set[layer_id].erase(unit);
        unit.hit_count++;
        active_set[layer_id].insert(unit);
    }

    uint32_t get_unit(uint32_t layer_id){
        GGML_ASSERT(layer_id < table.row_size());
        if (active_set[layer_id].empty()) return -1;
        auto cur = active_set[layer_id].begin();
        int id = cur->expert_id;
        active_set[layer_id].erase(cur);
        return id;
    }
};

class custom_moe_unified{
public:
    custom_moe_unified(const llama_model & model,float utilization,const std::string & fname,llama_model_loader & ml);
    ~custom_moe_unified() = default;
    //menbers
    std::vector<struct perf_time>               perf_t;
    std::unordered_map<std::string,uint32_t>    name_layer_map;   //mapping when runtime
    std::unique_ptr<custom_expert_manage>       manage;         //runtime management
    //
    // custom_moe_unified  API
    //
    uint32_t                                total_size() const;
    void                                    prefill_init();//pre-load the initial some layer of moe_weight
    // void                                    table_refresh();//no need now
    void                                    check_layer(uint32_t il);
    void                                    check_expert(uint32_t il,uint32_t id);
    llama_pos                               id_map(uint32_t il, int32_t selec_id);
    ggml_tensor *                           get_ups(struct ggml_context * ctx, uint32_t il) const;
    ggml_tensor *                           get_gates(struct ggml_context * ctx, uint32_t il) const;
    ggml_tensor *                           get_downs(struct ggml_context * ctx, uint32_t il) const;    
    
private:
    const llama_model                       &model;
    const llama_hparams                     &hparams;
    struct custom_pool_type{
        ggml_type                           up;
        ggml_type                           gate;
        ggml_type                           down;
    };
    //menbers
    uint32_t                                expert_ne[2];
    uint32_t                                nbyte_slot_up;
    uint32_t                                nbyte_slot_gate;
    uint32_t                                nbyte_slot_down;
    uint32_t                                n_slots_layer;    // number of slots in one layer
    struct  custom_pool_type                pool_type;
    struct  custom_expert_pool              pool;
    class   custom_async_io                 async_io;       
    class   custom_expert_table             table;    
    std::vector<ggml_context_ptr>           ctxs;
    std::vector<ggml_backend_buffer_ptr>    bufs;
    //
    //common
    //
    void        table_init(const llama_model & model,llama_model_loader & ml);
    uint32_t    get_padding() const;
    // void        load_data(llama_pos offset,uint32_t table_row, uint32_t table_col,active_type type,io_mode mode);// no need now
    //
    void        load_expert(uint32_t il, int32_t id,llama_pos target_pos,io_mode mode);
    void        load_layer(uint32_t il, llama_pos target_pos,io_mode mode);
    void        free_expert(uint32_t il,int32_t id);
    void        free_layer(uint32_t il);
    //
    // pool API
    // 
    llama_pos   pool_alloc(int32_t n);
    void        pool_free(llama_pos pos, int32_t n);
};

//
//custom op_kernel
//
void id_pos_map(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth,  void * userdata);

