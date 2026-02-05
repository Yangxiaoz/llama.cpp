#pragma once

#include "llama.h"
#include "llama-io.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"

#include <set>
#include <list>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <cstring>  
#include <map>
#include <random>
#include <fcntl.h>  
#include <unistd.h> 
#include <sys/mman.h> 
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
#define TABLE_SHIFT         1  //the id of table shift in table
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

enum class submit_type{
    UP = 0,
    GATE = 1,
    DOWN = 2
};

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

struct custom_hparams {
    const bool      optimization;
    const float     utilization       = 0.7f;
    const uint32_t  n_predict_experts = 1;
    const uint32_t  n_prefetch_layers = 1;
    uint32_t        n_preload_layers  = 0;
    uint32_t        head_il           = 0;
    uint32_t        tail_il           = 0;
    uint32_t        n_mean_expert     = 0;
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
struct custom_expert_pool{
    struct ggml_tensor *            up;
    struct ggml_tensor *            gate;
    struct ggml_tensor *            down;
    int32_t                         n_slots;
    std::map<llama_pos, int32_t>    free_slots;
};
struct custom_pool_type{
    ggml_type                           up;
    ggml_type                           gate;
    ggml_type                           down;
};
class custom_async_io{
    private:
    int                 fd;
    io_uring            ring;
    std::mutex          submit_mutex;
    io_uring_params     params{};
    // 状态表：key 是 idx (user_data), value 是当前状态
    // std::unordered_map<int, request_status> request_status_map_;

    //update the signal of cqe to request status map
    uint32_t process_cqe(struct io_uring_cqe* cqe) {
        uint64_t id = cqe->user_data;
        if (id == 0) {
            int res = cqe->res;
            int flags = cqe->flags;
            printf("[CQE] user_data=%lu, res=%d, flags=%d\n", 
                id, res, flags);
                throw std::runtime_error(std::string("error async_submit transmission:: idx_data of 0"));
        }
        // 标记该 CQE 已处理
        io_uring_cqe_seen(&ring, cqe);
        if (cqe->res < 0) {
            int err = -cqe->res;
            throw std::runtime_error(
                std::string("async I/O error: ") + strerror(err)
            );
        }
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

        // 1. 锁定提交队列 (SQ) ring
        if (mlock(ring.sq.ring_ptr, ring.sq.ring_sz)) {
            perror("mlock sq ring failed");
        }
        // 2. 锁定完成队列 (CQ) ring
        if (mlock(ring.cq.ring_ptr, ring.cq.ring_sz)) {
            perror("mlock cq ring failed");
        }
        size_t sqes_size = ring.sq.sqes->len * sizeof(struct io_uring_sqe);
        if (mlock(ring.sq.sqes, sqes_size) != 0) {
            throw std::runtime_error(std::string("Failed to mlock SQEs array: ") + strerror(errno));
        }
    }
    ~custom_async_io(){
        io_uring_queue_exit(&ring);
        close(fd);
    }

    //func
    void block_read(char * buf, size_t size, size_t offset);
    bool async_submit(uint64_t idx,char * buf, size_t size, size_t offset);
    std::vector<uint32_t> refresh_completions();

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
            if (finished_expert_count == n_expert) {
                is_finished = true;
            }
            return true; // 该 expert 刚完成
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
                experts[row][i].state = expert_state::Loading;

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

    int map(int row, int col,submit_type type) const{
        check_indices(row,col);
        return (row * n_col + col)*3 + static_cast<int>(type) + TABLE_SHIFT;
    }

    std::tuple<int, int, submit_type> unmap(int id) const {
        const int data = id - TABLE_SHIFT; 
        const int x = data / 3;
        submit_type type = static_cast<submit_type>(data % 3);
        const int row = x / n_col;
        const int col = x % n_col;

        check_indices(row, col); 
        return {row, col, type};
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
            auto [layer_id, expert_id,type] = unmap(data_id);
            auto is_complete = load_ctx.at(layer_id).transfor_complete(expert_id);
            if(is_complete){
                //layer load complete
                //mark all expert in layer as expertstate::InMemory
                GGML_ASSERT(experts[layer_id][expert_id].state == expert_state::Loading);
                mark(layer_id,expert_id,expert_state::InMemory,INVALID_POS);
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
    uint32_t n_layer;
    int32_t n_expert;
    std::vector<int32_t>                                    n_active;
    // 1. lru_lists[layer_id]: a doubly linked list for each layer, store the expert_id in LRU order
    std::vector<std::list<int32_t>>                         lru_lists;
    // 2. expert_iters[layer_id][expert_id]: a vector of iterators for quick access to each expert's position in the LRU list
    std::vector<std::vector<std::list<int32_t>::iterator>>  layer_iters;
    std::mutex                                              hit_mutex;
    void move_forward(uint32_t layer_id, std::list<int32_t>::iterator it, size_t n_steps) {
        auto& list = lru_lists[layer_id];
        
        if (it == list.begin()) {
            return;
        }
        // calculate the distance from begin to it 
        auto dist = std::distance(list.begin(), it);
        
        std::list<int32_t>::iterator dest_it;
        if ((size_t)dist < n_steps) {
            // if had been there, move to the front
            dest_it = list.begin();
        } else {
            // or move n_steps forward
            dest_it = std::prev(it, n_steps);
        }
        // move the element to the new position
        list.splice(dest_it, list, it);
    }

public: 
    custom_expert_manage(custom_expert_table& table) 
        : n_layer(table.row_size()),
          n_expert(table.col_size()),
          n_active(n_layer, 0)
    {
        // initialize the two-deminsional values
        lru_lists.resize(n_layer);
        layer_iters.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            layer_iters[i].resize(n_expert, lru_lists[i].end());
        }
    }
    
    void active(uint32_t layer_id, int32_t expert_id) {
        GGML_ASSERT(layer_id < n_layer && 0 <= expert_id && expert_id < n_expert);
        
        auto& iters = layer_iters[layer_id];
        auto& list = lru_lists[layer_id];
        GGML_ASSERT(iters[expert_id] == list.end() && "Expert already active");
        list.push_back(expert_id);
        iters[expert_id] = --list.end();
        n_active[layer_id] += 1;
        GGML_ASSERT(n_active[layer_id] <= n_expert && "Active count exceeds number of experts");
    }
    
    void active_layer(uint32_t layer_id) {
        GGML_ASSERT(layer_id < n_layer);
        for(int i = 0; i < n_expert; i++){
            active(layer_id, i);
        }
    }

    void erase(uint32_t layer_id, int32_t expert_id) {
        GGML_ASSERT(layer_id < n_layer && 0 <= expert_id && expert_id < n_expert);
        
        auto& iters = layer_iters[layer_id];
        auto& list = lru_lists[layer_id];

        auto it = iters[expert_id];
        if (it != list.end()) {
            list.erase(it); 
            iters[expert_id] = list.end(); // set to end to indicate not active
        }
        n_active[layer_id] -= 1;
        GGML_ASSERT(n_active[layer_id] >= 0 && "Active count below zero");
    }

    void erase_layer(uint32_t layer_id) {
        GGML_ASSERT(layer_id < n_layer);
        lru_lists[layer_id].clear();
        layer_iters[layer_id].clear();
        layer_iters[layer_id].resize(n_expert, lru_lists[layer_id].end());
        GGML_ASSERT(n_active[layer_id] == n_expert && "Active count should be n_expert before clearing");
        n_active[layer_id] = 0;
    }

    // only touch, do not change the  lru order
    void touch(uint32_t layer_id, int32_t expert_id) {
        GGML_ASSERT(layer_id < n_layer && 0 <= expert_id && expert_id < n_expert);

        auto& iters = layer_iters[layer_id];
        auto& list = lru_lists[layer_id];
        auto it = iters[expert_id];
        GGML_ASSERT(it == list.end());
        active(layer_id, expert_id);
    }

    // hit function
    void hit(uint32_t layer_id, int32_t expert_id, float probs) {
        std::lock_guard<std::mutex> lock(hit_mutex);
        GGML_ASSERT(layer_id < n_layer && 0 <= expert_id && expert_id < n_expert);
        auto& iters = layer_iters[layer_id];
        auto& list = lru_lists[layer_id];
        
        auto it = iters[expert_id];
        if (it != list.end()) {
            // has in the active list
            if(probs >= 0.1){
                list.splice(list.begin(), list, it);
            }else if (probs >= 0.008)
            {
                move_forward(layer_id, it, 20);
            }
            
            else{
                move_forward(layer_id, it, 10);
            }
        } else {
            // activate it
            active(layer_id, expert_id);
            move_forward(layer_id, iters[expert_id], 10);
        }
    }

    /*lru func*/
    /*return the expert id at position n from the back of the list*/
    int32_t get_back_n(uint32_t layer_id, int n) const {
        auto& list = lru_lists[layer_id];
        GGML_ASSERT(n >= 0 && n <= static_cast<int>(list.size()));

        auto rit = list.rbegin();
        std::advance(rit, n); 
        return *rit;
    }
    ///
    /*get the most expert layer*/
    /*if all layer except 'need_layer' have no enough experts,select the prev of cur_layer_id*/
    int32_t get_freeLayer_id(int32_t cur_layer_id,int32_t load_layer_id){
        GGML_ASSERT((cur_layer_id < static_cast<int32_t>(n_layer)) && (load_layer_id < static_cast<int32_t>(n_layer)));

        int32_t select_layer = -1;
        int32_t max_active = -1;
        //traversing all layer to get max active layer_id
        for(int32_t i = 1; i < static_cast<int32_t>(n_layer);i++){
            int32_t id = (cur_layer_id - i) >= 0 ? (cur_layer_id - i) : (n_layer + cur_layer_id - i);
            if((n_active[id] > max_active) && (id != load_layer_id)){
                max_active = n_active[id];
                select_layer = id;
            }
        }
        if (select_layer != -1){
            return select_layer;
        }
        for(int32_t i = 1; i < static_cast<int32_t>(n_layer);i++){
            int32_t id = (cur_layer_id - i) >= 0 ? (cur_layer_id - i) : (n_layer + cur_layer_id - i);
            if((n_active[id] > 0) && (id != load_layer_id)){
                return id;
            }
        }
        return -1;
    }

};

class custom_moe_unified{
public:
    custom_moe_unified(const llama_model & model,const std::string & fname,llama_model_loader & ml,bool enable_optimizations,float utilization,uint32_t n_prefetch_layers);
    ~custom_moe_unified() = default;
    //menbers
    bool                                        optimizations;
    // std::vector<struct perf_time>               perf_t;
    std::unique_ptr<custom_expert_manage>       manage;         //runtime management
    struct custom_hparams                       params;  
    const llama_hparams                         &hparams;
    //
    // custom_moe_unified  API
    //
    uint32_t                                total_size() const;
    void                                    prefill_init();//pre-load the initial some layer of moe_weight
    void                                    check_layer(uint32_t il);
    void                                    check_expert(uint32_t il,uint32_t id);
    bool                                    predict_expert(uint32_t il,uint32_t id);
    void                                    predict_layer(uint32_t head_il,uint32_t tail_il);
    llama_pos                               id_map(uint32_t il, int32_t selec_id);
    ggml_tensor *                           get_ups(struct ggml_context * ctx, uint32_t il) const;
    ggml_tensor *                           get_gates(struct ggml_context * ctx, uint32_t il) const;
    ggml_tensor *                           get_downs(struct ggml_context * ctx, uint32_t il) const;    
    
private:
    const llama_model                       &model;
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
    std::random_device                      rd;  // random seed
    //
    //common
    //
    void        table_init(const llama_model & model,llama_model_loader & ml);
    uint32_t    get_padding() const;
    // void        load_data(llama_pos offset,uint32_t table_row, uint32_t table_col,active_type type,io_mode mode);// no need now
    //
    void        _free_expert(uint32_t il,int32_t id);
    void        _free_layer(uint32_t il);
    void        load_expert(uint32_t il, int32_t id,llama_pos target_pos,io_mode mode);
    void        load_layer(uint32_t il, llama_pos target_pos,io_mode mode);
    llama_pos   free_random(const std::vector<uint32_t>& excluded_layers);
    llama_pos   free_lru(uint32_t cur_layer_id,uint32_t load_layer_id);

    //
    // pool API
    // 
    llama_pos   pool_alloc(int32_t n);
    void        pool_free(llama_pos pos, int32_t n);
};


struct runtime_ctx{
    custom_moe_unified  * moe_unified;
    int   il;
};
//
//custom op_kernel
//
void id_pos_map(struct ggml_tensor * dst , const struct ggml_tensor * a,const struct ggml_tensor * b, int ith, int nth,  void * userdata);

void custom_predict(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth,  void * userdata);