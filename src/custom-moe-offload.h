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
    io_uring            ring;       // async reads only (prefetch), completions handled by refresh_completions()
    io_uring            block_ring; // blocking reads only, so a wait can never consume an async completion
    std::mutex          submit_mutex;
    // 状态表：key 是 idx (user_data), value 是当前状态
    // std::unordered_map<int, request_status> request_status_map_;

    //update the signal of cqe to request status map, return {user_data, bytes read}
    std::pair<uint32_t, int> process_cqe(struct io_uring_cqe* cqe) {
        const uint64_t id  = cqe->user_data;
        const int      res = cqe->res;
        if (id == 0) {
            printf("[CQE] user_data=%lu, res=%d, flags=%d\n",
                id, res, cqe->flags);
                throw std::runtime_error(std::string("error async_submit transmission:: idx_data of 0"));
        }
        // 标记该 CQE 已处理 (after reading it: the slot may be reused once seen)
        io_uring_cqe_seen(&ring, cqe);
        if (res < 0) {
            throw std::runtime_error(
                std::string("async I/O error: ") + strerror(-res)
            );
        }
        return {static_cast<uint32_t>(id), res};
    }

    void init_ring(io_uring & r, unsigned queue_depth){
        io_uring_params params{};
        int ret = io_uring_queue_init_params(queue_depth, &r, &params);
        if (ret < 0) {
            throw std::runtime_error("Failed to initialize io_uring: " + std::string(strerror(-ret)));
        }
        //register the fd
        int fixed_fds[] = {fd};
        ret = io_uring_register_files(&r,fixed_fds,1);
        if (ret < 0) {
            throw std::runtime_error("Failed to register file to io_uring: " + std::string(strerror(-ret)));
        }

        // mlock is best effort: a low RLIMIT_MEMLOCK must not stop the load
        if (mlock(r.sq.ring_ptr, r.sq.ring_sz)) {
            perror("mlock sq ring failed");
        }
        if (mlock(r.cq.ring_ptr, r.cq.ring_sz)) {
            perror("mlock cq ring failed");
        }
        if (mlock(r.sq.sqes, r.sq.ring_entries * sizeof(struct io_uring_sqe))) {
            perror("mlock sqes failed");
        }
    }

    public:

    custom_async_io(const std::string & fname,int queue_depth = 640){
        fd = open(fname.c_str(), O_RDONLY);// note: O_DIRECT is currently not be support
        if (fd < 0) {
            throw std::runtime_error("Failed to open file: " + std::string(strerror(errno)));
        }
        init_ring(ring, queue_depth);
        init_ring(block_ring, 4); // block_read keeps at most one request in flight
    }
    ~custom_async_io(){
        io_uring_queue_exit(&block_ring);
        io_uring_queue_exit(&ring);
        close(fd);
    }

    //func
    void block_read(char * buf, size_t size, size_t offset);
    bool async_submit(uint64_t idx,char * buf, size_t size, size_t offset);
    std::vector<std::pair<uint32_t, int>> refresh_completions();

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
    std::vector<std::array<ggml_type, NUM_EXPERT>>  layer_type;  //up gate down
    std::vector<std::array<size_t, NUM_EXPERT>>     layer_nbyte; //bytes of one expert in the file, up gate down
    
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
        auto completions = async_io.refresh_completions();
        for (auto [data_id, n_read] : completions) {
            //get il and expert_id from id
            auto [layer_id, expert_id,type] = unmap(data_id);
            if ((size_t) n_read != layer_nbyte[layer_id][static_cast<int>(type)]) {
                throw std::runtime_error("async I/O error: short read");
            }
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


// standard LRU over all experts held in the pool (any layer)
// front = most recently used, back = least recently used
class custom_expert_manage {
private:
    using lru_key = std::pair<uint32_t, int32_t>; // {moe layer, expert id}

    uint32_t                                            n_layer;
    int32_t                                             n_expert;
    std::list<lru_key>                                  lru;
    // iters[il][id] points into lru, lru.end() = not tracked (std::list keeps end() valid)
    std::vector<std::vector<std::list<lru_key>::iterator>> iters;
    std::mutex                                          mutex; // use() is called by all threads in prefill

    void check(uint32_t il, int32_t id) const {
        GGML_ASSERT(il < n_layer && 0 <= id && id < n_expert && "expert_manage index out of bounds");
    }

public:
    custom_expert_manage(uint32_t n_layer, int32_t n_expert)
        : n_layer(n_layer),
          n_expert(n_expert),
          iters(n_layer, std::vector<std::list<lru_key>::iterator>(n_expert, lru.end())) {}

    // the expert got a slot in the pool (loading or loaded)
    void insert(uint32_t il, int32_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        check(il, id);
        GGML_ASSERT(iters[il][id] == lru.end() && "expert already in lru");
        lru.push_front({il, id});
        iters[il][id] = lru.begin();
    }

    void insert_layer(uint32_t il) {
        for (int32_t id = 0; id < n_expert; id++) {
            insert(il, id);
        }
    }

    // the expert is used by the current token
    void use(uint32_t il, int32_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        check(il, id);
        auto it = iters[il][id];
        GGML_ASSERT(it != lru.end() && "used expert is not in the pool");
        lru.splice(lru.begin(), lru, it);
    }

    // the expert left the pool
    void erase(uint32_t il, int32_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        check(il, id);
        auto it = iters[il][id];
        if (it != lru.end()) {
            lru.erase(it);
            iters[il][id] = lru.end();
        }
    }

    void erase_layer(uint32_t il) {
        for (int32_t id = 0; id < n_expert; id++) {
            erase(il, id);
        }
    }

    // least recently used expert accepted by can_evict(il, id), return false if there is none
    template <typename F>
    bool victim(F && can_evict, uint32_t & il, int32_t & id) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto rit = lru.rbegin(); rit != lru.rend(); ++rit) {
            if (can_evict(rit->first, rit->second)) {
                il = rit->first;
                id = rit->second;
                return true;
            }
        }
        return false;
    }
};

class custom_moe_unified;

struct runtime_ctx{
    custom_moe_unified  * moe_unified;
    int   il;
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
    size_t                                  total_size() const;
    void                                    prefill_init();//pre-load the initial some layer of moe_weight
    void                                    check_layer(uint32_t il);
    void                                    check_expert(uint32_t il,uint32_t id);
    bool                                    predict_expert(uint32_t il,uint32_t id);
    void                                    predict_layer(uint32_t head_il,uint32_t tail_il);
    llama_pos                               id_map(uint32_t il, int32_t selec_id);
    ggml_tensor *                           get_ups(struct ggml_context * ctx, uint32_t il) const;
    ggml_tensor *                           get_gates(struct ggml_context * ctx, uint32_t il) const;
    ggml_tensor *                           get_downs(struct ggml_context * ctx, uint32_t il) const;
    // userdata of the custom ops; owned here because a reused graph runs the same op many times
    runtime_ctx *                           get_runtime_ctx(uint32_t il);

private:
    const llama_model                       &model;
    //menbers
    // size_t: slot offsets (pos * nbyte_slot) can exceed 4 GiB
    // slot size = the largest expert over all moe layers, smaller experts leave a gap at the slot end
    size_t                                  nbyte_slot_up;
    size_t                                  nbyte_slot_gate;
    size_t                                  nbyte_slot_down;
    uint32_t                                n_slots_layer;    // number of slots in one layer
    struct  custom_pool_type                pool_type;
    struct  custom_expert_pool              pool;
    class   custom_async_io                 async_io;       
    class   custom_expert_table             table;    
    std::vector<ggml_context_ptr>           ctxs;
    std::vector<ggml_backend_buffer_ptr>    bufs;
    std::vector<runtime_ctx>                rt_ctxs;    // one per moe layer, never resized
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
    // an expert can be evicted if it is fully loaded and not in the layer being computed (cur_il)
    bool        can_evict(uint32_t il, int32_t id, uint32_t cur_il);
    llama_pos   free_random(uint32_t cur_il);
    llama_pos   free_lru(uint32_t cur_il);

    //
    // pool API
    // 
    llama_pos   pool_alloc(int32_t n);
    void        pool_free(llama_pos pos, int32_t n);
};

//
//custom op_kernel
//
void id_pos_map(struct ggml_tensor * dst , const struct ggml_tensor * a,const struct ggml_tensor * b, int ith, int nth,  void * userdata);

void custom_predict(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth,  void * userdata);