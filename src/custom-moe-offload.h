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
#include <functional>
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
    float           utilization       = 0.7f;
    uint32_t        io_queue          = 0;  // io-aware prefill: predict a new layer only while fewer experts are loading
    uint32_t        decode_prefetch   = 0;  // io-aware decode: max experts prefetched per layer, 0 = every missing expert
    uint32_t        n_preload_layers  = 0;  // whole layers that fit in the pool, loaded at model load
    uint32_t        reserve_budget    = 0;  // max predicted (reserved) experts: n_slots - n_expert, keeps room for the current layer
};

//
// strategy: each llama_moe_mode is a combination of independent components, see strategy_for()
// a new method is a new value of one component; changing one mode does not touch the others
//
enum class custom_evict_kind   { random, lru, arpe };
enum class custom_victim_kind  { policy, farthest_layer };
enum class custom_prefill_kind { none, cross_layer, io_aware };
enum class custom_decode_kind  { none, next_layer };

struct custom_strategy {
    io_mode             load;           // block: one expert at a time, async: all missing experts of a layer in parallel
    custom_evict_kind   evict;          // decode victims (and prefill victims with prefill_victim == policy)
    custom_victim_kind  prefill_victim; // farthest_layer: the layer whose next use is the latest, see victim_farthest()
    custom_prefill_kind prefill;
    custom_decode_kind  decode;
    uint32_t            decode_load;    // decode next_layer: max experts prefetched per layer, 0 = every missing expert
};

custom_strategy custom_strategy_for(enum llama_moe_mode mode, uint32_t decode_prefetch);
const char *    custom_moe_mode_name(enum llama_moe_mode mode);

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
    bool predicted; // loaded for a layer that is not computed yet: not evictable until that layer runs
    bool prefetched; // read by a predictor, cleared when its layer runs (stats only)
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

    // cq_entries = 0: kernel default (2 x queue_depth)
    void init_ring(io_uring & r, unsigned queue_depth, unsigned cq_entries = 0){
        io_uring_params params{};
        if (cq_entries > 0) {
            params.flags      = IORING_SETUP_CQSIZE | IORING_SETUP_CLAMP;
            params.cq_entries = cq_entries;
        }
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
        // kring_entries: the only entry count in liburing < 2.2 (2.5 deprecates it in favor of sq.ring_entries)
        if (mlock(r.sq.sqes, (size_t) *r.sq.kring_entries * sizeof(struct io_uring_sqe))) {
            perror("mlock sqes failed");
        }
    }

    public:

    custom_async_io(const std::string & fname,int queue_depth = 640){
        fd = open(fname.c_str(), O_RDONLY);// note: O_DIRECT is currently not be support
        if (fd < 0) {
            throw std::runtime_error("Failed to open file: " + std::string(strerror(errno)));
        }
        // every read is submitted at once (no SQ backlog), but completions are reaped late:
        // the CQ must hold all reads in flight, up to 3 per pool slot
        init_ring(ring, queue_depth, 32768);
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
    // wait = true: sleep until at least one request completes (the caller must have one in flight)
    std::vector<std::pair<uint32_t, int>> refresh_completions(bool wait = false);

};


class custom_expert_table{
private:
    uint32_t n_row;
    int32_t n_col;
    std::vector<std::vector<uint8_t>>               pending_io; // async reads left per expert (up gate down)
    uint32_t                                        n_loading = 0;  // experts with async reads in flight
    uint32_t                                        n_reserved = 0; // experts with predicted == true
    std::vector<int32_t>                            n_resident;     // experts per layer with a slot (Loading or InMemory)
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
        const bool was_resident = experts[row][col].state != expert_state::OnDisk;
        const bool is_resident  = state != expert_state::OnDisk;
        n_resident[row] += (int32_t) is_resident - (int32_t) was_resident;
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
        for(int32_t i =0; i < n_col; i++){
            mark(row, i, state, state == expert_state::Loading ? head_pos + i : INVALID_POS);
        }
    }

    // the 3 async reads of the expert are submitted
    void mark_async(uint32_t row, int32_t col){
        check_indices(row,col);
        GGML_ASSERT(experts[row][col].state == expert_state::Loading && pending_io[row][col] == 0);
        pending_io[row][col] = NUM_EXPERT;
        n_loading++;
    }

    uint32_t loading_count() const {
        return n_loading;
    }

    int32_t resident_count(uint32_t row) const {
        check_indices(row,0);
        return n_resident[row];
    }

    // reserve the expert for a prediction: not evictable until its layer runs
    void reserve(uint32_t row, int32_t col){
        check_indices(row,col);
        if(!experts[row][col].predicted){
            experts[row][col].predicted = true;
            n_reserved++;
        }
    }

    uint32_t reserved_count() const {
        return n_reserved;
    }

    // the layer runs now, its predictions are consumed (used or not)
    void clear_predicted(uint32_t row){
        check_indices(row,0);
        for(int32_t i = 0; i < n_col; i++){
            if(experts[row][i].predicted){
                experts[row][i].predicted = false;
                n_reserved--;
            }
        }
    }

    // a new ubatch starts: a prediction of a layer that never ran (an empty last layer) is stale
    void clear_all_predicted(){
        for(uint32_t row = 0; row < n_row; row++){
            clear_predicted(row);
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

    // wait = true: sleep until at least one async read completes, requires loading_count() > 0
    void refresh(class custom_async_io& async_io, bool wait = false){
        GGML_ASSERT(!wait || n_loading > 0);
        auto completions = async_io.refresh_completions(wait);
        for (auto [data_id, n_read] : completions) {
            //get il and expert_id from id
            auto [layer_id, expert_id,type] = unmap(data_id);
            if ((size_t) n_read != layer_nbyte[layer_id][static_cast<int>(type)]) {
                throw std::runtime_error("async I/O error: short read");
            }
            GGML_ASSERT(pending_io[layer_id][expert_id] > 0);
            if (--pending_io[layer_id][expert_id] == 0) {
                // all 3 parts are in the pool
                GGML_ASSERT(experts[layer_id][expert_id].state == expert_state::Loading);
                mark(layer_id,expert_id,expert_state::InMemory,INVALID_POS);
                n_loading--;
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


//
// eviction policy: tracks the experts held in the pool (any layer, loading or loaded) and picks the victim
// the policies are independent, a new cache management method is a new subclass (see custom_strategy)
//
using custom_expert_key = std::pair<uint32_t, int32_t>; // {moe layer, expert id}
using custom_evictable  = std::function<bool(uint32_t, int32_t)>;

class custom_evict_policy {
public:
    virtual ~custom_evict_policy() = default;
    virtual void insert(uint32_t il, int32_t id) = 0; // the expert got a slot in the pool
    virtual void use   (uint32_t il, int32_t id) = 0; // the expert is used by a token
    virtual void erase (uint32_t il, int32_t id) = 0; // the expert left the pool
    // an expert accepted by can_evict(il, id), return false if there is none
    virtual bool victim(const custom_evictable & can_evict, uint32_t & il, int32_t & id) = 0;

    // optional hooks for policies that model the routing (ARPE), single threaded
    // layer il starts and uses ids (unique), called before its missing experts are loaded
    virtual void layer_begin(uint32_t /*il*/, const std::vector<int32_t> & /*ids*/, bool /*prefill*/) {}
    // routing of layer il, ids[r*stride + k] for token r, after its experts are in memory and before the predictors run
    virtual void observe(uint32_t /*il*/, const int32_t * /*ids*/, int64_t /*n_used*/, int64_t /*n_tokens*/, int64_t /*stride*/, bool /*prefill*/) {}
    // true: no expert of the current layer is evictable, false: only the experts it uses are locked
    virtual bool lock_layer() const { return true; }

    void insert_layer(uint32_t il, int32_t n_expert) {
        for (int32_t id = 0; id < n_expert; id++) {
            insert(il, id);
        }
    }
};

// standard LRU: front = most recently used, back = least recently used
class custom_evict_lru : public custom_evict_policy {
private:
    uint32_t                                            n_layer;
    int32_t                                             n_expert;
    std::list<custom_expert_key>                        lru;
    // iters[il][id] points into lru, lru.end() = not tracked (std::list keeps end() valid)
    std::vector<std::vector<std::list<custom_expert_key>::iterator>> iters;
    std::mutex                                          mutex; // use() is called by all threads in prefill

    void check(uint32_t il, int32_t id) const {
        GGML_ASSERT(il < n_layer && 0 <= id && id < n_expert && "evict_lru index out of bounds");
    }

public:
    custom_evict_lru(uint32_t n_layer, int32_t n_expert)
        : n_layer(n_layer),
          n_expert(n_expert),
          iters(n_layer, std::vector<std::list<custom_expert_key>::iterator>(n_expert, lru.end())) {}

    void insert(uint32_t il, int32_t id) override {
        std::lock_guard<std::mutex> lock(mutex);
        check(il, id);
        GGML_ASSERT(iters[il][id] == lru.end() && "expert already in lru");
        lru.push_front({il, id});
        iters[il][id] = lru.begin();
    }

    void use(uint32_t il, int32_t id) override {
        std::lock_guard<std::mutex> lock(mutex);
        check(il, id);
        auto it = iters[il][id];
        GGML_ASSERT(it != lru.end() && "used expert is not in the pool");
        lru.splice(lru.begin(), lru, it);
    }

    void erase(uint32_t il, int32_t id) override {
        std::lock_guard<std::mutex> lock(mutex);
        check(il, id);
        auto it = iters[il][id];
        if (it != lru.end()) {
            lru.erase(it);
            iters[il][id] = lru.end();
        }
    }

    // the least recently used one
    bool victim(const custom_evictable & can_evict, uint32_t & il, int32_t & id) override {
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

// random eviction among the evictable experts
class custom_evict_random : public custom_evict_policy {
private:
    uint32_t                                n_layer;
    int32_t                                 n_expert;
    std::vector<custom_expert_key>          keys;
    std::vector<std::vector<int32_t>>       index; // index[il][id] into keys, -1 = not tracked
    std::mt19937                            gen{std::random_device{}()};
    std::mutex                              mutex;

    void check(uint32_t il, int32_t id) const {
        GGML_ASSERT(il < n_layer && 0 <= id && id < n_expert && "evict_random index out of bounds");
    }

public:
    custom_evict_random(uint32_t n_layer, int32_t n_expert)
        : n_layer(n_layer),
          n_expert(n_expert),
          index(n_layer, std::vector<int32_t>(n_expert, -1)) {}

    void insert(uint32_t il, int32_t id) override {
        std::lock_guard<std::mutex> lock(mutex);
        check(il, id);
        GGML_ASSERT(index[il][id] == -1 && "expert already tracked");
        index[il][id] = (int32_t) keys.size();
        keys.push_back({il, id});
    }

    void use(uint32_t il, int32_t id) override {
        check(il, id);
    }

    void erase(uint32_t il, int32_t id) override {
        std::lock_guard<std::mutex> lock(mutex);
        check(il, id);
        const int32_t i = index[il][id];
        if (i == -1) {
            return;
        }
        // swap with the last key, then pop
        keys[i] = keys.back();
        index[keys[i].first][keys[i].second] = i;
        keys.pop_back();
        index[il][id] = -1;
    }

    // scan from a random start, so every evictable expert can be picked
    bool victim(const custom_evictable & can_evict, uint32_t & il, int32_t & id) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (keys.empty()) {
            return false;
        }
        const size_t start = std::uniform_int_distribution<size_t>(0, keys.size() - 1)(gen);
        for (size_t k = 0; k < keys.size(); k++) {
            const auto & key = keys[(start + k) % keys.size()];
            if (can_evict(key.first, key.second)) {
                il = key.first;
                id = key.second;
                return true;
            }
        }
        return false;
    }
};

class custom_moe_unified;

// counters per phase (prefill / decode), printed when the model is freed
struct custom_moe_stats {
    uint64_t n_ensure        = 0; // layer runs
    uint64_t n_hit           = 0; // used experts already in memory
    uint64_t n_wait          = 0; // used experts still loading (prefetched, read not finished)
    uint64_t n_miss          = 0; // used experts read on demand
    uint64_t n_prefetch      = 0; // experts read by a predictor
    uint64_t n_prefetch_used = 0; // ... and used when their layer ran
    uint64_t n_bytes         = 0; // bytes read
    int64_t  t_ensure_us     = 0; // time in ensure: all threads wait for the experts of the layer
    int64_t  t_predict_us    = 0; // time in the predictors: alloc, evict, submit
};

struct runtime_ctx{
    custom_moe_unified  * moe_unified;
    int   il;
};

class custom_moe_unified{
public:
    custom_moe_unified(const llama_model & model,const std::string & fname,llama_model_loader & ml,enum llama_moe_mode mode,float utilization,uint32_t io_queue,uint32_t decode_prefetch,
                       const std::string & arpe_affinity, const std::string & arpe_calib);
    ~custom_moe_unified();
    //menbers
    const enum llama_moe_mode                   mode;
    const custom_strategy                       strategy;       // components of mode
    std::unique_ptr<custom_evict_policy>        evict;          // from strategy.evict
    struct custom_hparams                       params;
    const llama_hparams                         &hparams;
    // moe layer whose experts are mapped right now, set by id_pos_map and protected from eviction
    uint32_t                                    cur_il = 0;
    //
    // custom_moe_unified  API
    //
    size_t                                  total_size() const;
    void                                    prefill_init();//pre-load the initial some layer of moe_weight
    // phase of the ubatch, decided at the first moe layer: the last layer of a prompt may see only the output tokens
    bool                                    ubatch_prefill(uint32_t il, int64_t n_tokens);
    bool                                    in_prefill() const { return cur_prefill; }
    // layer il starts: make the experts ids (unique) resident, returns when all are in memory
    void                                    ensure(uint32_t il, const std::vector<int32_t> & ids, bool prefill);
    // fixed moe layers to predict from the input of moe layer il, by strategy.prefill / strategy.decode
    std::vector<uint32_t>                   predict_targets(uint32_t il, bool prefill) const;
    // io-aware prefill: moe layer il predicts a layer chosen at run time (next_frontier), the graph selects its router by index
    bool                                    predicts_frontier(uint32_t il, bool prefill) const;
    // the layer to predict now (-1 = none), called by custom_frontier after the current layer is ensured
    int32_t                                 next_frontier();
    int32_t                                 frontier_target = -1; // result of the last next_frontier()
    // the experts ids (unique) are predicted for layer target_il
    void                                    on_predicted(uint32_t target_il, const std::vector<int32_t> & ids, bool prefill);
    ggml_tensor *                           router(uint32_t il) const;      // ffn_gate_inp of moe layer il
    ggml_tensor *                           router_bias(uint32_t il) const; // ffn_exp_probs_b of moe layer il, may be null
    // the routers (biases) of all moe layers in one tensor, row il * n_expert + id; io-aware only, biases may be null
    ggml_tensor *                           routers_all       = nullptr; // [n_embd, n_expert * n_moe_layer]
    ggml_tensor *                           router_biases_all = nullptr; // [1, n_expert * n_moe_layer]
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
    // io-aware prefill: the first moe layer whose experts are not predicted yet in this ubatch
    uint32_t                                frontier = 0;
    custom_moe_stats                        stats[2];       // 0: prefill, 1: decode
    bool                                    cur_prefill = false; // phase of the current ubatch, set by ubatch_prefill
    bool                                    in_load = true;      // prefill_init still loading, the reads are not counted per phase
    std::vector<uint8_t>                    cur_used;       // [n_expert] experts used by cur_il, set by ensure
    //
    //common
    //
    void        table_init(const llama_model & model,llama_model_loader & ml);
    void        routers_init();
    uint32_t    get_padding() const;
    uint32_t    moe_layer_count() const { return hparams.n_layer() - hparams.n_layer_dense_lead; }
    //
    void        _free_expert(uint32_t il,int32_t id);
    // on-demand prefill: the whole layer enters the pool, whole layers are released to make room
    void        ensure_layer(uint32_t il, const std::vector<int32_t> & ids);
    bool        victim_layer(uint32_t & il); // the fully released layer whose next use is the farthest
    void        free_layer(uint32_t il);     // release every resident expert of a layer, slots return to the pool
    void        load_expert(uint32_t il, int32_t id,llama_pos target_pos,io_mode mode);
    void        load_layer(uint32_t il, llama_pos target_pos);
    // an expert can be evicted if it is fully loaded, not in the layer being computed (cur_il) and not reserved by a prediction
    bool        can_evict(uint32_t il, int32_t id, uint32_t cur_il);
    // same, but of cur_il only the experts it uses are locked (policies with lock_layer() == false)
    bool        can_evict_unused(uint32_t il, int32_t id);
    // an evictable expert of a layer whose next use comes after target_il (cyclic layer order from cur_il), farthest first
    bool        victim_farthest(uint32_t target_il, uint32_t & il, int32_t & id);
    // a slot for an expert of target_il: a free slot, else evict one (prefill: strategy.prefill_victim, decode: evict policy)
    // wait = true waits for in-flight loads when nothing can be evicted yet
    llama_pos   alloc_slot(uint32_t target_il, bool wait, bool prefill);
    // reserve the expert of a later layer and start its async load if needed, false if the budget or the pool is full
    bool        reserve_and_load(uint32_t il, int32_t id, bool prefill);

    //
    // predictors (strategy.prefill / strategy.decode)
    //
    void        prefetch_cross_layer(uint32_t il, const std::vector<int32_t> & ids);
    void        prefetch_frontier(uint32_t il, const std::vector<int32_t> & ids);
    void        predict_next_layer(uint32_t il, const std::vector<int32_t> & ids);

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

// io-aware prefill: dst [n_expert] I32 = rows of the chosen layer in routers_all, then custom_predict_frontier uses the prediction
void custom_frontier(struct ggml_tensor * dst, int ith, int nth, void * userdata);
void custom_predict_frontier(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth,  void * userdata);