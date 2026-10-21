#include "custom-moe-offload.h"
#include "custom-moe-arpe.h"

#include "ggml-cpp.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <condition_variable>


// #include ""
std::string extra_copyname(const std::string& str) ;

void custom_async_io::block_read(char * buf, size_t size, size_t offset){
    // get sqe_handle
    struct io_uring_sqe* sqe = io_uring_get_sqe(&block_ring);
    if (!sqe) {
        throw std::runtime_error("Failed to get SQE");
    }

    io_uring_prep_read(sqe, 0, buf, size, offset);//0: the index of register file
    sqe->flags |= IOSQE_FIXED_FILE;

    // submit
    int ret = io_uring_submit(&block_ring);
    if (ret < 0) {
        throw std::runtime_error("custom_asyncio::Failed to submit request:" + std::string(strerror(-ret)));
    }

    // waiting for complete
    struct io_uring_cqe* cqe;
    ret = io_uring_wait_cqe(&block_ring, &cqe);
    if (ret < 0) {
        throw std::runtime_error("custom_asyncio::Failed to wait for completion:" + std::string(strerror(-ret)));
    }
    //clean cur cqe
    int read_bytes = cqe->res;
    io_uring_cqe_seen(&block_ring, cqe);
    if (read_bytes < 0) {
        throw std::runtime_error("custom_asyncio::Read failed in block reading: " + std::string(strerror(-read_bytes)));
    }
    if ((size_t) read_bytes != size) {
        throw std::runtime_error("custom_asyncio::Short read in block reading");
    }
}

bool custom_async_io::async_submit(uint64_t idx,char * buf, size_t size, size_t offset){

    //mul-thread lock
    std::lock_guard<std::mutex> lock(submit_mutex);
    // first
    struct io_uring_sqe* sqe  = io_uring_get_sqe(&ring);
    if (!sqe){
        throw std::runtime_error("async_submit: Failed to get SQE");
    }
    io_uring_prep_read(sqe, 0, buf, size, offset);//0: the index of register file
    // IOSQE_ASYNC: without it the kernel first tries the buffered read inline, so the page cache copy (and the
    // readahead retries, as task_work) run on the submitting compute thread and stall the whole graph.
    // with it a kernel io-wq worker does the read and the copy, the compute threads only reap the completion
    sqe->flags |= IOSQE_FIXED_FILE | IOSQE_ASYNC;
    sqe->user_data = idx;

    
    if (sqe->user_data == 0) {
        throw std::runtime_error(std::string("async_submit error: set idx_data of 0"));
    }
    int ret = io_uring_submit(&ring);
    if (ret < 0) {
        fprintf(stderr, "erro sqe submit\n");
        throw std::runtime_error("async_submit: Failed to sqe submit:" + std::string(strerror(-ret)));
        return false;
    }

    return true;
}

// ==========================================
    // 新增方法 1: 刷新函数 (非阻塞)
    // 检查并处理所有当前已完成的请求
    // ==========================================
std::vector<std::pair<uint32_t, int>> custom_async_io::refresh_completions(bool wait) {
    struct io_uring_cqe* cqe;
    std::vector<std::pair<uint32_t, int>> completed;

    if (wait) {
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) {
            throw std::runtime_error("custom_asyncio::Failed to wait for completion:" + std::string(strerror(-ret)));
        }
    }
    while (io_uring_peek_cqe(&ring, &cqe) == 0) {
        // 获取到了一个 cqe
        completed.push_back(process_cqe(cqe)); // 内部调用 seen，移动队列指针
    }
    return completed;
}



custom_expert_table::custom_expert_table(uint32_t n_moe_layer, uint32_t n_expert)
    :n_row(n_moe_layer),
     n_col(n_expert),
     pending_io(n_moe_layer, std::vector<uint8_t>(n_expert, 0)),
     n_resident(n_moe_layer, 0),
     experts(n_moe_layer,std::vector<custom_expert_group>(n_expert)),
     layer_type(n_moe_layer),
     layer_nbyte(n_moe_layer)
    {}



custom_strategy custom_strategy_for(enum llama_moe_mode mode, uint32_t decode_prefetch){
    switch (mode) {
        case LLAMA_MOE_MODE_ON_DEMAND:
            return { io_mode::block, custom_evict_kind::random, custom_victim_kind::policy,         custom_prefill_kind::none,        custom_decode_kind::none,       0 };
        case LLAMA_MOE_MODE_CROSS_LAYER:
            // decode prefetches every missing expert of the next layer
            return { io_mode::async, custom_evict_kind::lru,    custom_victim_kind::farthest_layer, custom_prefill_kind::cross_layer, custom_decode_kind::next_layer, 0 };
        case LLAMA_MOE_MODE_IO_AWARE:
            // the decode predictor reuses the cross-layer component, but only prefetches what one layer can cover
            return { io_mode::async, custom_evict_kind::arpe,   custom_victim_kind::farthest_layer, custom_prefill_kind::io_aware,    custom_decode_kind::next_layer, decode_prefetch };
    }
    GGML_ABORT("custom moe: unknown mode %d", (int) mode);
}

const char * custom_moe_mode_name(enum llama_moe_mode mode){
    switch (mode) {
        case LLAMA_MOE_MODE_ON_DEMAND:   return "on-demand";
        case LLAMA_MOE_MODE_CROSS_LAYER: return "cross-layer";
        case LLAMA_MOE_MODE_IO_AWARE:    return "io-aware";
    }
    return "unknown";
}

custom_moe_unified::custom_moe_unified(const llama_model & model,const std::string & fname,llama_model_loader & ml,enum llama_moe_mode mode,float utilization,uint32_t io_queue,uint32_t decode_prefetch,
                                       const std::string & arpe_affinity, const std::string & arpe_calib)
    : mode(mode),
      strategy(custom_strategy_for(mode, decode_prefetch)),
      params{utilization,io_queue,decode_prefetch,0,0},
      hparams(model.hparams),
      model(model),
      async_io(fname),
      table(hparams.n_layer() - hparams.n_layer_dense_lead, hparams.n_expert){


    uint32_t n_expert = hparams.n_expert;
    // the slot of each part (up gate down) must fit the largest expert of all moe layers,
    // so the pool type comes from the layer with the largest tensor
    ggml_tensor * classic_up   = nullptr;
    ggml_tensor * classic_gate = nullptr;
    ggml_tensor * classic_down = nullptr;
    auto pick_largest = [](ggml_tensor *& classic, ggml_tensor * cur) {
        GGML_ASSERT(cur != nullptr && "custom moe: layer without separate up/gate/down expert tensors");
        if (classic != nullptr) {
            GGML_ASSERT(classic->ne[0] == cur->ne[0] && classic->ne[1] == cur->ne[1] && classic->ne[2] == cur->ne[2]);
        }
        if (classic == nullptr || ggml_nbytes(cur) > ggml_nbytes(classic)) {
            classic = cur;
        }
    };
    for (uint32_t il = hparams.n_layer_dense_lead; il < hparams.n_layer(); il++) {
        pick_largest(classic_up,   model.layers.at(il).ffn_up_exps);
        pick_largest(classic_gate, model.layers.at(il).ffn_gate_exps);
        pick_largest(classic_down, model.layers.at(il).ffn_down_exps);
    }

    this->pool_type = {
        classic_up->type,
        classic_gate->type,
        classic_down->type
    };

    this->nbyte_slot_up     = ggml_nbytes(classic_up)   / n_expert;
    this->nbyte_slot_gate   = ggml_nbytes(classic_gate) / n_expert;
    this->nbyte_slot_down   = ggml_nbytes(classic_down) / n_expert;
    this->n_slots_layer     = n_expert;

    //init the table
    table_init(model,ml);

    //init the moe_pool
    // the pool is read by the device that runs the expert matmuls, so it lives in device memory. io_uring cannot
    // write device memory: the experts are read into a host staging ring first and then copied into the pool
    // (copy_expert_to_pool). the iGPU must not read host mapped weights directly: ggml-cuda only handles that on
    // the MMVQ path (batch <= 8), larger batches are corrupted (feature.md 6.5)
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    // model.devices also holds an iGPU (e.g. Jetson), ggml_backend_dev_by_type(GPU) does not find it
    ggml_backend_dev_t dev = model.devices.empty() ? nullptr : model.devices.front().dev;

    if (!dev) {
        // FIXME: workaround for CPU backend buft having a NULL device
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
    }
    ggml_backend_buffer_type_t pool_buft = ggml_backend_dev_buffer_type(dev);
    if (!pool_buft) {
        pool_buft = cpu_buft;
    }
    // the CPU writes the staging buffer (io_uring, copies): it has to be host memory
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (!host_buft) {
        host_buft = cpu_buft;
    }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);

    //get the number of slots in pool
    size_t nbyte_group = nbyte_slot_up + nbyte_slot_gate + nbyte_slot_down;
    uint64_t n_byte_availa = (uint64_t)(utilization * props.memory_total);
    int32_t n_slots  = n_byte_availa / nbyte_group;
    // prefill keeps whole layers in the pool, so at least one layer must fit
    if (n_slots < (int32_t) n_expert) {
        throw std::runtime_error(format("%s: moe pool too small: %d slots for %u experts per layer, increase the memory utilization",
            __func__, n_slots, n_expert));
    }


// create a context for  buffer 
    ggml_init_params buffer_params = {
        /*.mem_size   =*/ size_t(5 *ggml_tensor_overhead()), //up gate down + routers_all, router_biases_all
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(buffer_params);
        if (!ctx) {
        throw std::runtime_error(format("%s: moe_pool_unified ctx alloc failed", __func__));
    }
    ctxs.emplace_back(ctx);

    //create moe_pool_tensors
    ggml_tensor * ups;
    ggml_tensor * gates;
    ggml_tensor * downs;
    //memory alignment assert
    const auto n_pad = get_padding();
    GGML_ASSERT(classic_up->nb[2] % n_pad == 0);
    GGML_ASSERT(classic_gate->nb[2] % n_pad == 0);
    GGML_ASSERT(classic_down->nb[2] % n_pad == 0);
    
    ups = ggml_new_tensor_3d(ctx, pool_type.up, classic_up->ne[0], classic_up->ne[1],n_slots);
    gates = ggml_new_tensor_3d(ctx, pool_type.gate, classic_gate->ne[0],classic_gate->ne[1],n_slots);
    downs = ggml_new_tensor_3d(ctx, pool_type.down, classic_down->ne[0],classic_down->ne[1],n_slots);
    ggml_format_name(ups,   "pool_ups");
    ggml_format_name(gates, "pool_gates");
    ggml_format_name(downs, "pool_downs");
    pool = {
        /*tensor_up         =*/ ups,
        /*tensor_gate       =*/ gates,
        /*tensor_down       =*/ downs,
        /*n_slots           =*/ n_slots,
        /*free_slots        =*/ {{0,n_slots}}
    };

    // io-aware prefill picks the predicted layer at run time: the graph selects its router rows with get_rows
    if (strategy.prefill == custom_prefill_kind::io_aware) {
        ggml_tensor * r0 = router(0);
        routers_all = ggml_new_tensor_2d(ctx, r0->type, r0->ne[0], r0->ne[1] * moe_layer_count());
        ggml_format_name(routers_all, "routers_all");
        if (router_bias(0) != nullptr) {
            router_biases_all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, r0->ne[1] * moe_layer_count());
            ggml_format_name(router_biases_all, "router_biases_all");
        }
    }


    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, pool_buft);
    if (!buf) {
        throw std::runtime_error("failed to allocate buffer for moe_pool");
    }
    // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bufs.emplace_back(buf);

    // staging ring: one slot per load in flight. n_stage bounds the number of parallel reads
    nbyte_stage_slot = nbyte_slot_up + nbyte_slot_gate + nbyte_slot_down;
    n_stage = 16;
    staging_buf.reset(ggml_backend_buft_alloc_buffer(host_buft, (size_t) n_stage * nbyte_stage_slot));
    if (!staging_buf) {
        throw std::runtime_error("failed to allocate the staging buffer for moe_pool");
    }
    stage_free.resize(n_stage);
    for (int32_t s = 0; s < n_stage; s++) {
        stage_free[s] = s;
    }
    table.on_loaded = [this](uint32_t il, int32_t id) {
        auto & e = table.at(il, id);
        GGML_ASSERT(e.stage >= 0);
        copy_expert_to_pool(il, id, e.stage);
        stage_release(e.stage);
        e.stage = -1;
    };

    routers_init();

    LLAMA_LOG_INFO("n_slot of pool is:%d \n",n_slots);
    LLAMA_LOG_INFO("%s: %10s moe_expert_pool buffer size = %8.2f MiB, staging %d x %.2f MiB in %s memory\n", __func__,
        ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0,
        n_stage, nbyte_stage_slot/1024.0/1024.0, ggml_backend_buft_name(host_buft));

    // the pre-load layer count is bounded by the model: a larger pool keeps every layer resident
    const uint32_t n_moe_layer = hparams.n_layer() - hparams.n_layer_dense_lead;

    switch (strategy.evict) {
        case custom_evict_kind::random: evict = std::make_unique<custom_evict_random>(n_moe_layer, (int32_t) n_expert); break;
        case custom_evict_kind::lru:    evict = std::make_unique<custom_evict_lru>   (n_moe_layer, (int32_t) n_expert); break;
        case custom_evict_kind::arpe:
            {
                // ARPE models a fixed top-k
                const uint32_t n_used = hparams.n_expert_used(hparams.n_layer_dense_lead);
                for (uint32_t il = hparams.n_layer_dense_lead; il < hparams.n_layer(); il++) {
                    GGML_ASSERT(hparams.n_expert_used(il) == n_used && "custom moe: ARPE needs the same top-k in every moe layer");
                }
                evict = std::make_unique<custom_evict_arpe>(n_moe_layer, (int32_t) n_expert, (int32_t) n_used, arpe_affinity, arpe_calib);
            } break;
    }
    cur_used.assign(n_expert, 0);

    params.n_preload_layers = std::min<uint32_t>(n_slots /n_slots_layer, n_moe_layer);
    // predictions may hold all slots but one layer, so the current layer always finds room (no deadlock)
    params.reserve_budget   = n_slots - n_expert;
    if (params.io_queue == 0) {
        params.io_queue = n_expert;
    }

    rt_ctxs.resize(n_moe_layer);
    for (uint32_t il = 0; il < n_moe_layer; il++) {
        rt_ctxs[il] = { this, (int) il };
    }
}

custom_moe_unified::~custom_moe_unified() {
    const char * names[2] = { "prefill", "decode" };
    for (int i = 0; i < 2; i++) {
        const auto & s = stats[i];
        if (s.n_ensure == 0) {
            continue;
        }
        LLAMA_LOG_INFO("custom_moe stats [%s]: %llu layer runs, used experts: %llu hit, %llu wait, %llu miss | prefetch %llu (used %llu) | read %.1f MiB | ensure %.1f ms, predict %.1f ms\n",
            names[i], (unsigned long long) s.n_ensure,
            (unsigned long long) s.n_hit, (unsigned long long) s.n_wait, (unsigned long long) s.n_miss,
            (unsigned long long) s.n_prefetch, (unsigned long long) s.n_prefetch_used,
            s.n_bytes / 1024.0 / 1024.0, s.t_ensure_us / 1000.0, s.t_predict_us / 1000.0);
    }
}

runtime_ctx * custom_moe_unified::get_runtime_ctx(uint32_t il) {
    GGML_ASSERT(il < rt_ctxs.size());
    return &rt_ctxs[il];
}

size_t custom_moe_unified::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

void custom_moe_unified::prefill_init(){

    for(uint32_t il = 0; il < params.n_preload_layers; il++){
        llama_pos slot_pos = pool_alloc(n_slots_layer);
        load_layer(il,slot_pos);
    }
    in_load = false;

    LLAMA_LOG_INFO("custom_moe: mode = %s, %u layers pre-loaded, reserve budget = %u experts",
        custom_moe_mode_name(mode), params.n_preload_layers, params.reserve_budget);
    if (strategy.prefill == custom_prefill_kind::io_aware) {
        LLAMA_LOG_CONT(", io queue = %u experts", params.io_queue);
    }
    if (strategy.decode == custom_decode_kind::next_layer) {
        LLAMA_LOG_CONT(", decode prefetch = %u experts/layer", strategy.decode_load);
    }
    LLAMA_LOG_CONT("\n");
}

bool custom_moe_unified::ubatch_prefill(uint32_t il, int64_t n_tokens){
    if(il == 0){
        cur_prefill = n_tokens > 1;
    }
    return cur_prefill;
}

bool custom_moe_unified::victim_farthest(uint32_t target_il, uint32_t & il, int32_t & id){
    // prefill visits the layers in order, so the next use of layer l is (l - cur_il) mod n layers away:
    // the layers finished in this ubatch come last (next ubatch or decode), the layer just finished is the farthest
    // only layers used after target_il may be evicted, a prefetch never evicts experts needed sooner than its own
    const uint32_t n_layer  = moe_layer_count();
    const uint32_t d_target = (target_il + n_layer - cur_il) % n_layer;
    for(uint32_t d = n_layer - 1; d > d_target; d--){
        const uint32_t l = (cur_il + d) % n_layer;
        if(table.resident_count(l) == 0){
            continue;
        }
        for(int32_t e = 0; e < (int32_t) hparams.n_expert; e++){
            if(can_evict(l,e,cur_il)){
                il = l;
                id = e;
                return true;
            }
        }
    }
    return false;
}

llama_pos custom_moe_unified::alloc_slot(uint32_t target_il, bool wait, bool prefill){
    const bool farthest = prefill && strategy.prefill_victim == custom_victim_kind::farthest_layer;
    while(1){
        llama_pos pos = pool_alloc(1);
        if(pos == -1){
            uint32_t il = 0;
            int32_t  id = 0;
            const bool found = farthest ? victim_farthest(target_il, il, id)
                                        : evict->victim([&](uint32_t l, int32_t e){
                                              return evict->lock_layer() ? can_evict(l,e,cur_il) : can_evict_unused(l,e);
                                          }, il, id);
            if(found){
                pos = table.at(il,id).pos;
                _free_expert(il,id);
            }
        }
        if(pos != -1 || !wait){
            return pos;
        }
        // every expert that could be evicted is still loading: wait for a read to finish
        GGML_ASSERT(table.loading_count() > 0 && "custom moe: no expert can be evicted");
        table.refresh(async_io, true);
    }
}

// the layer whose next use is the farthest (in a prefill that is the layer just finished), if all its experts may go
bool custom_moe_unified::victim_layer(uint32_t & il){
    const uint32_t n_layer = moe_layer_count();
    for(uint32_t d = n_layer - 1; d > 0; d--){
        const uint32_t l = (cur_il + d) % n_layer;
        if(table.resident_count(l) == 0){
            continue;
        }
        bool ok = true;
        for(int32_t e = 0; e < (int32_t) hparams.n_expert && ok; e++){
            const auto & x = table.at(l,e);
            ok = x.state != expert_state::Loading && !x.predicted;
        }
        if(ok){
            il = l;
            return true;
        }
    }
    return false;
}

// release a whole layer: every slot it holds goes back to the pool, adjacent slots merge into a run
void custom_moe_unified::free_layer(uint32_t il){
    for(int32_t e = 0; e < (int32_t) hparams.n_expert; e++){
        auto & x = table.at(il,e);
        if(x.state == expert_state::InMemory){
            const llama_pos pos = x.pos;
            _free_expert(il,e);
            pool_free(pos, 1);
        }
    }
}

// on-demand prefill: the layer enters the pool as a whole, and whole layers are released to make room
void custom_moe_unified::ensure_layer(uint32_t il, const std::vector<int32_t> & ids){
    if(table.resident_count(il) == (int32_t) hparams.n_expert){
        return;
    }
    // reload from a clean layer: experts of il that still sit in slots (from a decode) give them back
    free_layer(il);
    llama_pos pos = pool_alloc(n_slots_layer);
    while(pos < 0){
        uint32_t victim = 0;
        if(!victim_layer(victim)){
            break;
        }
        LLAMA_LOG_DEBUG("custom_moe: release layer %u for layer %u\n", victim, il);
        free_layer(victim);
        pos = pool_alloc(n_slots_layer);
    }
    if(pos >= 0){
        load_layer(il, pos);
        return;
    }
    // the free slots do not form a whole layer (a decode leaves scattered slots): load the experts in use one by one
    LLAMA_LOG_WARN("%s: no room for a whole layer, layer %u is loaded expert by expert\n", __func__, il);
    for(int32_t id : ids){
        if(table.at(il,id).state == expert_state::OnDisk){
            load_expert(il,id,alloc_slot(il, true, true),io_mode::block);
        }
    }
}

void custom_moe_unified::ensure(uint32_t il, const std::vector<int32_t> & ids, bool prefill){
    const int64_t t_start = ggml_time_us();
    auto & st = stats[prefill ? 0 : 1];
    st.n_ensure++;

    // layer il runs now: its predictions are consumed (used or not) and its experts are protected
    cur_il = il;
    std::fill(cur_used.begin(), cur_used.end(), 0);
    for(int32_t id : ids){
        cur_used[id] = 1;
    }
    table.clear_predicted(il);
    if(prefill && il == 0){
        // a new ubatch: no layer is predicted yet, a reservation left by the previous ubatch is stale
        frontier = 0;
        table.clear_all_predicted();
    }
    evict->layer_begin(il, ids, prefill);

    table.refresh(async_io);
    for(int32_t id : ids){
        auto & e = table.at(il,id);
        switch(e.state){
            case expert_state::InMemory: st.n_hit++;  break;
            case expert_state::Loading:  st.n_wait++; break;
            case expert_state::OnDisk:   st.n_miss++; break;
        }
        if(e.prefetched){
            st.n_prefetch_used++;
        }
    }
    for(int32_t id = 0; id < (int32_t) hparams.n_expert; id++){
        table.at(il,id).prefetched = false;
    }
    if(prefill && strategy.load == io_mode::block){
        // on-demand prefill works on whole layers (see ensure_layer)
        ensure_layer(il, ids);
    } else if(strategy.load == io_mode::block){
        for(int32_t id : ids){
            if(table.at(il,id).state == expert_state::OnDisk){
                load_expert(il,id,alloc_slot(il, true, prefill),io_mode::block);
            }
        }
    }else{
        // submit all missing experts first, so the reads run in parallel
        for(int32_t id : ids){
            if(table.at(il,id).state == expert_state::OnDisk){
                // a staging slot is needed for the read; the layers in use have priority over prefetches
                while(stage_free.empty()){
                    table.refresh(async_io, true);
                }
                load_expert(il,id,alloc_slot(il, true, prefill),io_mode::async);
            }
        }
    }
    // predicted experts may still be loading
    for(int32_t id : ids){
        while(table.at(il,id).state != expert_state::InMemory){
            table.refresh(async_io, true);
        }
    }
    st.t_ensure_us += ggml_time_us() - t_start;
}

bool custom_moe_unified::reserve_and_load(uint32_t il, int32_t id, bool prefill){
    auto & e = table.at(il,id);
    if(e.predicted){
        return true;
    }
    if(table.reserved_count() >= params.reserve_budget){
        return false;
    }
    if(e.state == expert_state::OnDisk){
        if(stage_free.empty()){
            return false; // no staging slot: a prefetch must not block the reads of the layers in use
        }
        llama_pos pos = alloc_slot(il, false, prefill);
        if(pos == -1){
            return false;
        }
        load_expert(il,id,pos,io_mode::async);
        e.prefetched = true;
        stats[prefill ? 0 : 1].n_prefetch++;
    }
    table.reserve(il,id);
    return true;
}

// cross-layer prefill: every predicted expert of the next layer il, no wait; what does not fit is loaded when the layer runs
void custom_moe_unified::prefetch_cross_layer(uint32_t il, const std::vector<int32_t> & ids){
    for(int32_t id : ids){
        reserve_and_load(il,id,true);
    }
}

// io-aware prefill: every predicted expert of the frontier layer il
void custom_moe_unified::prefetch_frontier(uint32_t il, const std::vector<int32_t> & ids){
    bool done = true;
    for(int32_t id : ids){
        done = reserve_and_load(il,id,true) && done;
    }
    // a layer that did not fit stays the frontier: the next layer predicts it again with a fresher input
    if(done){
        frontier = il + 1;
    }
}

// decode: prefetch the predicted experts of layer il that are not in the pool, up to strategy.decode_load (0 = all)
void custom_moe_unified::predict_next_layer(uint32_t il, const std::vector<int32_t> & ids){
    uint32_t n_load = 0;
    for(int32_t id : ids){
        if(table.at(il,id).state != expert_state::OnDisk){
            continue;
        }
        if(strategy.decode_load > 0 && n_load >= strategy.decode_load){
            break;
        }
        if(!reserve_and_load(il,id,false)){
            break;
        }
        n_load++;
    }
}

int32_t custom_moe_unified::next_frontier(){
    const int64_t t_start = ggml_time_us();
    table.refresh(async_io);
    const uint32_t n_layer = moe_layer_count();
    frontier = std::max(frontier, cur_il + 1);
    // a layer with all experts in the pool has nothing to load
    while(frontier < n_layer && table.resident_count(frontier) == (int32_t) hparams.n_expert){
        frontier++;
    }
    // IO is the slower side: while the queue is long, a prediction now would only wait in it,
    // so skip this layer and let a later layer predict the frontier with a fresher input.
    // the next layer is needed right away, it is predicted whatever the queue
    frontier_target = -1;
    // the last moe layer only computes the output tokens and would need a prediction of its own rows:
    // it is not worth a whole layer read, its few experts are loaded when it runs
    if(frontier + 1 < n_layer && (frontier == cur_il + 1 || table.loading_count() < params.io_queue)){
        frontier_target = (int32_t) frontier;
    }
    stats[0].t_predict_us += ggml_time_us() - t_start;
    return frontier_target;
}

bool custom_moe_unified::predicts_frontier(uint32_t il, bool prefill) const {
    return prefill && strategy.prefill == custom_prefill_kind::io_aware && il + 1 < moe_layer_count();
}

void custom_moe_unified::on_predicted(uint32_t target_il, const std::vector<int32_t> & ids, bool prefill){
    const int64_t t_start = ggml_time_us();
    table.refresh(async_io);
    if(prefill){
        switch(strategy.prefill){
            case custom_prefill_kind::none:        break;
            case custom_prefill_kind::cross_layer: prefetch_cross_layer(target_il, ids); break;
            case custom_prefill_kind::io_aware:    prefetch_frontier(target_il, ids); break;
        }
    }else{
        switch(strategy.decode){
            case custom_decode_kind::none:         break;
            case custom_decode_kind::next_layer:   predict_next_layer(target_il, ids); break;
        }
    }
    stats[prefill ? 0 : 1].t_predict_us += ggml_time_us() - t_start;
}

std::vector<uint32_t> custom_moe_unified::predict_targets(uint32_t il, bool prefill) const {
    std::vector<uint32_t> targets;
    const uint32_t n_moe_layer = hparams.n_layer() - hparams.n_layer_dense_lead;
    if(!prefill){
        switch(strategy.decode){
            case custom_decode_kind::none:
                break;
            case custom_decode_kind::next_layer:
                if(il + 1 < n_moe_layer){
                    targets.push_back(il + 1);
                }
                break;
        }
        return targets;
    }
    switch(strategy.prefill){
        case custom_prefill_kind::none:
            break;
        case custom_prefill_kind::cross_layer:
            // the input of layer il predicts layer il+1, its loads overlap with the compute of layer il
            if(il + 1 < n_moe_layer){
                targets.push_back(il + 1);
            }
            break;
        case custom_prefill_kind::io_aware:
            // the target is chosen at run time, see predicts_frontier()
            break;
    }
    return targets;
}

ggml_tensor * custom_moe_unified::router(uint32_t il) const {
    return model.layers.at(il + hparams.n_layer_dense_lead).ffn_gate_inp;
}

ggml_tensor * custom_moe_unified::router_bias(uint32_t il) const {
    return model.layers.at(il + hparams.n_layer_dense_lead).ffn_exp_probs_b;
}

//mapping the select_id to pos of pool
llama_pos custom_moe_unified::id_map(uint32_t il,  int32_t selec_id){
    return table.at(il,selec_id).pos;
}
ggml_tensor * custom_moe_unified::get_ups(struct ggml_context * ctx, uint32_t il)const{
    ggml_type cur_type = table.layer_type.at(il).at(0);// up gate down, up == 0
    if(cur_type == pool_type.up){
        return pool.up;
    } else{
        size_t          nb_0 = ggml_type_size(cur_type);
        size_t          nb_1 = ggml_row_size(cur_type,pool.up->ne[0]);
        size_t          nb_2 = pool.up->nb[2];
        ggml_tensor *   res  = ggml_view_3d(ctx,pool.up,pool.up->ne[0],pool.up->ne[1],pool.up->ne[2],nb_1,nb_2,0);
        res->nb[0] = nb_0;
        res->type   = cur_type;
        return res;        
    }
}
ggml_tensor * custom_moe_unified::get_gates(struct ggml_context * ctx, uint32_t il)const{
    ggml_type cur_type = table.layer_type.at(il).at(1);// up gate down, gate == 1
    if(cur_type == pool_type.gate){
        return pool.gate;
    } else{
        size_t          nb_0 = ggml_type_size(cur_type);
        size_t          nb_1 = ggml_row_size(cur_type,pool.gate->ne[0]);
        size_t          nb_2 = pool.gate->nb[2];
        ggml_tensor *   res  = ggml_view_3d(ctx,pool.gate,pool.gate->ne[0],pool.gate->ne[1],pool.gate->ne[2],nb_1,nb_2,0);
        res->nb[0]  = nb_0;
        res->type   = cur_type;
        return res;
    }
}
ggml_tensor * custom_moe_unified::get_downs(struct ggml_context * ctx, uint32_t il)const{
    ggml_type       cur_type = table.layer_type.at(il).at(2);// up gate down, down == 2
    if(cur_type == pool_type.down){
        return pool.down;
    } else{
        size_t          nb_0 = ggml_type_size(cur_type);
        size_t          nb_1 = ggml_row_size(cur_type,pool.down->ne[0]);
        size_t          nb_2 = pool.down->nb[2];
        ggml_tensor *   res  = ggml_view_3d(ctx,pool.down,pool.down->ne[0],pool.down->ne[1],pool.down->ne[2],nb_1,nb_2,0);
        res->nb[0]  = nb_0;
        res->type   = cur_type;
        return res;
    }
}

void custom_moe_unified::routers_init(){
    if (routers_all == nullptr) {
        return;
    }
    const size_t nb_router = ggml_nbytes(router(0));
    const size_t nb_bias   = hparams.n_expert * sizeof(float);
    // the pool buffer may be device memory: copy through the backend, never into tensor->data from the CPU
    std::vector<char> tmp(std::max(nb_router, nb_bias));
    for (uint32_t il = 0; il < moe_layer_count(); il++) {
        ggml_tensor * r = router(il);
        GGML_ASSERT(r->type == routers_all->type && ggml_nbytes(r) == nb_router && ggml_is_contiguous(r));
        ggml_backend_tensor_get(r, tmp.data(), 0, nb_router);
        ggml_backend_tensor_set(routers_all, tmp.data(), il * nb_router, nb_router);

        ggml_tensor * b = router_bias(il);
        GGML_ASSERT((b != nullptr) == (router_biases_all != nullptr) && "custom moe: router bias in some layers only");
        if (b != nullptr) {
            GGML_ASSERT(b->type == GGML_TYPE_F32 && ggml_nbytes(b) == nb_bias);
            ggml_backend_tensor_get(b, tmp.data(), 0, nb_bias);
            ggml_backend_tensor_set(router_biases_all, tmp.data(), il * nb_bias, nb_bias);
        }
    }
}

void custom_moe_unified::table_init(const llama_model & model,llama_model_loader & ml){

    custom_expert_table & table = this->table;

    // Checking the dims of a two-dimensional table
    uint32_t n_row = table.experts.size();       //n_moe_layers
    uint32_t n_col = table.experts[0].size();    //n_experts
    uint32_t n_layer = hparams.n_layer();
    uint32_t n_dense = hparams.n_layer_dense_lead;
    GGML_ASSERT(n_row == (n_layer - n_dense));
    GGML_ASSERT(n_col == hparams.n_expert);

    for(uint32_t i = 0; i < n_row; i++){

        ggml_tensor * ups   =  model.layers.at(i + n_dense).ffn_up_exps;
        ggml_tensor * gates =  model.layers.at(i + n_dense).ffn_gate_exps;
        ggml_tensor * downs =  model.layers.at(i + n_dense).ffn_down_exps;

        const auto * weight_ups = ml.get_weight(ggml_get_name(ups));
        const auto * weight_gates = ml.get_weight(ggml_get_name(gates));
        const auto * weight_downs = ml.get_weight(ggml_get_name(downs));
        GGML_ASSERT(weight_ups->tensor->type == weight_gates->tensor->type);

        ggml_type up_type = weight_ups->tensor->type;
        ggml_type gate_type = weight_gates->tensor->type;
        ggml_type down_type = weight_downs->tensor->type;
        table.layer_type[i] = {
            up_type,
            gate_type,
            down_type
        };
        // experts are stored back to back in the file tensor
        const size_t nbyte_up   = ggml_nbytes(weight_ups->tensor)   / n_col;
        const size_t nbyte_gate = ggml_nbytes(weight_gates->tensor) / n_col;
        const size_t nbyte_down = ggml_nbytes(weight_downs->tensor) / n_col;
        GGML_ASSERT(nbyte_up <= nbyte_slot_up && nbyte_gate <= nbyte_slot_gate && nbyte_down <= nbyte_slot_down);
        table.layer_nbyte[i] = { nbyte_up, nbyte_gate, nbyte_down };

        expert_state initial_state = expert_state::OnDisk;
        for(uint32_t j = 0; j < n_col; j++){
            table.at(i,j).state    = initial_state;
            table.at(i,j).pos      = -1;
            table.at(i,j).predicted = false;
            table.at(i,j).stage     = -1;
            table.at(i,j).prefetched = false;

            table.at(i,j).up.idx   = weight_ups->idx;
            table.at(i,j).up.offs  = weight_ups->offs   + j * nbyte_up;

            table.at(i,j).gate.idx = weight_gates->idx;
            table.at(i,j).gate.offs= weight_gates->offs + j * nbyte_gate;

            table.at(i,j).down.idx = weight_downs->idx;
            table.at(i,j).down.offs= weight_downs->offs + j * nbyte_down;
        }
    }
}

uint32_t custom_moe_unified::get_padding() const {
    //TODO: check the padding Platform porting compatibility
    return 32u;
}

void custom_moe_unified::_free_expert(uint32_t il,int32_t id){
    GGML_ASSERT(!table.at(il,id).predicted);
    GGML_ASSERT(table.at(il,id).stage < 0 && "freed expert still holds a staging slot");
    table.mark(il,id,expert_state::OnDisk,INVALID_POS);
    evict->erase(il,id);
}

// a staging slot for one expert (up + gate + down). wait = false: -1 when every slot is in use
int32_t custom_moe_unified::stage_alloc(bool wait){
    while(1){
        if(!stage_free.empty()){
            const int32_t s = stage_free.back();
            stage_free.pop_back();
            return s;
        }
        if(!wait){
            return -1;
        }
        // every staging slot is held by a read in flight: wait for one to complete (and be copied into the pool)
        GGML_ASSERT(table.loading_count() > 0 && "custom moe: no staging slot can be freed");
        table.refresh(async_io, true);
    }
}

void custom_moe_unified::stage_release(int32_t s){
    GGML_ASSERT(0 <= s && s < n_stage);
    stage_free.push_back(s);
}

char * custom_moe_unified::stage_ptr(int32_t s, int part){
    GGML_ASSERT(0 <= s && s < n_stage && 0 <= part && part < NUM_EXPERT);
    size_t off = 0;
    if(part == static_cast<int>(submit_type::GATE)){
        off = nbyte_slot_up;
    } else if(part == static_cast<int>(submit_type::DOWN)){
        off = nbyte_slot_up + nbyte_slot_gate;
    }
    return static_cast<char*>(ggml_backend_buffer_get_base(staging_buf.get())) + (size_t) s * nbyte_stage_slot + off;
}

// the read finished into staging: move it into the pool slot (device memory on CUDA, tensor_set waits for the copy)
void custom_moe_unified::copy_expert_to_pool(uint32_t il, int32_t id, int32_t s){
    const llama_pos pos = table.at(il,id).pos;
    GGML_ASSERT(pos >= 0);
    const auto & nbyte = table.layer_nbyte[il];
    ggml_backend_tensor_set(pool.up,   stage_ptr(s, static_cast<int>(submit_type::UP)),   (size_t) pos * nbyte_slot_up,   nbyte[0]);
    ggml_backend_tensor_set(pool.gate, stage_ptr(s, static_cast<int>(submit_type::GATE)), (size_t) pos * nbyte_slot_gate, nbyte[1]);
    ggml_backend_tensor_set(pool.down, stage_ptr(s, static_cast<int>(submit_type::DOWN)), (size_t) pos * nbyte_slot_down, nbyte[2]);
}

void custom_moe_unified::load_expert(uint32_t il, int32_t id,llama_pos target_pos,io_mode mode){
    GGML_ASSERT(target_pos >= 0);
    table.mark(il,id,expert_state::Loading,target_pos);
    evict->insert(il,id);
    const auto & nbyte = table.layer_nbyte[il];
    stats[cur_prefill ? 0 : 1].n_bytes += nbyte[0] + nbyte[1] + nbyte[2];
    // io_uring reads into a staging slot, the data is copied into the pool afterwards
    const int32_t s = stage_alloc(mode == io_mode::block);
    GGML_ASSERT(s >= 0 && "custom moe: the caller must leave a staging slot free");
    //load data
        //io::block
    if(mode == io_mode::block){
        async_io.block_read(stage_ptr(s, static_cast<int>(submit_type::UP)),   nbyte[0], table.at(il,id).up.offs);
        async_io.block_read(stage_ptr(s, static_cast<int>(submit_type::GATE)), nbyte[1], table.at(il,id).gate.offs);
        async_io.block_read(stage_ptr(s, static_cast<int>(submit_type::DOWN)), nbyte[2], table.at(il,id).down.offs);
        copy_expert_to_pool(il,id,s);
        stage_release(s);

        table.mark(il,id,expert_state::InMemory,INVALID_POS);// and, the pos of table.mark()(aka '0') is meaningless
    }else{//io::async

        //submit async read, table.refresh() copies it into the pool and marks it InMemory when all 3 parts are done
        table.at(il,id).stage = s;
        table.mark_async(il,id);
        async_io.async_submit(table.map(il,id,submit_type::UP),   stage_ptr(s, static_cast<int>(submit_type::UP)),   nbyte[0], table.at(il,id).up.offs);
        async_io.async_submit(table.map(il,id,submit_type::GATE), stage_ptr(s, static_cast<int>(submit_type::GATE)), nbyte[1], table.at(il,id).gate.offs);
        async_io.async_submit(table.map(il,id,submit_type::DOWN), stage_ptr(s, static_cast<int>(submit_type::DOWN)), nbyte[2], table.at(il,id).down.offs);
    }
}


/*load all data with a layer, blocking (only used at model load)
input:
    il:         the id of table_layer
    target_pos: the pos of pool load to
*/
void custom_moe_unified::load_layer(uint32_t il, llama_pos target_pos){
    table.mark_layer(il,expert_state::Loading,target_pos);
    evict->insert_layer(il, (int32_t) hparams.n_expert);
    const auto & nbyte = table.layer_nbyte[il];
    if(!in_load){
        stats[cur_prefill ? 0 : 1].n_bytes += (size_t) hparams.n_expert * (nbyte[0] + nbyte[1] + nbyte[2]);
    }
    // one expert at a time: read into staging, copy into the pool
    for(int32_t id = 0; id < static_cast<int32_t>(hparams.n_expert); id++){
        const int32_t s = stage_alloc(true);
        async_io.block_read(stage_ptr(s, static_cast<int>(submit_type::UP)),   nbyte[0], table.at(il,id).up.offs);
        async_io.block_read(stage_ptr(s, static_cast<int>(submit_type::GATE)), nbyte[1], table.at(il,id).gate.offs);
        async_io.block_read(stage_ptr(s, static_cast<int>(submit_type::DOWN)), nbyte[2], table.at(il,id).down.offs);
        copy_expert_to_pool(il,id,s);
        stage_release(s);
        table.mark(il,id,expert_state::InMemory,INVALID_POS);
    }
}

bool custom_moe_unified::can_evict(uint32_t il, int32_t id, uint32_t cur_il){
    // the experts of cur_il may already be mapped for this token, a loading expert has a read in flight,
    // a predicted expert waits for its layer (without this, lru would evict it before the finished layers)
    const auto & e = table.at(il,id);
    return il != cur_il && e.state == expert_state::InMemory && !e.predicted;
}

bool custom_moe_unified::can_evict_unused(uint32_t il, int32_t id){
    // only the experts in cur_used are mapped for this ubatch, the other experts of cur_il are free to go
    const auto & e = table.at(il,id);
    return (il != cur_il || !cur_used[id]) && e.state == expert_state::InMemory && !e.predicted;
}


/* 
input: 
    n: num of slots  
return:
    pos of alloc in pool
    pos == -1:  alloc failed
*/
llama_pos custom_moe_unified::pool_alloc(int32_t n){
    if(n <= 0)              return -1;
    if(n > pool.n_slots)   return -1;
    auto& free_slots = pool.free_slots;
    for(auto it = free_slots.begin(); it != free_slots.end(); it++){
        llama_pos cur_pos   = it->first;
        int32_t    length    = it->second;
        if (n <= length){
            free_slots.erase(it);
            if(n < length){
                //insert a new slot_map
                free_slots[cur_pos + n] = length - n;
            }                    
            return cur_pos;
        }
    }
    return -1;

}

void custom_moe_unified::pool_free(llama_pos pos, int32_t n){
    if(n <= 0) return;
    if(pos < 0 || ((pos + n - 1) > pool.n_slots))return;
    auto& free_slots = pool.free_slots;

    auto next = free_slots.upper_bound(pos);
    auto prev = (next != free_slots.begin()) ? std::prev(next) : free_slots.end();

    bool merged = false;
    if(prev != free_slots.end() && (prev->first + prev->second == pos)){
        prev->second += n;
        merged = true;
    }
    if((next != free_slots.end()) && (pos + n == next->first)){
        if(merged){
            prev->second += next->second;
        } else{
            free_slots[pos] = n + next->second;
        }
        free_slots.erase(next);    
    } else if (!merged){
        free_slots[pos] = n;
    }
    
}

/*
multi-thread
*/
class CallSync {
public:
    std::atomic<int> count{0};
    std::mutex mtx;
    std::condition_variable cv;
    bool init_done = false;
    
    void wait_initialization(std::function<void()> init_fn) {
        std::unique_lock<std::mutex> lock(mtx);
        if (!init_done) {
            //initial
            init_fn();
            init_done = true;
            cv.notify_all();
        }
        cv.wait(lock, [&]{ return init_done == true; });
    }
    
    void mark_completion(int nth) {
        int current = count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (current == nth) {
            std::unique_lock<std::mutex> lock(mtx);
            // reset
            init_done = false;
            count.store(0, std::memory_order_relaxed);
        }
    }
};
//adjustment only of jetson nano
std::string extra_copyname(const std::string& str) {
    size_t first = str.find_first_of('#');
    size_t second = str.find_first_of('#', first + 1);
    
    if (first != std::string::npos && second != std::string::npos) {
        return str.substr(first + 1, second - first - 1);
    }
    return "";
}

// expert ids [n_ids, n_tokens] -> unique ids, in order of first use
static std::vector<int32_t> unique_ids(const struct ggml_tensor * a, uint32_t n_expert){
    const int32_t * data   = (const int32_t *) ggml_get_data(a);
    const int64_t   stride = a->nb[1] / sizeof(int32_t);
    std::vector<bool>    seen(n_expert, false);
    std::vector<int32_t> ids;
    for(int64_t i = 0; i < a->ne[1]; i++){
        for(int64_t j = 0; j < a->ne[0]; j++){
            const int32_t id = data[i*stride + j];
            GGML_ASSERT(0 <= id && (uint32_t) id < n_expert);
            if(!seen[id]){
                seen[id] = true;
                ids.push_back(id);
            }
        }
    }
    return ids;
}

// mapping the select_id to pos of pool
void id_pos_map(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth,  void * userdata){
    GGML_ASSERT(userdata != nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, a));
    GGML_ASSERT(a->ne[2] == a->ne[3]);
    GGML_ASSERT(a->ne[2] == 1);
    const int32_t       * a_data        = (int32_t *)ggml_get_data(a);
    int32_t             * dst_data      = (int32_t *)ggml_get_data(dst);
    runtime_ctx         * ctx           = (runtime_ctx *)userdata;
    custom_moe_unified  * moe_unified   = ctx->moe_unified;
    int                   il            = ctx->il;
    GGML_UNUSED(b); // only a graph dependency, the eviction policy does not use the router probs

    const int n_ids   = a->ne[0];
    const int n_token = a->ne[1];

    // load the experts of this layer used by any token, then record the uses in token order (lru access sequence)
    // and the routing (ARPE); single threaded: CallSync in prefill, ith == 0 in decode
    auto load_and_use = [&]{
        const bool prefill = moe_unified->ubatch_prefill(il, n_token);
        moe_unified->ensure(il, unique_ids(a, moe_unified->hparams.n_expert), prefill);
        moe_unified->evict->observe(il, a_data, n_ids, n_token, a->nb[1] / 4, prefill);
        for(int i = 0; i < n_token; i++){
            for(int j = 0; j < n_ids; j++){
                moe_unified->evict->use(il, a_data[i*(a->nb[1] / 4) + j]);
            }
        }
    };

    if(n_token > 1){// prefill: one thread loads, then all threads map
        static CallSync sync;
        sync.wait_initialization(load_and_use);
        const int dr = (n_token + nth - 1) / nth;
        const int ie0 = dr * ith;
        const int ie1 = MIN(ie0 + dr, n_token);
        for(int i = ie0; i < ie1; i++){
            for(int j = 0; j < n_ids; j++){
                dst_data[i*(dst->nb[1] / 4) +j] = moe_unified->id_map(il,a_data[i*(a->nb[1] / 4) + j]);
            }
        }

        sync.mark_completion(nth);
    } else{// decode: too little work to split
        if(ith == 0){
            load_and_use();
            for(int j = 0; j < n_ids; j++){
                dst_data[j] = moe_unified->id_map(il,a_data[j]);
            }
        }
    }
}

// ctx->il is the predicted (target) moe layer, the layer being computed is moe_unified->cur_il
// what happens with the prediction depends on the strategy, see custom_moe_unified::on_predicted
void custom_predict(struct ggml_tensor * dst , const struct ggml_tensor * top_k, int ith, int nth,  void * userdata){
    GGML_ASSERT(userdata != nullptr);
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);

    runtime_ctx         * ctx               = (runtime_ctx *)userdata;
    custom_moe_unified  * moe_unified       = ctx->moe_unified;

    if(ith != 0){
        return;
    }
    // unique ids in order of first use: for one token this is the top-k order
    moe_unified->on_predicted(ctx->il, unique_ids(top_k, moe_unified->hparams.n_expert), moe_unified->in_prefill());
}

// io-aware prefill: choose the layer to predict now, write its rows in routers_all
void custom_frontier(struct ggml_tensor * dst, int ith, int nth, void * userdata){
    GGML_ASSERT(userdata != nullptr);
    GGML_UNUSED(nth);

    runtime_ctx         * ctx           = (runtime_ctx *)userdata;
    custom_moe_unified  * moe_unified   = ctx->moe_unified;

    if(ith != 0){
        return;
    }
    const int32_t target   = moe_unified->next_frontier();
    // nothing to predict: any valid rows, custom_predict_frontier ignores the result
    const int32_t layer    = target >= 0 ? target : ctx->il;
    const int32_t n_expert = moe_unified->hparams.n_expert;
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && ggml_nelements(dst) == n_expert);
    int32_t * rows = (int32_t *) dst->data;
    for(int32_t j = 0; j < n_expert; j++){
        rows[j] = layer * n_expert + j;
    }
}

void custom_predict_frontier(struct ggml_tensor * dst , const struct ggml_tensor * top_k, int ith, int nth,  void * userdata){
    GGML_ASSERT(userdata != nullptr);
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);

    runtime_ctx         * ctx           = (runtime_ctx *)userdata;
    custom_moe_unified  * moe_unified   = ctx->moe_unified;

    if(ith != 0 || moe_unified->frontier_target < 0){
        return;
    }
    moe_unified->on_predicted((uint32_t) moe_unified->frontier_target, unique_ids(top_k, moe_unified->hparams.n_expert), true);
}
