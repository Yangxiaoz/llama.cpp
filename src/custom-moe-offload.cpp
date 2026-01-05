#include "custom-moe-offload.h"

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
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
        throw std::runtime_error("Failed to get SQE");
    }

    io_uring_prep_read(sqe, 0, buf, size, offset);//0: the index of register file
    sqe->flags |= IOSQE_FIXED_FILE;

    // submit
    int ret = io_uring_submit(&ring);
    if (ret < 0) {
        throw std::runtime_error("custom_asyncio::Failed to submit request:" + std::string(strerror(-ret)));
    }

    // waiting for complete
    struct io_uring_cqe* cqe;
    ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
        throw std::runtime_error("custom_asyncio::Failed to wait for completion:" + std::string(strerror(-ret)));
    }
    //clean cur cqe
    int read_bytes = cqe->res;
    io_uring_cqe_seen(&ring, cqe);
    if (read_bytes < 0) {
        throw std::runtime_error("custom_asyncio::Read failed in block reading");
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
    sqe->flags |= IOSQE_FIXED_FILE;
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
std::vector<uint32_t> custom_async_io::refresh_completions() {
    struct io_uring_cqe* cqe;
    std::vector<uint32_t> completed_ids;

    while (io_uring_peek_cqe(&ring, &cqe) == 0) {
        // 获取到了一个 cqe
        uint32_t id = process_cqe(cqe); // 内部调用 seen，移动队列指针
        if (id == 0) {
            continue;
        }
        completed_ids.push_back(id);
    }
    return completed_ids;   
}



custom_expert_table::custom_expert_table(uint32_t n_moe_layer, uint32_t n_expert)
    :n_row(n_moe_layer), 
     n_col(n_expert),
     load_ctx(n_moe_layer, LoadContext(n_expert)),
     experts(n_moe_layer,std::vector<custom_expert_group>(n_expert)),
     layer_type(n_moe_layer)
    {}



custom_moe_unified::custom_moe_unified(const llama_model & model,const std::string & fname,llama_model_loader & ml,bool enable_optimizations,float utilization) 
    : params{enable_optimizations,utilization,1,2,0,0,0,0},
      hparams(model.hparams),
      model(model),
      async_io(fname),
      table(hparams.n_layer - hparams.n_layer_dense_lead, hparams.n_expert){


    uint32_t n_expert = hparams.n_expert;
    //select the last layer as typical expert
    ggml_tensor * classic_up = model.layers.back().ffn_up_exps;                 
    ggml_tensor * classic_gate = model.layers.back().ffn_gate_exps;
    ggml_tensor * classic_down = model.layers.back().ffn_down_exps;             

    GGML_ASSERT(classic_up->type == classic_gate->type);   //assert the gate is equal to up
    this->expert_ne[0] = classic_up->ne[0];
    this->expert_ne[1] = classic_up->ne[1];
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
    //TBD: need to refactor the logic for buft select
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    // check if it is possible to use buffer_from_host_ptr with GPU
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);

    if (!dev) {
        // FIXME: workaround for CPU backend buft having a NULL device
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
    }
    auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
    if(!host_buft){
        host_buft = cpu_buft;
    }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);

    //get the number of slots in pool
    uint32_t nbyte_group = nbyte_slot_up + nbyte_slot_gate + nbyte_slot_down; 
    uint64_t n_byte_availa = (uint64_t)(utilization * props.memory_total);
    int32_t n_slots  = n_byte_availa / nbyte_group;


// create a context for  buffer 
    ggml_init_params buffer_params = {
        /*.mem_size   =*/ size_t(3 *ggml_tensor_overhead()), //3: up gate down
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


    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
    if (!buf) {
        throw std::runtime_error("failed to allocate buffer for moe_pool");
    }
    // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bufs.emplace_back(buf);
    
    LLAMA_LOG_INFO("n_slot of pool is:%d \n",n_slots);
    LLAMA_LOG_INFO("%s: %10s moe_expert_pool buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

    //initialize a blank management instance
    manage = std::make_unique<custom_expert_manage>(table);

    params.n_preload_layers = n_slots /n_slots_layer;
    params.head_il = 0;
    params.tail_il = n_slots /n_slots_layer;
    params.n_mean_expert = n_slots / hparams.n_layer;
  
}

uint32_t custom_moe_unified::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

void custom_moe_unified::prefill_init(){
    
    for(uint32_t il = 0; il < params.n_preload_layers; il++){
        llama_pos slot_pos = pool_alloc(n_slots_layer);
        load_layer(il,slot_pos,io_mode::block);
    }
    
    LLAMA_LOG_INFO("custom_moe prefill init: %d layers have been pre-load",params.n_preload_layers);
}

void custom_moe_unified::check_layer(uint32_t il){
    table.refresh(async_io);
    if(table.layer_check(il)){
        return;
    }
    if(table.is_layer_trigger(il)){//the layer load has been triggered
        //wait for loading complete
        while(table.layer_check(il) == false){
            table.refresh(async_io);
        }
        //layer load complete
        //mark all expert in layer as expertstate::InMemory
        table.mark_layer(il,expert_state::InMemory,INVALID_POS);
        return;

    }else{// no trigger yet
        //blocking load the layer now 
        GGML_ASSERT(il > 0);//the first layer should always be activate
        uint32_t prev_il = il -1;

        //TEMP: tempprary implementation 
        //always use the pool of prev layer to load cur layer of weight
        GGML_ASSERT(table.layer_check(prev_il) == true);
        //Reuse the pool of prev_layer for cur_layer
        llama_pos pos = table.at(prev_il,LAYER_HEAD).pos;
        //free the prev_layer in table
        _free_layer(prev_il);

        //load layer to pool
        load_layer(il,pos,io_mode::block);
    }
}

void custom_moe_unified::check_expert(uint32_t il,uint32_t id){

    auto state = table.at(il,id).state;
    table.refresh(async_io);
    uint32_t curLayer = il;
    uint32_t loadLayer = (il + 1 > (hparams.n_layer - hparams.n_layer_dense_lead -1)) ? il : il + 1;
    if(state == expert_state::OnDisk){
        llama_pos pos = pool_alloc(1);
        if(pos == -1){
            if(params.optimization){//params.optimization
                pos = free_lru(curLayer,loadLayer);//try to free from other layer first
            }else{
                pos = free_random({curLayer,loadLayer});
            }
            GGML_ASSERT(pos != -1);
        }  
        load_expert(il,id,pos,io_mode::block);
    }else if(state == expert_state::Loading){
        while(1){
            table.refresh(async_io);
            if(table.at(il,id).state == expert_state::InMemory){
                return;
            }
        }
    }
}

bool custom_moe_unified::predict_expert(uint32_t il,uint32_t id){
    if(table.at(il,id).state == expert_state::OnDisk){
        llama_pos pos = pool_alloc(1);
        if(pos == -1){
            pos = free_lru(il - 1,il);
            // pos = free_random({il - 1,il});
            GGML_ASSERT(pos != -1);
        }  
        load_expert(il,id,pos,io_mode::async);
        return true;
    }
    return false;
}
void custom_moe_unified::predict_layer(uint32_t head_il,uint32_t tail_il){
    GGML_ASSERT(tail_il <(hparams.n_layer - hparams.n_layer_dense_lead));

    //TEMP: tempprary implementation 
    //always use the pool of prev layer to load cur layer of weight
    GGML_ASSERT(table.layer_check(head_il) == true);
    //Reuse the pool of prev_layer for cur_layer
    llama_pos pos = table.at(head_il,LAYER_HEAD).pos;
    //free the prev_layer in table
    _free_layer(head_il);
    //load layer to pool
    load_layer(tail_il,pos,io_mode::async);
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

void custom_moe_unified::table_init(const llama_model & model,llama_model_loader & ml){

    custom_expert_table & table = this->table;

    // Checking the dims of a two-dimensional table
    uint32_t n_row = table.experts.size();       //n_moe_layers
    uint32_t n_col = table.experts[0].size();    //n_experts
    uint32_t n_layer = hparams.n_layer;
    uint32_t n_dense = hparams.n_layer_dense_lead;
    GGML_ASSERT(n_row == (n_layer - n_dense));
    GGML_ASSERT(n_col == hparams.n_expert);


    std::map<ggml_type,size_t> type_map;
    auto type_for_size = [&](ggml_type type) -> size_t {
        auto it = type_map.find(type);
        if (it == type_map.end()) {
            size_t nbyte =(expert_ne[0] * expert_ne[1])* ggml_type_size(type) / ggml_blck_size(type);
            type_map[type] = nbyte;
            return nbyte;
        }
        return it->second;
    };

    for(uint32_t i = 0; i < n_row; i++){

        ggml_tensor * ups   =  model.layers.at(i + n_dense).ffn_up_exps;
        ggml_tensor * gates =  model.layers.at(i + n_dense).ffn_gate_exps;
        ggml_tensor * downs =  model.layers.at(i + n_dense).ffn_down_exps;

        const auto * weight_ups = ml.get_weight(ggml_get_name(ups));
        const auto * weight_gates = ml.get_weight(ggml_get_name(gates));
        const auto * weight_downs = ml.get_weight(ggml_get_name(downs));
        GGML_ASSERT(weight_ups->tensor->type = weight_gates->tensor->type);

        ggml_type up_type = weight_ups->tensor->type;
        ggml_type gate_type = weight_gates->tensor->type;
        ggml_type down_type = weight_downs->tensor->type;
        table.layer_type[i] = {
            up_type,
            gate_type,
            down_type
        };
        expert_state initial_state = expert_state::OnDisk;
        for(uint32_t j = 0; j < n_col; j++){
            table.at(i,j).state    = initial_state;
            table.at(i,j).pos      = -1;

            table.at(i,j).up.idx   = weight_ups->idx; 
            table.at(i,j).up.offs  = weight_ups->offs + j*  type_for_size(up_type);

            table.at(i,j).gate.idx = weight_gates->idx;
            table.at(i,j).gate.offs= weight_gates->offs +j* type_for_size(gate_type);

            table.at(i,j).down.idx = weight_downs->idx;
            table.at(i,j).down.offs= weight_downs->offs +j* type_for_size(down_type);
        }
    }   
}

uint32_t custom_moe_unified::get_padding() const {
    //TODO: check the padding Platform porting compatibility
    return 32u;
}

void custom_moe_unified::_free_expert(uint32_t il,int32_t id){
    table.mark(il,id,expert_state::OnDisk,INVALID_POS);
    manage.get()->erase(il,id);
}
void custom_moe_unified::_free_layer(uint32_t il){
    table.mark_layer(il,expert_state::OnDisk,INVALID_POS);
    manage.get()->erase_layer(il);
}

void custom_moe_unified::load_expert(uint32_t il, int32_t id,llama_pos target_pos,io_mode mode){
    table.mark(il,id,expert_state::Loading,target_pos);
    manage.get()->touch(il,id);
    //load data
        //io::block
    if(mode == io_mode::block){
        async_io.block_read(static_cast<char*>(pool.up->data)   + target_pos * nbyte_slot_up,     nbyte_slot_up,      table.at(il,id).up.offs);
        async_io.block_read(static_cast<char*>(pool.gate->data) + target_pos * nbyte_slot_gate,   nbyte_slot_gate,    table.at(il,id).gate.offs);
        async_io.block_read(static_cast<char*>(pool.down->data) + target_pos * nbyte_slot_down,   nbyte_slot_down,    table.at(il,id).down.offs);

        table.mark(il,id,expert_state::InMemory,INVALID_POS);// and, the pos of table.mark()(aka '0') is meaningless
    }else{//io::async

        //submit async read
        int identity_index = table.map(il,id,submit_type::UP);
        async_io.async_submit(identity_index,static_cast<char*>(pool.up->data)   + target_pos * nbyte_slot_up,   nbyte_slot_up,   table.at(il,id).up.offs);
        identity_index = table.map(il,id,submit_type::GATE);
        async_io.async_submit(identity_index,static_cast<char*>(pool.gate->data) + target_pos * nbyte_slot_gate, nbyte_slot_gate, table.at(il,id).gate.offs);
        identity_index = table.map(il,id,submit_type::DOWN);
        async_io.async_submit(identity_index,static_cast<char*>(pool.down->data) + target_pos * nbyte_slot_down, nbyte_slot_down, table.at(il,id).down.offs);
    }
}


/*load all data with a layer
input:
    il:         the id of table_layer
    target_pos: the pos of pool load to
*/
void custom_moe_unified::load_layer(uint32_t il, llama_pos target_pos,io_mode mode){
    table.mark_layer(il,expert_state::Loading,target_pos);
    //the reason there directly use active_layer is prefill just call load_layer once
    manage.get()->active_layer(il);
    //load data
        //io::block
    if(mode == io_mode::block){
        //up
        for(int32_t id = 0; id < static_cast<int32_t>(hparams.n_expert); id++){
            async_io.block_read(static_cast<char*>(pool.up->data)   + (target_pos + id) * nbyte_slot_up,     nbyte_slot_up,      table.at(il,id).up.offs);
        }
        //gate
        for(int32_t id = 0; id < static_cast<int32_t>(hparams.n_expert); id++){
            async_io.block_read(static_cast<char*>(pool.gate->data) + (target_pos + id) * nbyte_slot_gate,   nbyte_slot_gate,    table.at(il,id).gate.offs);
        }
        //down
        for(int32_t id = 0; id < static_cast<int32_t>(hparams.n_expert); id++){
            async_io.block_read(static_cast<char*>(pool.down->data) + (target_pos + id) * nbyte_slot_down,   nbyte_slot_down,    table.at(il,id).down.offs);
        }
        table.mark_layer(il,expert_state::InMemory,INVALID_POS);
    }else{//io::async        
        for(int32_t id = 0; id < static_cast<int32_t>(hparams.n_expert); id++){
            int identity_index = table.map(il,id,submit_type::UP);
            async_io.async_submit(identity_index,static_cast<char*>(pool.up->data)   + (target_pos + id) * nbyte_slot_up,   nbyte_slot_up,   table.at(il,id).up.offs);
        };
        for(int32_t id = 0; id < static_cast<int32_t>(hparams.n_expert); id++){
            int identity_index = table.map(il,id,submit_type::GATE);
            async_io.async_submit(identity_index,static_cast<char*>(pool.gate->data) + (target_pos + id) * nbyte_slot_gate, nbyte_slot_gate, table.at(il,id).gate.offs);
        };
        for(int32_t id = 0; id < static_cast<int32_t>(hparams.n_expert); id++){
            int identity_index = table.map(il,id,submit_type::DOWN);
            async_io.async_submit(identity_index,static_cast<char*>(pool.down->data) + (target_pos + id) * nbyte_slot_down, nbyte_slot_down, table.at(il,id).down.offs);
        };
    }

}
llama_pos custom_moe_unified::free_random(const std::vector<uint32_t>& excluded_layers){

    //To recheck the random free strategy
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<int> dis_layer(0, table.row_size() - 1);
    std::uniform_int_distribution<int> dis_expert(0, table.col_size() - 1);

    //find a layer to free
    while(1){
        int32_t layer_id = static_cast<int32_t> (dis_layer(gen));
        if(std::find(excluded_layers.begin(), excluded_layers.end(), layer_id) != excluded_layers.end()){
            continue;
        }
        uint32_t expert_id = static_cast<uint32_t> (dis_expert(gen));
        if(table.at(layer_id,expert_id).state == expert_state::InMemory){
            //free this expert
            llama_pos pos = table.at(layer_id,expert_id).pos;
            _free_expert(layer_id,expert_id);
            return pos;
        }

    }
}
llama_pos custom_moe_unified::free_lru(uint32_t cur_layer_id, uint32_t load_layer_id){

    auto management = manage.get();
    //now, temp get the last active expert

    int32_t free_layer_id = management->get_freeLayer_id(cur_layer_id,load_layer_id);
    GGML_ASSERT(free_layer_id != -1 && "erro in free_lru: no layer to free");
    int32_t expert_id = management->get_back_n(free_layer_id, 0);
    GGML_ASSERT(expert_id != -1 && "erro in free_lru: no expert to free");
    GGML_ASSERT(table.at(free_layer_id,expert_id).state == expert_state::InMemory);

    llama_pos pos = table.at(free_layer_id,expert_id).pos;
    _free_expert(free_layer_id,expert_id);
    return pos;
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

// mapping the select_id to pos of pool
void id_pos_map(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth,  void * userdata){
    GGML_ASSERT(userdata != nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, a));
    GGML_ASSERT(a->ne[2] == a->ne[3]);
    GGML_ASSERT(a->ne[2] == 1);
    
    bool prefill = (a->ne[1] > 1);
    const int32_t       * a_data        = (int32_t *)ggml_get_data(a);
    int32_t             * dst_data      = (int32_t *)ggml_get_data(dst);
    runtime_ctx         * ctx           = (runtime_ctx *)userdata;
    custom_moe_unified  * moe_unified   = ctx->moe_unified;
    int                   il            = ctx->il;
    auto manage = moe_unified->manage.get();

    ////////////////////

    int n_ids   = a->ne[0];
    int n_token = a->ne[1];
    //wait to check layer state when prefill
    static CallSync sync;
    if(prefill){// parallel
        //prefill check
        sync.wait_initialization([&]{//only first thread will call 
        if(prefill)moe_unified->check_layer(il);
        });
        const int dr = (n_token + nth - 1) / nth;
        const int ie0 = dr * ith;
        const int ie1 = MIN(ie0 + dr, n_token);
        //mapping
        for(int i = ie0; i < ie1; i++){
            for(int j = 0; j < n_ids; j++){
                //TBD: warnning! multithread erro
                // manage->hit(il,a_data[i*(a->nb[1] / 4) + j]);
                dst_data[i*(dst->nb[1] / 4) +j] = moe_unified->id_map(il,a_data[i*(a->nb[1] / 4) + j]);
            }
        }
        sync.mark_completion(nth);
    } else{// no parallel
        if(ith == 0){
            for(int i = 0; i < n_ids; i++){
                //check
                moe_unified->check_expert(il,a_data[i]);
                //record the hit event
                manage->hit(il,a_data[i]);
                //mapping
                dst_data[i] = moe_unified->id_map(il,a_data[i]);
            }
            delete ctx;
        }
    }
}

void custom_predict(struct ggml_tensor * dst , const struct ggml_tensor * top_k, int ith, int nth,  void * userdata){
    GGML_ASSERT(userdata != nullptr);
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);

    
    const int32_t       * topk_data         = (int32_t *)ggml_get_data(top_k);
    runtime_ctx         * ctx               = (runtime_ctx *)userdata;
    custom_moe_unified  * moe_unified       = ctx->moe_unified;
    uint32_t              il                = ctx->il;

    // float             * dst_data      = (float *)ggml_get_data(dst);
    const int64_t n_select = top_k->ne[0];
    const int64_t n_tokens = top_k->ne[1];

    if(ith == 0){
        if(n_tokens == 1){//decode
            for(int32_t i = 0; i < n_select;i++){
                int32_t expert_id = topk_data[i];
                if(moe_unified->predict_expert(il + 1,expert_id))
                    break;
            }
        }else{//prefill
            //TODO: need head_il and tail_il
            if(il == 0)return;
            if((il == moe_unified->hparams.n_layer - moe_unified->hparams.n_layer_dense_lead -2))return;
            //predict or prefetch the next layer
            if((il <= moe_unified->params.n_prefetch_layers) || (il == moe_unified->params.tail_il -1)){
                uint32_t head_il = moe_unified->params.head_il;
                uint32_t tail_il = moe_unified->params.tail_il;
                moe_unified->predict_layer(head_il,tail_il);
                moe_unified->params.head_il = head_il + 1;
                moe_unified->params.tail_il = tail_il + 1;
                return;
            }
        }
        delete ctx;
    }

    
}