#include "custom-moe-offload.h"

#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>



class custom_io_read{
public:
    custom_io_read(llama_file *f): file(f) {}

    void read_to(void * dst, size_t size) {
        file->read_raw(dst, size);
    }
private:
    llama_file * file;
};


custom_expert_table::custom_expert_table(uint32_t n_moe_layer, uint32_t n_expert)
    :  experts(n_moe_layer,std::vector<custom_expert_group>(n_expert)),
       n_row(n_moe_layer),
       n_col(n_expert){}



custom_moe_unified::custom_moe_unified(const llama_model & model,float utilization,const std::string & fname,llama_model_loader & ml) 
    : table(model.hparams.n_layer - model.hparams.n_layer_dense_lead, model.hparams.n_expert),
      model(model),
      hparams(model.hparams){

    uint32_t n_expert = hparams.n_expert;
    //select up as typical expert
    ggml_tensor * classic_up_gate = model.layers.back().ffn_up_exps;                 //select the "up" of last layer as typical expert
    GGML_ASSERT(classic_up_gate->type == model.layers.back().ffn_gate_exps->type);   //assert the gate is equal to up
    ggml_tensor * classic_down = model.layers.back().ffn_down_exps;             //select the "down" of last layer as typical expert

    this->type_up_gate  = classic_up_gate->type;
    this->type_down     = classic_down->type;  

    this->nbyte_up_gate = ggml_nbytes(classic_up_gate)   / n_expert;
    this->nbyte_down    = ggml_nbytes(classic_down) / n_expert;

    this->nbyte_group   = nbyte_up_gate * 2 + nbyte_down; //up、gate、down

    //init the table
    table_init(model,fname,ml);

    //init the moe_pools
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

    uint64_t n_byte_availa = (uint64_t)(utilization * props.memory_total);
    int32_t n_slots  = n_byte_availa / nbyte_group;
    // uint64_t n_size = M_PAD((n_byte_availa/3),padding);



// create a context for  buffer 
    ggml_init_params params = {
        /*.mem_size   =*/ size_t(3 *ggml_tensor_overhead()), //3: up gate down
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
        if (!ctx) {
        throw std::runtime_error(format("%s: moe_pool_unified ctx alloc failed", __func__));
    }

    ctxs.emplace_back(ctx);

    //create moe_pool_tensors
    ggml_tensor * up;
    ggml_tensor * gate;
    ggml_tensor * down;
    //memory alignment assert
    const auto n_pad = get_padding();
    GGML_ASSERT(classic_up_gate->nb[2] % n_pad == 0);
    GGML_ASSERT(classic_down->nb[2] % n_pad == 0);
    up = ggml_new_tensor_3d(ctx, type_up_gate, classic_up_gate->ne[0], classic_up_gate->ne[1],n_slots);
    gate = ggml_new_tensor_3d(ctx, type_up_gate, classic_up_gate->ne[0],classic_up_gate->ne[1],n_slots);
    down = ggml_new_tensor_3d(ctx, type_down, classic_down->ne[0],classic_down->ne[1],n_slots);
    ggml_format_name(up,   "pool_ups");
    ggml_format_name(gate, "pool_gate");
    ggml_format_name(down, "pool_downs");
    pools = {
        /*tensor_up         =*/ up,
        /*tensor_gate       =*/ gate,
        /*tensor_down       =*/ down,
        /*n_slots           =*/ n_slots,
        /*free_slots        =*/ {{0,n_slots}}
    };


    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
    if (!buf) {
        throw std::runtime_error("failed to allocate buffer for moe_pools");
    }
    // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bufs.emplace_back(buf);
    
    LLAMA_LOG_INFO("n_slot of pool is:%d \n",n_slots);
    LLAMA_LOG_INFO("%s: %10s moe_expert_pool buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

}

uint32_t custom_moe_unified::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

void custom_moe_unified::table_init(const llama_model & model,const std::string & fname,llama_model_loader & ml){

    custom_expert_table & table    = this->table;

    uint32_t n_layer = hparams.n_layer;
    uint32_t n_dense = hparams.n_layer_dense_lead;
    //init files
    table.files.emplace_back(new llama_file(fname.c_str(), "rb"));

    // Checking the dims of a two-dimensional table
    uint32_t n_row = table.experts.size();       //n_moe_layers
    uint32_t n_col = table.experts[0].size();    //n_experts
    GGML_ASSERT(n_row == (n_layer - n_dense));
    GGML_ASSERT(n_col == hparams.n_expert);
    
    for(uint32_t i = 0; i < n_row; i++){

        ggml_tensor * ups   =  model.layers.at(i + n_dense).ffn_up_exps;
        ggml_tensor * gates =  model.layers.at(i + n_dense).ffn_gate_exps;
        ggml_tensor * downs =  model.layers.at(i + n_dense).ffn_down_exps;

        const auto * weight_ups = ml.get_weight(ggml_get_name(ups));
        const auto * weight_gates = ml.get_weight(ggml_get_name(gates));
        const auto * weight_downs = ml.get_weight(ggml_get_name(downs));
        // weight_ups->offs;
        expert_state initial_state = expert_state::OnDisk;
        for(uint32_t j = 0; j < n_col; j++){
            table.at(i,j).state    = initial_state;
            table.at(i,j).pos      = -1;

            table.at(i,j).up.idx   = weight_ups->idx; 
            table.at(i,j).up.offs  = weight_ups->offs + j*  nbyte_up_gate;

            table.at(i,j).gate.idx = weight_gates->idx;
            table.at(i,j).gate.offs= weight_gates->offs +j* nbyte_up_gate;

            table.at(i,j).down.idx = weight_downs->idx;
            table.at(i,j).down.offs= weight_downs->offs +j* nbyte_down;
        }
    }   
}


void custom_moe_unified::prefill_init(){
    uint32_t n_slot_layer =  hparams.n_expert;
    uint32_t n_layer_load =  pools.n_slots / n_slot_layer;
    for(uint32_t il = 0; il < n_layer_load; il++){
        llama_pos slot_pos = pool_alloc(n_slot_layer);
        load_layer(il,slot_pos);
    }
    n_prefill_loaded = n_layer_load;
    LLAMA_LOG_INFO("custom_moe prefill init: %d layers have been pre-load",n_prefill_loaded);
}

void custom_moe_unified::prefill_step(){
    //TBD: need to implement
    llama_pos target_pos =  table.at(i_cur_layer - 1,0).pos;
    load_layer(i_cur_layer,target_pos);
    i_cur_layer++;
}

/* 
input: 
    enum alloc_type{
        Layer   = 1,
        Singel  = 2
    };
return:
    pos of alloc in pools
    pos == -1:  alloc failed
*/
llama_pos custom_moe_unified::pool_alloc(int32_t n){
    if(n <= 0)              return -1;
    if(n > pools.n_slots)   return -1;
    auto& free_slots = pools.free_slots;
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
    if(pos < 0 || ((pos + n - 1) > pools.n_slots))return;
    auto& free_slots = pools.free_slots;

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


void custom_moe_unified::load_data(ggml_tensor * dst,llama_pos offset,size_t n_size,uint32_t row, uint32_t col){
    std::vector<no_init<uint8_t>> read_buf;
    //memory alignment assert
    uint32_t n_pad = get_padding();
    GGML_ASSERT((offset % n_pad) == 0);
    uint16_t    idx  = table.at(row,col).gate.idx;
    size_t      offs = table.at(row,col).gate.offs;
    const auto & file = table.files.at(idx);
    
    if (ggml_backend_buffer_is_host(dst->buffer)) {

        file->seek(offs, SEEK_SET);
        file->read_raw(static_cast<char*>(dst->data) + offset, n_size);
    } else{
            if (0) {//TBD: need to support async load
                // file->seek(weight->offs, SEEK_SET);

                // size_t bytes_read = 0;

                // while (bytes_read < n_size) {
                //     size_t read_iteration = std::min<size_t>(buffer_size, n_size - bytes_read);

                //     ggml_backend_event_synchronize(events[buffer_idx]);
                //     file->read_raw(host_ptrs[buffer_idx], read_iteration);
                //     ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[buffer_idx], bytes_read, read_iteration);
                //     ggml_backend_event_record(events[buffer_idx], upload_backend);

                //     bytes_read += read_iteration;
                //     ++buffer_idx;
                //     buffer_idx %= n_buffers;
                // }
            } else {
                read_buf.resize(n_size);
                file->seek(offs, SEEK_SET);
                file->read_raw(read_buf.data(), n_size);
                ggml_backend_tensor_set(dst, read_buf.data(), offset, n_size);
            }
    }

}

void custom_moe_unified::load_expert(uint32_t il, uint32_t id){
    llama_pos slots_pos = pool_alloc(1);

    load_data(pools.up,slots_pos,nbyte_up_gate,il,id);
    load_data(pools.gate,slots_pos,nbyte_up_gate,il,id);
    load_data(pools.down,slots_pos,nbyte_up_gate,il,id);

    //updata table
    //table.mark_layer();

    //load
}
void custom_moe_unified::load_layer(uint32_t il, llama_pos target_pos){
    int n_expert = hparams.n_expert;
    //load
    load_data(pools.up,target_pos,nbyte_up_gate * n_expert,il,0);
    load_data(pools.gate,target_pos,nbyte_up_gate * n_expert,il,0);
    load_data(pools.down,target_pos,nbyte_down * n_expert,il,0);
    //update table
    table.mark_layer(il,expert_state::InMemory,target_pos);

}
uint32_t custom_moe_unified::get_padding() const {
    //TODO: check the padding Platform porting compatibility
    return 32u;
}