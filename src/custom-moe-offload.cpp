#include "custom-moe-offload.h"

#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
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
       layer_type(n_moe_layer),
       layer_state(n_moe_layer,OnDisk),
       n_row(n_moe_layer),
       n_col(n_expert){}


custom_moe_unified::custom_moe_unified(const llama_model & model,float utilization,const std::string & fname,llama_model_loader & ml) 
    : model(model),
      hparams(model.hparams),
      table(hparams.n_layer - hparams.n_layer_dense_lead, hparams.n_expert){

    uint32_t n_expert = hparams.n_expert;
    //select the last layer as typical expert
    ggml_tensor * classic_up = model.layers.back().ffn_up_exps;                 
    ggml_tensor * classic_gate = model.layers.back().ffn_gate_exps;
    ggml_tensor * classic_down = model.layers.back().ffn_down_exps;             

    GGML_ASSERT(classic_up->type == classic_gate->type);   //assert the gate is equal to up
    this->ne[0] = classic_up->ne[0];
    this->ne[1] = classic_up->ne[1];
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
    table_init(model,fname,ml);

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

}

uint32_t custom_moe_unified::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

void custom_moe_unified::table_init(const llama_model & model,const std::string & fname,llama_model_loader & ml){

    custom_expert_table & table = this->table;

    // Checking the dims of a two-dimensional table
    uint32_t n_row = table.experts.size();       //n_moe_layers
    uint32_t n_col = table.experts[0].size();    //n_experts
    uint32_t n_layer = hparams.n_layer;
    uint32_t n_dense = hparams.n_layer_dense_lead;
    GGML_ASSERT(n_row == (n_layer - n_dense));
    GGML_ASSERT(n_col == hparams.n_expert);
    //init files
    table.files.emplace_back(new llama_file(fname.c_str(), "rb"));

    std::map<ggml_type,size_t> type_map;
    auto type_for_size = [&](ggml_type type) -> size_t {
        auto it = type_map.find(type);
        if (it == type_map.end()) {
            size_t nbyte =(ne[0] * ne[1])* ggml_type_size(type) / ggml_blck_size(type);
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
        size_t a = type_for_size(up_type);
        size_t b = type_for_size(down_type);
        printf("%ld  %ld  \n",a,b);
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


void custom_moe_unified::prefill_init(){
    uint32_t n_slot_layer =  hparams.n_expert;
    uint32_t n_layer_load =  pool.n_slots / n_slot_layer;
    for(uint32_t il = 0; il < n_layer_load; il++){
        llama_pos slot_pos = pool_alloc(n_slot_layer);
        load_layer(il,slot_pos);
    }
    LLAMA_LOG_INFO("custom_moe prefill init: %d layers have been pre-load",n_layer_load);
}

void custom_moe_unified::check_layer(uint32_t il){
    if(table.layer_state.at(il) == expert_state::OnDisk){
        uint32_t prev_il = il -1;
        GGML_ASSERT(table.at(prev_il,LAYER_HEAD).state = expert_state::InMemory);

        //Reuse the pool of prev_layer for cur_layer
        llama_pos pos = table.at(prev_il,LAYER_HEAD).pos;
        //free the prev_layer in table
        table.mark_layer(prev_il,expert_state::OnDisk,-1);
        //load layer to pool
        load_layer(il,pos);
    } 
}

void custom_moe_unified::check_expert(uint32_t il,uint32_t id){
    if(table.at(il,id).state == expert_state::OnDisk){
        llama_pos pos = pool_alloc(1);
        load_expert(il,id,pos);
    }
}
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


void custom_moe_unified::load_data(llama_pos offset,uint32_t row, uint32_t col){
    std::vector<no_init<uint8_t>> read_buf;
    uint16_t    idx  = table.at(row,col).up.idx;
    GGML_ASSERT(table.at(row,col).gate.idx == idx && table.at(row,col).down.idx == idx);
    const auto &file = table.files.at(idx);

    size_t      up_offs = table.at(row,col).up.offs;
    size_t      gate_offs = table.at(row,col).gate.offs;
    size_t      down_offs = table.at(row,col).down.offs;
    //TBD: need create nbyte_expert vector  in (table_init)?
    if (ggml_backend_buffer_is_host(pool.up->buffer)) {
        file->seek(up_offs, SEEK_SET);
        file->read_raw(static_cast<char*>(pool.up->data) + offset * nbyte_slot_up, nbyte_slot_up);
        
        file->seek(gate_offs, SEEK_SET);
        file->read_raw(static_cast<char*>(pool.gate->data) + offset * nbyte_slot_gate, nbyte_slot_gate);
        file->seek(down_offs, SEEK_SET);
        file->read_raw(static_cast<char*>(pool.down->data) + offset * nbyte_slot_down, nbyte_slot_down);
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
                read_buf.resize( nbyte_slot_down);
                GGML_ASSERT(nbyte_slot_down > nbyte_slot_up && nbyte_slot_down > nbyte_slot_gate);
                file->seek(up_offs, SEEK_SET);
                file->read_raw(read_buf.data(),nbyte_slot_up);
                ggml_backend_tensor_set(pool.up, read_buf.data(), offset * nbyte_slot_up, nbyte_slot_up);
                file->seek(gate_offs, SEEK_SET);
                file->read_raw(read_buf.data(), nbyte_slot_gate);
                ggml_backend_tensor_set(pool.gate, read_buf.data(), offset * nbyte_slot_gate,nbyte_slot_gate);
                file->seek(down_offs, SEEK_SET);
                //TBD the n_size should be check
                file->read_raw(read_buf.data(), nbyte_slot_down);
                ggml_backend_tensor_set(pool.down, read_buf.data(), offset * nbyte_slot_down, nbyte_slot_down);
            }
    }

}

void custom_moe_unified::load_expert(uint32_t il, uint32_t id,llama_pos target_pos){
    //load
    load_data(target_pos,il,id);
    //update table
    table.mark(il,id,expert_state::InMemory,target_pos);
}


/*load all data with a layer
input:
    il:         the id of table_layer
    target_pos: the pos of pool load to
*/
void custom_moe_unified::load_layer(uint32_t il, llama_pos target_pos){        
    for(size_t i = 0; i < hparams.n_expert; i++){
        load_data(target_pos + i,il,i);
    }
    //update table
    table.mark_layer(il,expert_state::InMemory,target_pos);

}
uint32_t custom_moe_unified::get_padding() const {
    //TODO: check the padding Platform porting compatibility
    return 32u;
}
//
//custom op_kernel
//

// mapping the select_id to pos of pool
void id_pos_map(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth,  void * userdata){
    GGML_ASSERT(userdata != nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, a));
    GGML_ASSERT(a->ne[2] == a->ne[3]);
    GGML_ASSERT(a->ne[2] == 1);
    bool prefill = (a->ne[1] > 1);
    static std::once_flag init_flag;
    const int32_t       * a_data        = (int32_t *)ggml_get_data(a);
    int32_t             * dst_data      = (int32_t *)ggml_get_data(dst);
    custom_moe_unified  * moe_unified   = (custom_moe_unified *)userdata;

    //get the tensor's index of layer
    std::string a_name(a->name);
    auto it = moe_unified->name_layer_map.find(a_name);
    GGML_ASSERT(it != moe_unified->name_layer_map.end());
    uint32_t il = it->second;

    //thread safety
    std::call_once(init_flag, [moe_unified,il,prefill]() {
        if(prefill){
            moe_unified->check_layer(il);
        }
    });
    
    int n_ids   = a->ne[0];
    int n_token = a->ne[1];
    if(prefill){// parallel
        const int dr = (n_token + nth - 1) / nth;
        const int ie0 = dr * ith;
        const int ie1 = MIN(ie0 + dr, n_token);
        //check layer_state
        //TBD: bug!!parallel Conflicting Access
        //mapping
        for(int i = ie0; i < ie1; i++){
            for(int j = 0; j < n_ids; j++){

                dst_data[i*(dst->nb[1] / 4) +j] = moe_unified->id_map(il,a_data[i*(a->nb[1] / 4) + j]);
            }
        }
    } else{// no parallel
        if(ith == 0){
            for(int i = 0; i < n_ids; i++){
                //check
                moe_unified->check_expert(il,a_data[i]);
                //mapping
                dst_data[i] = moe_unified->id_map(il,a_data[i]);
            }
        }
    }

}