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


custom_expert_table::custom_expert_table(int n_moe_layer, int n_expert)
    :  experts(n_moe_layer,std::vector<custom_expert_group>(n_expert)),
       n_row(n_moe_layer),
       n_col(n_expert){}


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
        size_t nbyte_expert = this->nbyte_expert;
        for(uint32_t j = 0; j < n_col; j++){
            table.at(i,j).state    = initial_state;
            table.at(i,j).pos      = -1;

            table.at(i,j).up.idx   = weight_ups->idx; 
            table.at(i,j).up.offs  = weight_ups->offs + j* nbyte_expert;

            table.at(i,j).gate.idx = weight_gates->idx;
            table.at(i,j).gate.offs= weight_gates->offs +j* nbyte_expert;

            table.at(i,j).down.idx = weight_downs->idx;
            table.at(i,j).down.offs= weight_downs->offs +j* nbyte_expert;
        }
    }
    
    
}

custom_moe_unified::custom_moe_unified(const llama_model & model,float utilization,const std::string & fname,llama_model_loader & ml) 
    : table(model.hparams.n_layer - model.hparams.n_layer_dense_lead, model.hparams.n_expert),
      model(model),
      hparams(model.hparams){
    //select up as typical expert
    ggml_tensor * classic_expert = model.layers.back().ffn_up_exps;//select the "up" of last layer as typical expert

    this->nbyte_expert = ggml_nbytes(classic_expert) / hparams.n_expert;
    this->type_expert = classic_expert->type;

    //init the table
    table_init(model,fname,ml);

    //init the moe_pools
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
    // const auto padding = get_padding();

    uint32_t n_byte_grpup = nbyte_expert * 3;// 3: up、gate、down
    int32_t n_slots = 0;
    n_slots  = n_byte_availa / n_byte_grpup;
    // uint64_t n_size = M_PAD((n_byte_availa/3),padding);

    // cells.resize(n_slots);

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
    up = ggml_new_tensor_3d(ctx, type_expert, classic_expert->ne[0], classic_expert->ne[1],n_slots);
    gate = ggml_new_tensor_3d(ctx, type_expert, classic_expert->ne[0],classic_expert->ne[1],n_slots);
    down = ggml_new_tensor_3d(ctx, type_expert, classic_expert->ne[0],classic_expert->ne[1],n_slots);
    ggml_format_name(up,   "pool_ups");
    ggml_format_name(gate, "pool_gate");
    ggml_format_name(down, "pool_downs");
    //TBD: need to get size of pool and check the padding

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

uint32_t custom_moe_unified::get_padding() const {
    //TODO: check the padding Platform porting compatibility
    return 32u;
}

uint32_t custom_moe_unified::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}
