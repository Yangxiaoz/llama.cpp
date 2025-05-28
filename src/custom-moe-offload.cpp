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

custom_moe_unified::custom_moe_unified(const llama_model & model,ggml_backend_buffer_type_t buft,float utilization) 
    : table(model.hparams.n_layer - model.hparams.n_layer_dense_lead, model.hparams.n_expert),
      model(model),
      hparams(model.hparams){
    int32_t n_slots = 0;
    //select up as typical expert
    ggml_tensor * expert = model.layers.back().ffn_up_exps;//select the "up" of last layer as typical expert
    nbyte_layer_experts = ggml_nbytes(expert);
    nbyte_expert = nbyte_layer_experts / hparams.n_expert;
    size_t nbyte_expert_group = nbyte_expert * 3; // 3: up、gate、down
    type_expert =  expert->type;
    
    //TBD: need determin
    // ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    // check if it is possible to use buffer_from_host_ptr with this buffer type
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        // FIXME: workaround for CPU backend buft having a NULL device
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
    }
    auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
    if(!host_buft){
        host_buft = buft;
    }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);

    //TBD: temp for cpu, need to determine which buffer should be select
    // bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
    // bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

    int64_t n_byte_availa = (int64_t)(utilization * props.memory_free);
    n_slots  = n_byte_availa / nbyte_expert_group;

    groups.reserve(n_slots);

// create a context for  buffer 
    ggml_init_params params = {
        /*.mem_size   =*/ size_t(n_slots * 3 *ggml_tensor_overhead()), //3: up gate down
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
        if (!ctx) {
        throw std::runtime_error(format("%s: moe_pool_unified ctx alloc failed", __func__));
    }

    ctxs.emplace_back(ctx);
    
    //create moe_pool_tensors
    for(int i = 0; i < n_slots; i++){
        struct ffn_expert_group cur;

        cur.up = ggml_new_tensor_2d(ctx, type_expert, expert->ne[0], expert->ne[1]);
        cur.gate = ggml_new_tensor_2d(ctx, type_expert, expert->ne[0],expert->ne[1]);
        cur.down = ggml_new_tensor_2d(ctx, type_expert, expert->ne[0],expert->ne[1]);
        ggml_format_name(cur.up,   "%d_up", i);
        ggml_format_name(cur.gate, "%d_gate", i);
        ggml_format_name(cur.down, "%d_down", i);

        groups.push_back(cur);
    }

// allocate tensors and initialize the buffers to avoid NaNs in the padding
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
    if (!buf) {
        throw std::runtime_error("failed to allocate buffer for moe_pools");
    }
    

    {//TBD: need to determine if need
    // indicate that this buffer contains weights
    // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
        // ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    LLAMA_LOG_INFO("%s: %10s moe_expert_pool buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);


    bufs.emplace_back(buf);

}


size_t custom_moe_unified::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

void custom_moe_unified::clear() {

    // for (int32_t i = 0; i < (int32_t) size; ++i) {
    //     cells[i].pos = -1;
    //     cells[i].seq_id.clear();
    // }
    // head = 0;
    // used = 0;

    // for (auto & buf : bufs) {
    //     ggml_backend_buffer_clear(buf.get(), 0);
    // }
}

bool custom_moe_unified::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    // uint32_t new_head = size;

    // if (p0 < 0) {
    //     p0 = 0;
    // }

    // if (p1 < 0) {
    //     p1 = std::numeric_limits<llama_pos>::max();
    // }

    // for (uint32_t i = 0; i < size; ++i) {
    //     if (cells[i].pos >= p0 && cells[i].pos < p1) {
    //         if (seq_id < 0) {
    //             cells[i].seq_id.clear();
    //         } else if (cells[i].has_seq_id(seq_id)) {
    //             cells[i].seq_id.erase(seq_id);
    //         } else {
    //             continue;
    //         }
    //         if (cells[i].is_empty()) {
    //             // keep count of the number of used cells
    //             if (cells[i].pos >= 0) {
    //                 used--;
    //             }

    //             cells[i].pos = -1;

    //             if (new_head == size) {
    //                 new_head = i;
    //             }
    //         }
    //     }
    // }

    // // If we freed up a slot, set head to it so searching can start there.
    // if (new_head != size && new_head < head) {
    //     head = new_head;
    // }

    return true;
}

void custom_moe_unified::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(seq_id_src);
    GGML_UNUSED(seq_id_dst);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    // if (seq_id_src == seq_id_dst) {
    //     return;
    // }

    // if (p0 < 0) {
    //     p0 = 0;
    // }

    // if (p1 < 0) {
    //     p1 = std::numeric_limits<llama_pos>::max();
    // }

    // // otherwise, this is the KV of a Transformer-like model
    // head = 0;

    // for (uint32_t i = 0; i < size; ++i) {
    //     if (cells[i].has_seq_id(seq_id_src) && cells[i].pos >= p0 && cells[i].pos < p1) {
    //         cells[i].seq_id.insert(seq_id_dst);
    //     }
    // }
}

void custom_moe_unified::seq_keep(llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);
    // uint32_t new_head = size;

    // for (uint32_t i = 0; i < size; ++i) {
    //     if (!cells[i].has_seq_id(seq_id)) {
    //         if (cells[i].pos >= 0) {
    //             used--;
    //         }

    //         cells[i].pos = -1;
    //         cells[i].seq_id.clear();

    //         if (new_head == size){
    //             new_head = i;
    //         }
    //     } else {
    //         cells[i].seq_id.clear();
    //         cells[i].seq_id.insert(seq_id);
    //     }
    // }

    // // If we freed up a slot, set head to it so searching can start there.
    // if (new_head != size && new_head < head) {
    //     head = new_head;
    // }
}

void custom_moe_unified::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(delta);
    // if (delta == 0) {
    //     return;
    // }

    // uint32_t new_head = size;

    // if (p0 < 0) {
    //     p0 = 0;
    // }

    // if (p1 < 0) {
    //     p1 = std::numeric_limits<llama_pos>::max();
    // }

    // // If there is no range then return early to avoid looping over the
    // if (p0 == p1) {
    //     return;
    // }

    // for (uint32_t i = 0; i < size; ++i) {
    //     if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
    //         has_shift = true;
    //         cells[i].pos   += delta;
    //         cells[i].delta += delta;

    //         if (cells[i].pos < 0) {
    //             if (!cells[i].is_empty()) {
    //                 used--;
    //             }
    //             cells[i].pos = -1;
    //             cells[i].seq_id.clear();
    //             if (new_head == size) {
    //                 new_head = i;
    //             }
    //         }
    //     }
    // }

    // // If we freed up a slot, set head to it so searching can start there.
    // // Otherwise we just start the next search from the beginning.
    // head = new_head != size ? new_head : 0;
}

void custom_moe_unified::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(d);
    // if (d == 1) {
    //     return;
    // }

    // if (p0 < 0) {
    //     p0 = 0;
    // }

    // if (p1 < 0) {
    //     p1 = std::numeric_limits<llama_pos>::max();
    // }

    // // If there is no range then return early to avoid looping over the cache.
    // if (p0 == p1) {
    //     return;
    // }

    // for (uint32_t i = 0; i < size; ++i) {
    //     if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
    //         has_shift = true;

    //         {
    //             llama_pos p_old = cells[i].pos;
    //             cells[i].pos   /= d;
    //             cells[i].delta += cells[i].pos - p_old;
    //         }
    //     }
    // }
}

llama_pos custom_moe_unified::seq_pos_max(llama_seq_id seq_id) const {
    GGML_UNUSED(seq_id);

    llama_pos result = 0;

    // for (uint32_t i = 0; i < size; ++i) {
    //     if (cells[i].has_seq_id(seq_id)) {
    //         result = std::max(result, cells[i].pos);
    //     }
    // }

    return result;
}



bool custom_moe_unified::get_can_edit() const {
    return false;
}