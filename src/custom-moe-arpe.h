#pragma once

#include "custom-moe-offload.h"

#include <cstdint>
#include <string>
#include <vector>

//
// ARPE: affinity-aware reuse-probability eviction (examples/moe-offload/ARPE_施工指南.md)
//
// decode victim = the evictable expert whose layer, at its next visit, is least likely to select it:
//   p(l,e) = (cnt + W0*prior) / (seen + W0) + alpha * rho^(next_tok - last - 1)        history, every layer
//   p(c+1,j) = lam * q[j] + (1 - lam) * p(c+1,j)                                       affinity, next layer only
// the statistics are per request (reset when a prefill follows a decode); A0 comes from the calibration counts
//

struct custom_arpe_params {
    double W0       = 40.0; // weight of the prefill prior, in decode tokens
    double alpha    = 0.16; // recency bonus of an expert used by the previous token
    double rho      = 0.1;  // recency decay per token
    double lam      = 0.6;  // weight of the affinity prediction for the next layer
    double w_aff    = 20.0; // weight of the offline affinity prior A0, in online pairs
    bool   online   = true; // online affinity counts Co/No of the request
    bool   prior    = true; // prefill prior (else K/E)
};

class custom_evict_arpe : public custom_evict_policy {
public:
    // affinity_path: counts file for A0 ("" = uniform), calib_file: add the counts of this run to this file at exit ("" = off)
    custom_evict_arpe(uint32_t n_layer, int32_t n_expert, int32_t n_used, const std::string & affinity_path, const std::string & calib_file);
    ~custom_evict_arpe() override;

    // residency is read through can_evict (table state), nothing to track here
    void insert(uint32_t, int32_t) override {}
    void use   (uint32_t, int32_t) override {}
    void erase (uint32_t, int32_t) override {}

    bool victim(const custom_evictable & can_evict, uint32_t & il, int32_t & id) override;

    void layer_begin(uint32_t il, const std::vector<int32_t> & ids, bool prefill) override;
    void observe(uint32_t il, const int32_t * ids, int64_t n_used, int64_t n_tokens, int64_t stride, bool prefill) override;
    // only the experts in use are locked, the rest of the current layer competes like any other layer (guide 8.1)
    bool lock_layer() const override { return false; }

private:
    static constexpr int64_t NEVER = INT64_MIN;

    const uint32_t              L;
    const int32_t               E;
    const int32_t               K;
    const custom_arpe_params    hp;
    const std::string           calib_path;

    // global: A0[l][i][j], l < L-1, smoothed from the calibration counts
    std::vector<float>          A0;

    // per request
    std::vector<uint32_t>       cnt;      // [L][E] decode selections
    std::vector<uint32_t>       seen;     // [L]    decode tokens done
    std::vector<int64_t>        last;     // [L][E] last decode token that selected the expert, -1 = last prompt token
    std::vector<double>         prior;    // [L][E]
    std::vector<uint64_t>       pc;       // [L][E] prefill selections, summed over the ubatches of the prompt
    std::vector<uint64_t>       pc_total; // [L]
    std::vector<uint32_t>       Co;       // [L-1][E][E] online pairs (layer l selects i, layer l+1 selects j, same token)
    std::vector<uint32_t>       No;       // [L-1][E]
    std::vector<int32_t>        prev_sel; // routing of layer prev_layer, n_prev rows of K ids
    int64_t                     n_prev     = 0;
    int32_t                     prev_layer = -1;
    bool                        had_decode = false; // a prefill after a decode starts a new request

    // calibration counts of this run (all requests), only with calib_path
    std::vector<double>         cal_C;
    std::vector<double>         cal_N;

    // the layer being computed
    uint32_t                    cur_layer = 0;
    std::vector<int32_t>        locked;   // S_c, the experts used by cur_layer
    bool                        observed  = false; // observe() of cur_layer already ran (prefetch of the next layer)

    // p of the last victim() call, valid until the layer or the statistics change
    std::vector<double>         p;
    bool                        p_valid = false;
    std::vector<double>         rho_pow;  // rho^d

    // eviction distribution (guide 12.3), by layer distance from cur_layer
    uint64_t                    n_victim      = 0;
    uint64_t                    n_victim_cur  = 0; // cur_layer (experts not used by this token)
    uint64_t                    n_victim_next = 0; // cur_layer + 1
    uint64_t                    n_victim_prev = 0; // cur_layer - 1

    void reset_request();
    void pair(uint32_t l, const int32_t * a, const int32_t * b); // layer l row a, layer l+1 row b
    void compute_p();
    // returns an arpe_counts_status (custom-moe-arpe.cpp)
    int  load_counts(const std::string & path, std::vector<double> & C, std::vector<double> & N) const;
    void save_counts(const std::string & path, const std::vector<double> & C, const std::vector<double> & N) const;
};
