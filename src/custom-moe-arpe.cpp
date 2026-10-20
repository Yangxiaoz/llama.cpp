#include "custom-moe-arpe.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// counts file: header {magic, version, n_layer, n_expert, n_used}, then N [(L-1)*E] and C [(L-1)*E*E] as double
static constexpr uint32_t ARPE_COUNTS_MAGIC   = 0x45505241; // "ARPE"
static constexpr uint32_t ARPE_COUNTS_VERSION = 1;

// rho^d underflows to 0 in double long before this, so the table is exact
static constexpr int64_t  ARPE_RHO_POW_SIZE   = 512;

enum arpe_counts_status { ARPE_COUNTS_ABSENT, ARPE_COUNTS_OK, ARPE_COUNTS_MISMATCH };

custom_evict_arpe::custom_evict_arpe(uint32_t n_layer, int32_t n_expert, int32_t n_used, const std::string & affinity_path, const std::string & calib_file)
    : L(n_layer),
      E(n_expert),
      K(n_used),
      hp(),
      calib_path(calib_file) {
    GGML_ASSERT(L >= 1 && E > 0 && 0 < K && K <= E);
    const size_t n_pair_layer = L - 1;

    // A0 = (C + 1) / (N + E/K): every row sums to K, no counts = uniform K/E
    std::vector<double> C;
    std::vector<double> N;
    int status = ARPE_COUNTS_ABSENT;
    if (!affinity_path.empty()) {
        status = load_counts(affinity_path, C, N);
        if (status == ARPE_COUNTS_ABSENT) {
            LLAMA_LOG_WARN("%s: ARPE affinity file %s not found, using a uniform prior\n", __func__, affinity_path.c_str());
        }
    }
    const bool has_counts = status == ARPE_COUNTS_OK;
    A0.resize(n_pair_layer * E * E);
    double n_pairs = 0.0;
    for (size_t l = 0; l < n_pair_layer; l++) {
        for (int32_t i = 0; i < E; i++) {
            const size_t row = l * E + i;
            const double n   = has_counts ? N[row] : 0.0;
            n_pairs += n;
            for (int32_t j = 0; j < E; j++) {
                const double c = has_counts ? C[row * E + j] : 0.0;
                A0[row * E + j] = (float) ((c + 1.0) / (n + (double) E / K));
            }
        }
    }
    if (has_counts) {
        LLAMA_LOG_INFO("%s: ARPE affinity prior from %s (%.0f pairs)\n", __func__, affinity_path.c_str(), n_pairs);
    } else {
        LLAMA_LOG_INFO("%s: ARPE uniform affinity prior\n", __func__);
    }

    if (!calib_path.empty()) {
        cal_C.assign(n_pair_layer * E * E, 0.0);
        cal_N.assign(n_pair_layer * E, 0.0);
    }

    rho_pow.resize(ARPE_RHO_POW_SIZE);
    for (int64_t d = 0; d < ARPE_RHO_POW_SIZE; d++) {
        rho_pow[d] = std::pow(hp.rho, (double) d);
    }

    reset_request();
}

custom_evict_arpe::~custom_evict_arpe() {
    if (n_victim > 0) {
        LLAMA_LOG_INFO("custom_moe ARPE: %llu decode evictions: %.1f%% current layer, %.1f%% next layer, %.1f%% previous layer\n",
            (unsigned long long) n_victim,
            100.0 * n_victim_cur  / n_victim,
            100.0 * n_victim_next / n_victim,
            100.0 * n_victim_prev / n_victim);
    }
    if (calib_path.empty()) {
        return;
    }
    // add the counts of this run to the file
    std::vector<double> C;
    std::vector<double> N;
    const int status = load_counts(calib_path, C, N);
    if (status == ARPE_COUNTS_MISMATCH) {
        LLAMA_LOG_WARN("custom_moe ARPE: %s belongs to another model, calibration counts not written\n", calib_path.c_str());
        return;
    }
    if (status == ARPE_COUNTS_ABSENT) {
        C.assign(cal_C.size(), 0.0);
        N.assign(cal_N.size(), 0.0);
    }
    double n_run   = 0.0;
    double n_total = 0.0;
    for (size_t k = 0; k < N.size(); k++) {
        N[k]    += cal_N[k];
        n_run   += cal_N[k];
        n_total += N[k];
    }
    for (size_t k = 0; k < C.size(); k++) {
        C[k] += cal_C[k];
    }
    save_counts(calib_path, C, N);
    LLAMA_LOG_INFO("custom_moe ARPE: calibration counts written to %s (%.0f pairs from this run, %.0f in total)\n", calib_path.c_str(), n_run, n_total);
}

void custom_evict_arpe::reset_request() {
    const size_t n_pair_layer = L - 1;
    cnt.assign((size_t) L * E, 0);
    seen.assign(L, 0);
    last.assign((size_t) L * E, NEVER);
    prior.assign((size_t) L * E, (double) K / E);
    pc.assign((size_t) L * E, 0);
    pc_total.assign(L, 0);
    Co.assign(n_pair_layer * E * E, 0);
    No.assign(n_pair_layer * E, 0);
    prev_sel.clear();
    n_prev     = 0;
    prev_layer = -1;
    had_decode = false;
    p_valid    = false;
}

void custom_evict_arpe::layer_begin(uint32_t il, const std::vector<int32_t> & ids, bool prefill) {
    GGML_ASSERT(il < L);
    if (prefill && il == 0 && had_decode) {
        // single request mode (guide 11.4): a new prompt after a decode is a new request
        reset_request();
    }
    cur_layer = il;
    locked    = ids;
    observed  = false;
    p_valid   = false;
}

void custom_evict_arpe::pair(uint32_t l, const int32_t * a, const int32_t * b) {
    for (int32_t k = 0; k < K; k++) {
        const size_t row = (size_t) l * E + a[k];
        No[row]++;
        if (!cal_N.empty()) {
            cal_N[row] += 1.0;
        }
        for (int32_t kk = 0; kk < K; kk++) {
            Co[row * E + b[kk]]++;
            if (!cal_C.empty()) {
                cal_C[row * E + b[kk]] += 1.0;
            }
        }
    }
}

void custom_evict_arpe::observe(uint32_t il, const int32_t * ids, int64_t n_used, int64_t n_tokens, int64_t stride, bool prefill) {
    GGML_ASSERT(il == cur_layer && n_used == K);
    if (n_tokens <= 0) {
        return;
    }
    const size_t base = (size_t) il * E;
    // the pairs come from the same token at two adjacent layers, never across tokens (guide 13.5)
    const bool has_prev = il > 0 && prev_layer == (int32_t) il - 1;

    if (prefill) {
        // prior from all prompt tokens, summed over the ubatches of the prompt
        for (int64_t r = 0; r < n_tokens; r++) {
            for (int32_t k = 0; k < K; k++) {
                pc[base + ids[r*stride + k]]++;
            }
        }
        pc_total[il] += (uint64_t) n_tokens * K;
        if (hp.prior) {
            for (int32_t e = 0; e < E; e++) {
                prior[base + e] = K * (pc[base + e] + 1.0) / (pc_total[il] + (double) E);
            }
        }
        // only the last prompt token counts as decode token -1 (guide 13.4)
        std::fill(last.begin() + base, last.begin() + base + E, NEVER);
        for (int32_t k = 0; k < K; k++) {
            last[base + ids[(n_tokens - 1)*stride + k]] = -1;
        }
        // the last layer may see only the output tokens: align the last n rows
        if (has_prev) {
            const int64_t n = std::min(n_tokens, n_prev);
            for (int64_t r = 0; r < n; r++) {
                pair(il - 1, prev_sel.data() + (n_prev - n + r) * K, ids + (n_tokens - n + r) * stride);
            }
        }
    } else {
        GGML_ASSERT(n_tokens == 1);
        if (has_prev) {
            pair(il - 1, prev_sel.data() + (n_prev - 1) * K, ids);
        }
        const int64_t t = seen[il];
        for (int32_t k = 0; k < K; k++) {
            cnt[base + ids[k]]++;
            last[base + ids[k]] = t;
        }
        seen[il]++;
        had_decode = true;
        // the prefetch of the next layer evicts with this token already counted (guide 6.3)
        observed = true;
    }

    // keep the routing for the pairing with the next layer
    prev_sel.resize((size_t) n_tokens * K);
    for (int64_t r = 0; r < n_tokens; r++) {
        std::copy(ids + r*stride, ids + r*stride + K, prev_sel.begin() + r*K);
    }
    n_prev     = n_tokens;
    prev_layer = (int32_t) il;
    p_valid    = false;
}

void custom_evict_arpe::compute_p() {
    const int64_t c = cur_layer;
    // the token being decoded, before (observed = false) or after (observed = true) observe() of layer c
    const int64_t t = (int64_t) seen[c] - (observed ? 1 : 0);

    p.resize((size_t) L * E);
    for (int64_t l = 0; l < L; l++) {
        // layers up to c are done for this token, their next visit is the next token
        const int64_t next_tok = t + (l <= c ? 1 : 0);
        const double  denom    = seen[l] + hp.W0;
        for (int32_t e = 0; e < E; e++) {
            const size_t k = (size_t) l * E + e;
            double f = (cnt[k] + hp.W0 * prior[k]) / denom;
            if (last[k] != NEVER) {
                const int64_t d = std::max<int64_t>(next_tok - last[k] - 1, 0);
                if (d < ARPE_RHO_POW_SIZE) {
                    f += hp.alpha * rho_pow[d];
                }
            }
            p[k] = f;
        }
    }

    // affinity: the experts of layer c predict layer c+1, K x E work
    if (hp.lam > 0.0 && c + 1 < L) {
        std::vector<double> q(E, 0.0);
        for (int32_t i : locked) {
            const size_t row = (size_t) c * E + i;
            const double inv = 1.0 / (No[row] + hp.w_aff);
            for (int32_t j = 0; j < E; j++) {
                const double a = hp.online ? (Co[row * E + j] + hp.w_aff * A0[row * E + j]) * inv : (double) A0[row * E + j];
                q[j] += a / K;
            }
        }
        const size_t next = (size_t) (c + 1) * E;
        for (int32_t j = 0; j < E; j++) {
            p[next + j] = hp.lam * q[j] + (1.0 - hp.lam) * p[next + j];
        }
    }
    p_valid = true;
}

bool custom_evict_arpe::victim(const custom_evictable & can_evict, uint32_t & il, int32_t & id) {
    if (!p_valid) {
        compute_p();
    }
    const int64_t c      = cur_layer;
    int64_t       best_l = -1;
    int32_t       best_e = -1;
    double        best   = 0.0;
    // row-major scan with a strict <: ties go to the smallest (l, e) (guide 8.2)
    for (int64_t l = 0; l < L; l++) {
        // tie-break: prefer the layer visited last, the current layer counts as the farthest
        const int64_t dist = l == c ? L : (l - c + L) % L;
        for (int32_t e = 0; e < E; e++) {
            if (!can_evict((uint32_t) l, e)) {
                continue;
            }
            const double score = p[(size_t) l * E + e] - 1e-9 * dist;
            if (best_l < 0 || score < best) {
                best_l = l;
                best_e = e;
                best   = score;
            }
        }
    }
    if (best_l < 0) {
        return false;
    }
    il = (uint32_t) best_l;
    id = best_e;

    n_victim++;
    if (best_l == c) {
        n_victim_cur++;
    } else if (best_l == (c + 1) % L) {
        n_victim_next++;
    } else if (best_l == (c + L - 1) % L) {
        n_victim_prev++;
    }
    return true;
}

int custom_evict_arpe::load_counts(const std::string & path, std::vector<double> & C, std::vector<double> & N) const {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return ARPE_COUNTS_ABSENT;
    }
    uint32_t hdr[5] = {};
    bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 &&
              hdr[0] == ARPE_COUNTS_MAGIC && hdr[1] == ARPE_COUNTS_VERSION &&
              hdr[2] == L && hdr[3] == (uint32_t) E && hdr[4] == (uint32_t) K;
    if (ok) {
        N.resize((size_t) (L - 1) * E);
        C.resize((size_t) (L - 1) * E * E);
        ok = fread(N.data(), sizeof(double), N.size(), f) == N.size() &&
             fread(C.data(), sizeof(double), C.size(), f) == C.size();
    }
    fclose(f);
    if (!ok) {
        LLAMA_LOG_WARN("custom_moe ARPE: %s is not a counts file of this model (%u layers, %d experts, top-%d)\n", path.c_str(), L, E, K);
        return ARPE_COUNTS_MISMATCH;
    }
    return ARPE_COUNTS_OK;
}

void custom_evict_arpe::save_counts(const std::string & path, const std::vector<double> & C, const std::vector<double> & N) const {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LLAMA_LOG_WARN("custom_moe ARPE: cannot write %s\n", path.c_str());
        return;
    }
    const uint32_t hdr[5] = { ARPE_COUNTS_MAGIC, ARPE_COUNTS_VERSION, L, (uint32_t) E, (uint32_t) K };
    const bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1 &&
                    fwrite(N.data(), sizeof(double), N.size(), f) == N.size() &&
                    fwrite(C.data(), sizeof(double), C.size(), f) == C.size();
    fclose(f);
    if (!ok) {
        LLAMA_LOG_WARN("custom_moe ARPE: write error on %s\n", path.c_str());
    }
}
