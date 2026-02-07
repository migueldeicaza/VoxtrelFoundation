#include "voxtral_metal.h"

#if !defined(VOXTRAL_ENABLE_METAL)

int vox_metal_init(void) {
    return 0;
}

int vox_metal_available(void) {
    return 0;
}

void vox_metal_shutdown(void) {
}

void vox_metal_sgemm_bf16(int M, int N, int K,
                           const float *A,
                           const uint16_t *B_bf16,
                           float *C) {
    (void)M; (void)N; (void)K; (void)A; (void)B_bf16; (void)C;
}

void vox_metal_sgemm(int M, int N, int K,
                     const float *A,
                     const float *B,
                     float *C) {
    (void)M; (void)N; (void)K; (void)A; (void)B; (void)C;
}

void vox_metal_fused_qkv_bf16(int M, int K,
                                const float *input,
                                const uint16_t *wq_bf16, int Nq,
                                const uint16_t *wk_bf16, int Nk,
                                const uint16_t *wv_bf16, int Nv,
                                float *q, float *k, float *v) {
    (void)M; (void)K; (void)input; (void)wq_bf16; (void)Nq; (void)wk_bf16;
    (void)Nk; (void)wv_bf16; (void)Nv; (void)q; (void)k; (void)v;
}

void vox_metal_fused_norm_qkv_bf16(int M, int K,
                                     const float *x,
                                     const float *norm_weight, float eps,
                                     const uint16_t *wq_bf16, int Nq,
                                     const uint16_t *wk_bf16, int Nk,
                                     const uint16_t *wv_bf16, int Nv,
                                     float *q, float *k, float *v) {
    (void)M; (void)K; (void)x; (void)norm_weight; (void)eps; (void)wq_bf16;
    (void)Nq; (void)wk_bf16; (void)Nk; (void)wv_bf16; (void)Nv; (void)q;
    (void)k; (void)v;
}

void vox_metal_fused_ffn_bf16(int M, int dim, int hidden,
                               const float *input,
                               const uint16_t *w1_bf16,
                               const uint16_t *w3_bf16,
                               const uint16_t *w2_bf16,
                               float *output) {
    (void)M; (void)dim; (void)hidden; (void)input; (void)w1_bf16;
    (void)w3_bf16; (void)w2_bf16; (void)output;
}

void vox_metal_fused_wo_ffn_bf16(int M, int dim, int q_dim, int hidden,
                                   float *x,
                                   const float *attn_out,
                                   const uint16_t *wo_bf16,
                                   const float *ffn_norm, float eps,
                                   const float *ada_scale,
                                   const uint16_t *w1_bf16,
                                   const uint16_t *w3_bf16,
                                   const uint16_t *w2_bf16) {
    (void)M; (void)dim; (void)q_dim; (void)hidden; (void)x; (void)attn_out;
    (void)wo_bf16; (void)ffn_norm; (void)eps; (void)ada_scale; (void)w1_bf16;
    (void)w3_bf16; (void)w2_bf16;
}

void vox_metal_batched_attention(float *out,
                                  const float *Q, const float *K, const float *V,
                                  int seq_q, int seq_k,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, float scale,
                                  int window_size, int q_offset) {
    (void)out; (void)Q; (void)K; (void)V; (void)seq_q; (void)seq_k;
    (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)scale;
    (void)window_size; (void)q_offset;
}

void vox_metal_encoder_attention(float *out,
                                   const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, float scale,
                                   int window_size, int q_offset) {
    (void)out; (void)Q; (void)K; (void)V; (void)seq_q; (void)seq_k;
    (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)scale;
    (void)window_size; (void)q_offset;
}

int vox_metal_fused_logits_bf16(int dim, int vocab_size,
                                  const float *x,
                                  const float *norm_weight, float eps,
                                  const uint16_t *tok_emb_bf16,
                                  float *logits_out) {
    (void)dim; (void)vocab_size; (void)x; (void)norm_weight; (void)eps;
    (void)tok_emb_bf16; (void)logits_out;
    return 0;
}

void vox_metal_decoder_start(const float *x, int dim) {
    (void)x; (void)dim;
}

void vox_metal_decoder_end(void) {
}

void vox_metal_decoder_norm_qkv(int K,
                                  const float *norm_weight, float eps,
                                  const uint16_t *wq_bf16, int Nq,
                                  const uint16_t *wk_bf16, int Nk,
                                  const uint16_t *wv_bf16, int Nv,
                                  float *q, float *k, float *v) {
    (void)K; (void)norm_weight; (void)eps; (void)wq_bf16; (void)Nq;
    (void)wk_bf16; (void)Nk; (void)wv_bf16; (void)Nv; (void)q; (void)k; (void)v;
}

void vox_metal_decoder_wo_ffn_next_qkv(int dim, int q_dim, int hidden,
                                         const float *attn_out,
                                         const uint16_t *wo_bf16,
                                         const float *ffn_norm, float eps,
                                         const float *ada_scale,
                                         const uint16_t *w1_bf16,
                                         const uint16_t *w3_bf16,
                                         const uint16_t *w2_bf16,
                                         const float *next_attn_norm,
                                         const uint16_t *next_wq_bf16, int next_Nq,
                                         const uint16_t *next_wk_bf16, int next_Nk,
                                         const uint16_t *next_wv_bf16, int next_Nv,
                                         float *q, float *k, float *v) {
    (void)dim; (void)q_dim; (void)hidden; (void)attn_out; (void)wo_bf16;
    (void)ffn_norm; (void)eps; (void)ada_scale; (void)w1_bf16; (void)w3_bf16;
    (void)w2_bf16; (void)next_attn_norm; (void)next_wq_bf16; (void)next_Nq;
    (void)next_wk_bf16; (void)next_Nk; (void)next_wv_bf16; (void)next_Nv;
    (void)q; (void)k; (void)v;
}

int vox_metal_decoder_wo_ffn_logits(int dim, int q_dim, int hidden, int vocab_size,
                                      const float *attn_out,
                                      const uint16_t *wo_bf16,
                                      const float *ffn_norm, float eps,
                                      const float *ada_scale,
                                      const uint16_t *w1_bf16,
                                      const uint16_t *w3_bf16,
                                      const uint16_t *w2_bf16,
                                      const float *final_norm,
                                      const uint16_t *tok_emb_bf16,
                                      float *logits_out) {
    (void)dim; (void)q_dim; (void)hidden; (void)vocab_size; (void)attn_out;
    (void)wo_bf16; (void)ffn_norm; (void)eps; (void)ada_scale; (void)w1_bf16;
    (void)w3_bf16; (void)w2_bf16; (void)final_norm; (void)tok_emb_bf16;
    (void)logits_out;
    return 0;
}

void *vox_metal_shared_alloc(size_t size) {
    (void)size;
    return NULL;
}

void vox_metal_shared_free(void *ptr) {
    (void)ptr;
}

int vox_metal_decoder_full_step(void *ctx, const float *rope_freqs, float *logits) {
    (void)ctx; (void)rope_freqs; (void)logits;
    return 0;
}

int vox_metal_encoder_full_step(void *ctx, float *x, int new_len,
                                const float *rope_freqs, int cache_len) {
    (void)ctx; (void)x; (void)new_len; (void)rope_freqs; (void)cache_len;
    return 0;
}

void vox_metal_decoder_prefill_step(void *ctx, float *x, int seq_len,
                                    const float *rope_freqs) {
    (void)ctx; (void)x; (void)seq_len; (void)rope_freqs;
}

void vox_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    (void)bf16_weights; (void)num_elements;
}

void vox_metal_warmup_decoder_ops(void *ctx) {
    (void)ctx;
}

void vox_metal_warmup_merged_2(const uint16_t *a, size_t a_n,
                              const uint16_t *b, size_t b_n) {
    (void)a; (void)a_n; (void)b; (void)b_n;
}

void vox_metal_warmup_merged_3(const uint16_t *a, size_t a_n,
                              const uint16_t *b, size_t b_n,
                              const uint16_t *c, size_t c_n) {
    (void)a; (void)a_n; (void)b; (void)b_n; (void)c; (void)c_n;
}

size_t vox_metal_memory_used(void) {
    return 0;
}

#endif
