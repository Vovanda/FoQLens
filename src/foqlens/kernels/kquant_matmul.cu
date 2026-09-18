// A linear layer read straight from its k-quant copy (refinements.KRefinedWeight): a row of weights is fetched only
// to the depth its block of rows is read at, so the bytes that cross the memory bus follow the layout (issue #3).
//
// The copy is the base as ggml blocks (block_q2_K / block_q4_K, kquant.gguf_blocks) and refinement planes of 2-bit
// codes, four to a byte, the first code in the low bits. A weight is built exactly as the torch path builds it -
// base = d*scale*code - dmin*min, then + step/4^k * (code - 1.5) per refinement, every operation rounded on its own
// (no fused multiply-add) - and rounded to bf16 before it meets the input, as F.linear on the bf16 weight does. Only
// the order of the sum over the input differs from cuBLAS.
//
// A warp owns one output row; its lanes split it by format blocks (16 weights of Q2_K, 32 of Q4_K) and add their
// partial sums at the end. The warps of a thread block - rows of one block of TILE_ROWS, so of one depth - share the
// tokens' inputs, loaded into shared memory a chunk of columns at a time. A lane loads a block's codes, each refinement
// plane and the inputs as whole words. A warp carries BATCH_TILE tokens at once, each read at its own depth: depth 0
// is ZERO and reads nothing. One kernel per format, so the block size and the base depth are constants and the loops
// unroll.

#include <cuda_fp16.h>

#define QK_K 256
#define TILE_ROWS 64         // a FoQLens block of rows: the unit a depth is set on
#define BATCH_TILE 8         // tokens a warp carries at once
#define X_CHUNK 2048         // columns of the inputs a thread block holds in shared memory at once: 32 KiB for 8 tokens
#define WARP 32
#define MAX_REFINEMENTS 3    // refinements above a Q2_K base to D8; a Q4_K base holds 2 and reaches D8 with 2
#define QS_OFFSET 16         // the codes follow 16 bytes in both blocks: scales of Q2_K; d, dmin and scales of Q4_K

__device__ __forceinline__ float from_bf16(unsigned int bits) { return __uint_as_float(bits << 16); }

// float -> bf16 -> float, round to nearest even, as torch converts a finite float
__device__ __forceinline__ float round_bf16(float value) {
  unsigned int u = __float_as_uint(value);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return __uint_as_float(u & 0xFFFF0000u);
}

__device__ __forceinline__ float from_fp16(const unsigned char* p) {
  return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}

// A 16-byte-aligned read of eight bf16 inputs.
__device__ __forceinline__ void load8(const unsigned short* p, float* out) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  const unsigned int words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    out[2 * k] = from_bf16(words[k] & 0xFFFFu);
    out[2 * k + 1] = from_bf16(words[k] >> 16);
  }
}

// The scale, min and the block's codes (one byte per weight, as the torch path unpacks them).
template <bool Q2, int BLOCK, int BYTES>
__device__ __forceinline__ void decode(const unsigned char* sb, int j, float& step, float& offset, unsigned char* codes) {
  int scale, minimum;
  float d, dmin;
  if (Q2) {
    scale = sb[j] & 0xF;
    minimum = sb[j] >> 4;
    d = from_fp16(sb + 80);
    dmin = from_fp16(sb + 82);
  } else {
    const unsigned char* s = sb + 4;
    if (j < 4) {
      scale = s[j] & 63;
      minimum = s[j + 4] & 63;
    } else {
      scale = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
      minimum = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
    d = from_fp16(sb);
    dmin = from_fp16(sb + 2);
  }
  step = __fmul_rn(d, (float)scale);
  offset = __fmul_rn(dmin, (float)minimum);
  const unsigned char* qs = sb + QS_OFFSET;
  if (Q2) {
    // weights 16j .. 16j+15 of a super-block: bytes 32*half + (16j mod 32) ..., shifted by 2*group
    const int i0 = j * BLOCK;
    const unsigned int* words = reinterpret_cast<const unsigned int*>(qs + 32 * (i0 >> 7) + (i0 & 31));
    const int shift = 2 * ((i0 >> 5) & 3);
#pragma unroll
    for (int w = 0; w < BLOCK / 4; ++w) {
      const unsigned int word = words[w];
#pragma unroll
      for (int b = 0; b < 4; ++b) codes[4 * w + b] = ((word >> (8 * b)) >> shift) & 3;
    }
  } else {
    // weights 32j .. 32j+31: bytes 32*(j/2) ..., the low nibble for even j, the high one for odd j
    const unsigned int* words = reinterpret_cast<const unsigned int*>(qs + 32 * (j >> 1));
    const int shift = 4 * (j & 1);
#pragma unroll
    for (int w = 0; w < BLOCK / 4; ++w) {
      const unsigned int word = words[w];
#pragma unroll
      for (int b = 0; b < 4; ++b) codes[4 * w + b] = ((word >> (8 * b)) >> shift) & 0xF;
    }
  }
}

// One weight of a block, built to the depths its tokens read and added to their sums. With one depth for the tile it
// is rounded once, at that depth; otherwise at every depth some token reads.
template <int BLOCK, int BASE_DEPTH>
__device__ __forceinline__ void accumulate(
    float w, const float step, const int kk, const unsigned int (&planes)[MAX_REFINEMENTS][BLOCK / 16],
    const int levels, const bool one_depth, const int (&reads)[BATCH_TILE], const float (&xs)[BATCH_TILE][8],
    const int k, float (&acc)[BATCH_TILE]) {
  float plane_step = step * 0.25f;  // a power of two: exact
  if (one_depth) {
#pragma unroll
    for (int e = 0; e < MAX_REFINEMENTS; ++e)
      if (e < levels) {
        const int c = (planes[e][kk / 16] >> (2 * (kk % 16))) & 3;
        w = __fadd_rn(w, __fmul_rn(plane_step, (float)c - 2.0f + 0.5f));
        plane_step *= 0.25f;
      }
    const float wb = round_bf16(w);
#pragma unroll
    for (int t = 0; t < BATCH_TILE; ++t)
      if (reads[t] > 0) acc[t] = __fmaf_rn(wb, xs[t][k], acc[t]);
    return;
  }
  float wb = round_bf16(w);
#pragma unroll
  for (int t = 0; t < BATCH_TILE; ++t)
    if (reads[t] == BASE_DEPTH) acc[t] = __fmaf_rn(wb, xs[t][k], acc[t]);
#pragma unroll
  for (int e = 0; e < MAX_REFINEMENTS; ++e)
    if (e < levels) {
      const int c = (planes[e][kk / 16] >> (2 * (kk % 16))) & 3;
      w = __fadd_rn(w, __fmul_rn(plane_step, (float)c - 2.0f + 0.5f));
      plane_step *= 0.25f;
      wb = round_bf16(w);
#pragma unroll
      for (int t = 0; t < BATCH_TILE; ++t)
        if (reads[t] == BASE_DEPTH + e + 1) acc[t] = __fmaf_rn(wb, xs[t][k], acc[t]);
    }
}

template <bool Q2, int BLOCK, int BYTES, int BASE_DEPTH>
__device__ __forceinline__ void kquant_rows(
    const unsigned char* __restrict__ blocks, const unsigned char* __restrict__ refinements,
    const unsigned short* __restrict__ x, const unsigned char* __restrict__ depth, float* __restrict__ y,
    const int n_refinements, const int n_tokens, const int in_features, const int out_features, const int depth_stride) {
  __shared__ __align__(16) unsigned short tile[BATCH_TILE * X_CHUNK];  // the tokens' inputs, one chunk of columns
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int lane = threadIdx.x;
  const int thread = threadIdx.y * WARP + lane;
  const int threads = blockDim.y * WARP;
  const int token0 = blockIdx.y * BATCH_TILE;
  const bool active = row < out_features;  // a warp past the last row still loads its share of the inputs

  const int stored = BASE_DEPTH + n_refinements;
  const long long row_bytes = (long long)(in_features / QK_K) * BYTES;
  const long long refinement_row = (long long)in_features / 4;
  const long long refinement_plane = (long long)out_features * refinement_row;

  // The rows of a thread block lie in one block of TILE_ROWS: every warp reads the same depths.
  int reads[BATCH_TILE];  // the depth each token of the tile reads these rows to; 0 reads nothing
  int deepest = 0;
  bool one_depth = true;  // every token of the tile that reads at all reads the same depth: the common case
#pragma unroll
  for (int t = 0; t < BATCH_TILE; ++t) {
    const int token = token0 + t;
    int d = 0;
    if (token < n_tokens) {
      d = depth[(long long)token * depth_stride + (blockIdx.x * blockDim.y) / TILE_ROWS];
      if (d > 0) d = min(max(d, BASE_DEPTH), stored);  // never shallower than the base, never deeper than stored
    }
    reads[t] = d;
    if (d > 0 && deepest > 0 && d != deepest) one_depth = false;
    deepest = max(deepest, d);
  }

  float acc[BATCH_TILE];
#pragma unroll
  for (int t = 0; t < BATCH_TILE; ++t) acc[t] = 0.0f;

  if (deepest > 0) {  // the same for the whole thread block, so every thread reaches every barrier
    const int levels = deepest - BASE_DEPTH;  // refinements this tile reads
    for (int c0 = 0; c0 < in_features; c0 += X_CHUNK) {
      const int width = min(X_CHUNK, in_features - c0);
      // the whole thread block loads the chunk: 8 inputs a thread at a time, a token with nothing to read left out
      for (int i = thread * 8; i < BATCH_TILE * width; i += threads * 8) {
        const int t = i / width, c = i % width;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (reads[t] > 0) v = *reinterpret_cast<const uint4*>(x + (long long)(token0 + t) * in_features + c0 + c);
        *reinterpret_cast<uint4*>(tile + t * X_CHUNK + c) = v;
      }
      __syncthreads();
      if (active) {
        for (int b = c0 / BLOCK + lane; b < (c0 + width) / BLOCK; b += WARP) {
          const int super_block = b / (QK_K / BLOCK);
          const int j = b % (QK_K / BLOCK);
          const int column0 = super_block * QK_K + j * BLOCK;
          float step, offset;
          unsigned char codes[BLOCK];
          decode<Q2, BLOCK, BYTES>(blocks + row * row_bytes + (long long)super_block * BYTES, j, step, offset, codes);
          unsigned int planes[MAX_REFINEMENTS][BLOCK / 16];  // BLOCK weights of a plane: BLOCK / 4 bytes
#pragma unroll
          for (int e = 0; e < MAX_REFINEMENTS; ++e)
            if (e < levels) {
              const unsigned int* p = reinterpret_cast<const unsigned int*>(
                  refinements + e * refinement_plane + row * refinement_row + column0 / 4);
#pragma unroll
              for (int w = 0; w < BLOCK / 16; ++w) planes[e][w] = p[w];
            }
#pragma unroll
          for (int chunk = 0; chunk < BLOCK / 8; ++chunk) {
            float xs[BATCH_TILE][8];
#pragma unroll
            for (int t = 0; t < BATCH_TILE; ++t) load8(tile + t * X_CHUNK + column0 - c0 + 8 * chunk, xs[t]);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
              const int kk = 8 * chunk + k;  // the weight's place in its block
              const float w = __fsub_rn(__fmul_rn(step, (float)codes[kk]), offset);
              accumulate<BLOCK, BASE_DEPTH>(w, step, kk, planes, levels, one_depth, reads, xs, k, acc);
            }
          }
        }
      }
      __syncthreads();
    }
  }

  if (!active) return;
#pragma unroll
  for (int t = 0; t < BATCH_TILE; ++t) {
    float v = acc[t];
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    if (lane == 0 && token0 + t < n_tokens) y[(long long)(token0 + t) * out_features + row] = v;
  }
}

#define ARGUMENTS                                                                                                 \
  const unsigned char* __restrict__ blocks,       /* [out, in / QK_K, block bytes] */                             \
  const unsigned char* __restrict__ refinements,  /* [n_refinements, out, in / 4] */                              \
  const unsigned short* __restrict__ x,           /* [n_tokens, in] bf16 */                                       \
  const unsigned char* __restrict__ depth,        /* [n_tokens or 1, out / TILE_ROWS]: the depth of each block */ \
  float* __restrict__ y,                          /* [n_tokens, out] */                                           \
  const int n_refinements, const int n_tokens, const int in_features, const int out_features,                    \
  const int depth_stride                          /* 0: one layout for every token; out / TILE_ROWS: one each */

extern "C" __global__ void kquant_matmul_q2_k(ARGUMENTS) {
  kquant_rows<true, 16, 84, 1>(blocks, refinements, x, depth, y, n_refinements, n_tokens, in_features, out_features,
                               depth_stride);
}

extern "C" __global__ void kquant_matmul_q4_k(ARGUMENTS) {
  kquant_rows<false, 32, 144, 2>(blocks, refinements, x, depth, y, n_refinements, n_tokens, in_features, out_features,
                                 depth_stride);
}
