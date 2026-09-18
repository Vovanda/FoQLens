// The k-quant copy multiplied on tensor cores (mma.sync m16n8k16, bf16 in, fp32 accumulate): the same weights as
// kquant_matmul.cu, bit for bit, decoded straight into the A fragment instead of being multiplied on CUDA cores.
//
// A thread block covers 16 rows - the M of one mma, inside one block of TILE_ROWS, so one depth per token - and
// TOKENS tokens. Its warps split the input (split-K): warp k takes super-blocks k, k + split, ...; a decoding step
// has few tokens, so without the split a module would give the card too few warps to hide its reads. A warp stages
// its rows' base blocks and refinement planes of a super-block in its own part of shared memory and takes the
// super-block in 16 steps of K = 16. A step is one block of Q2_K, Q3_K or Q6_K or half a Q4_K block, so a row has
// one scale (and one min) in it. Per step a thread builds the eight weights of its A fragment - rows g and g+8, columns 2q, 2q+1, 2q+8,
// 2q+9 (g = lane / 4, q = lane % 4) - once, refines them up to the deepest depth any token reads, and at every depth
// some token reads issues the mma over all token tiles, the B fragment read straight from the input in global memory;
// a token that reads another depth enters it with a zero input, so it adds exact zeros. Depth 0 is ZERO and reads
// nothing. At the end the warps' partial sums are added in the order of the warps.
//
// A weight is built as the torch path builds it - base = d*scale*code - dmin*min, then + step/4^k * (code - 1.5) per
// refinement, every operation rounded on its own - and rounded to bf16; a product of two bf16 is exact in fp32, and an
// mma adds its products with truncation (Fasi et al. 2021). A token's column of an mma depends on its own inputs only,
// and the order of every sum is fixed by the shapes, so a token's output does not depend on the other tokens.

#include <cuda_fp16.h>

#define QK_K 256
#define WARP 32
#define TILE_ROWS 64                 // a FoQLens block of rows: the unit a depth is set on
#define ROWS 16                      // rows of a thread block: the mma's M
#define MAX_SPLIT 8                  // warps of a thread block at most, each over its share of the input
#define TOKENS 32                    // tokens of a thread block: 4 tiles of the mma's N = 8
#define TOKEN_TILES (TOKENS / 8)
#define STEPS (QK_K / 16)            // steps of K = 16 in a super-block
#define PLANE_WORDS (QK_K / 16)      // 32-bit words of one refinement plane over a super-block row: 2 bits a weight
#define QS_OFFSET 16                 // the codes follow 16 bytes in both blocks

__device__ __forceinline__ float round_bf16(float value) {  // float -> bf16 -> float, nearest even, as torch
  unsigned int u = __float_as_uint(value);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return __uint_as_float(u & 0xFFFF0000u);
}

__device__ __forceinline__ unsigned int bf16_pair(float low, float high) {  // two values already rounded to bf16
  return (__float_as_uint(low) >> 16) | (__float_as_uint(high) & 0xFFFF0000u);
}

__device__ __forceinline__ float from_fp16(const unsigned char* p) {
  return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}

__device__ __forceinline__ void mma(float (&c)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The bases the kernel reads, as ggml lays them out. Q2_K and Q4_K are asymmetric - d*scale*code - dmin*min - Q3_K and
// Q6_K symmetric - d*scale*(code - zero point) with a signed scale; kquant.from_gguf_blocks reads the same bytes.
enum Base { Q2K, Q3K, Q4K, Q6K };
#define Q3_K_SCALE_BIAS 32  // block_q3_K stores a signed 6-bit scale as scale + 32
#define Q3_K_ZERO 4         // a Q3_K code reads as code - 4
#define Q6_K_ZERO 32        // a Q6_K code reads as code - 32

// The super-block's multipliers of a row: d, and dmin of an asymmetric base, read once per super-block.
template <int B>
__device__ __forceinline__ void multipliers(const unsigned char* sb, float& d, float& dmin) {
  constexpr int D = B == Q2K ? 80 : B == Q4K ? 0 : B == Q3K ? 108 : 208;  // where d lies in the block
  d = from_fp16(sb + D);
  dmin = (B == Q2K || B == Q4K) ? from_fp16(sb + D + 2) : 0.0f;
}

// Four bytes of a row at columns 2q, 2q+1, 2q+8, 2q+9 of a 16-byte run: two 16-bit reads.
__device__ __forceinline__ void four_bytes(const unsigned char* run, int q, unsigned int (&b)[4]) {
  const unsigned int low = reinterpret_cast<const unsigned short*>(run)[q];
  const unsigned int high = reinterpret_cast<const unsigned short*>(run)[q + 4];
  b[0] = low & 0xFF;
  b[1] = low >> 8;
  b[2] = high & 0xFF;
  b[3] = high >> 8;
}

// The step and offset of a row's 16 weights at step s of a super-block, and their codes at the four columns a thread
// holds: c = 2q, 2q+1, 2q+8, 2q+9. Weight 16s + c of a super-block is weight 128h + 32g + j: h = s / 8,
// g = (s / 2) mod 4, j = 16 (s mod 2) + c.
template <int B>
__device__ __forceinline__ void decode_step(const unsigned char* sb, int s, int q, float d, float dmin, float& step,
                                            float& offset, float (&code)[4]) {
  const int h = s >> 3, g = (s >> 1) & 3, j0 = 16 * (s & 1);
  unsigned int b[4];
  if (B == Q2K) {
    // scales[16] (scale | min << 4), qs[64]: code 2 bits at 2g of qs[32h + j]
    step = __fmul_rn(d, (float)(sb[s] & 0xF));
    offset = __fmul_rn(dmin, (float)(sb[s] >> 4));
    four_bytes(sb + QS_OFFSET + 32 * h + j0, q, b);
#pragma unroll
    for (int k = 0; k < 4; ++k) code[k] = (float)((b[k] >> (2 * g)) & 3);
  } else if (B == Q4K) {
    // d, dmin, scales[12] (6-bit scales and mins of 8 blocks of 32), qs[128]: 4 bits of qs[32 (j / 2) + ...]
    const int j = s >> 1;
    const unsigned char* sc = sb + 4;
    int scale, minimum;
    if (j < 4) {
      scale = sc[j] & 63;
      minimum = sc[j + 4] & 63;
    } else {
      scale = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
      minimum = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    step = __fmul_rn(d, (float)scale);
    offset = __fmul_rn(dmin, (float)minimum);
    four_bytes(sb + QS_OFFSET + 32 * (j >> 1) + j0, q, b);
#pragma unroll
    for (int k = 0; k < 4; ++k) code[k] = (float)((b[k] >> (4 * (j & 1))) & 0xF);
  } else if (B == Q3K) {
    // hmask[32], qs[64], scales[12], d: 2 bits at 2g of qs[32h + j], the high bit at 4h + g of hmask[j]
    const unsigned char* sc = sb + 96;
    const int scale = (((sc[s & 7] >> (4 * (s >> 3))) & 0xF) | (((sc[8 + (s & 3)] >> (2 * (s >> 2))) & 3) << 4))
                      - Q3_K_SCALE_BIAS;
    step = __fmul_rn(d, (float)scale);
    offset = __fmul_rn(step, (float)Q3_K_ZERO);  // a power of two: exact, as torch's steps * zero point
    unsigned int m[4];
    four_bytes(sb + 32 + 32 * h + j0, q, b);
    four_bytes(sb + j0, q, m);
#pragma unroll
    for (int k = 0; k < 4; ++k) code[k] = (float)(((b[k] >> (2 * g)) & 3) | (((m[k] >> (4 * h + g)) & 1) << 2));
  } else {
    // ql[128], qh[64], scales[16] int8, d: weight 128h + 32r + j, r = 2p + c', reads nibble p of ql[64h + 32c' + j]
    // and, as its high two bits, bits 2r of qh[32h + j]
    step = __fmul_rn(d, (float)(signed char)sb[192 + s]);
    offset = __fmul_rn(step, (float)Q6_K_ZERO);
    const int r = g, p = r >> 1, c = r & 1;
    unsigned int m[4];
    four_bytes(sb + 64 * h + 32 * c + j0, q, b);
    four_bytes(sb + 128 + 32 * h + j0, q, m);
#pragma unroll
    for (int k = 0; k < 4; ++k) code[k] = (float)(((b[k] >> (4 * p)) & 0xF) | (((m[k] >> (2 * r)) & 3) << 4));
  }
}

// The 2-bit codes of a refinement at step s and the thread's four columns: the step's 16 codes are word s of the
// row's plane, weight c in bits 2c.
__device__ __forceinline__ void plane_codes(const unsigned int* plane, int s, int q, float (&code)[4]) {
  const unsigned int word = plane[s];
  code[0] = (float)((word >> (4 * q)) & 3);
  code[1] = (float)((word >> (4 * q + 2)) & 3);
  code[2] = (float)((word >> (4 * q + 16)) & 3);
  code[3] = (float)((word >> (4 * q + 18)) & 3);
}

// The weights at `depth`, if some token of the block reads it, times every token tile. B is read from the input:
// token g of tile n, columns 2q, 2q+1 and 2q+8, 2q+9 of the step; a token that reads another depth, or none, enters
// with zeros and its input is not read.
__device__ __forceinline__ void multiply_at(int depth, const float (&w)[8], unsigned int present,
                                            const int (&mine)[TOKEN_TILES], int tiles,
                                            const unsigned int* const (&xrow)[TOKEN_TILES], int column,
                                            float (&acc)[TOKEN_TILES][4]) {
  if (!((present >> depth) & 1u)) return;
  float wb[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) wb[k] = round_bf16(w[k]);
  const unsigned int a[4] = {bf16_pair(wb[0], wb[1]), bf16_pair(wb[4], wb[5]), bf16_pair(wb[2], wb[3]),
                             bf16_pair(wb[6], wb[7])};
#pragma unroll
  for (int n = 0; n < TOKEN_TILES; ++n)
    if (n < tiles) {
      unsigned int b0 = 0u, b1 = 0u;
      if (mine[n] == depth) {
        b0 = __ldg(xrow[n] + column);
        b1 = __ldg(xrow[n] + column + 4);
      }
      mma(acc[n], a, b0, b1);
    }
}

// The word a base block is copied in: 32 bits where its size allows, 16 where it does not (block_q3_K is 110 bytes,
// block_q6_K 210).
template <bool FOUR> struct Word { typedef unsigned int T; };
template <> struct Word<false> { typedef unsigned short T; };

template <int B, int BYTES, int BASE_DEPTH, int MAX_REFINEMENTS>
__device__ __forceinline__ void kquant_mma_rows(
    const unsigned char* __restrict__ blocks, const unsigned char* __restrict__ refinements,
    const unsigned short* __restrict__ x, const unsigned char* __restrict__ depth, float* __restrict__ y,
    const int n_refinements, const int n_tokens, const int in_features, const int out_features, const int depth_stride) {
  typedef typename Word<BYTES % 4 == 0>::T W;
  constexpr int WORDS = BYTES / sizeof(W);  // words of a base block
  // a warp's staging of one super-block, in dynamic shared memory sized to the warps the launch gives; after the loop
  // the same memory holds the warps' partial sums, so 8 warps stay within the 48 KiB a launch gets without opting in
  struct Stage {
    W base[ROWS][WORDS];
    unsigned int planes[MAX_REFINEMENTS][ROWS][PLANE_WORDS];
  };
  static_assert(sizeof(Stage) >= WARP * TOKEN_TILES * 4 * sizeof(float), "a warp's sums must fit in its stage");
  extern __shared__ __align__(16) unsigned char dynamic[];
  __shared__ int reads[TOKENS];  // the depth each token reads these rows to; 0 reads nothing

  const int lane = threadIdx.x, warp = threadIdx.y, split = blockDim.y;
  const int thread = warp * WARP + lane;
  const int g = lane >> 2, q = lane & 3;
  const int row0 = blockIdx.x * ROWS;
  const int token0 = blockIdx.y * TOKENS;
  const int stored = BASE_DEPTH + n_refinements;
  const int super_blocks = in_features / QK_K;
  const long long row_bytes = (long long)super_blocks * BYTES;
  const long long refinement_row = (long long)in_features / 4;
  const long long refinement_plane = (long long)out_features * refinement_row;
  Stage& stage = reinterpret_cast<Stage*>(dynamic)[warp];

  if (thread < TOKENS) {
    const int token = token0 + thread;
    int d = 0;
    if (token < n_tokens) {
      d = depth[(long long)token * depth_stride + row0 / TILE_ROWS];
      if (d > 0) d = min(max(d, BASE_DEPTH), stored);  // never shallower than the base, never deeper than stored
    }
    reads[thread] = d;
  }
  __syncthreads();
  unsigned int present = 0;  // a bit per depth some token reads: the same in every thread
#pragma unroll
  for (int t = 0; t < TOKENS; ++t) present |= 1u << reads[t];
  present &= ~1u;
  const int deepest = present ? 31 - __clz(present) : 0;
  const int levels = deepest > 0 ? deepest - BASE_DEPTH : 0;
  const int tiles = min(TOKEN_TILES, (n_tokens - token0 + 7) / 8);  // token tiles holding a token at all
  int mine[TOKEN_TILES];  // the depth of the token whose inputs this thread feeds to B: token g of every tile
  const unsigned int* xrow[TOKEN_TILES];  // that token's input as pairs of bf16, from column 2q
#pragma unroll
  for (int n = 0; n < TOKEN_TILES; ++n) {
    mine[n] = reads[n * 8 + g];
    const int token = min(token0 + n * 8 + g, n_tokens - 1);  // a token past the last one is never read
    xrow[n] = reinterpret_cast<const unsigned int*>(x + (long long)token * in_features) + q;
  }

  float acc[TOKEN_TILES][4];
#pragma unroll
  for (int n = 0; n < TOKEN_TILES; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[n][k] = 0.0f;

  if (present) {
    for (int sb = warp; sb < super_blocks; sb += split) {
      // the warp's rows of this super-block, read as whole words; a row past the last one is left out
      for (int i = lane; i < ROWS * WORDS; i += WARP) {
        const int r = i / WORDS, w = i % WORDS;
        if (row0 + r < out_features)
          stage.base[r][w] = reinterpret_cast<const W*>(blocks + (row0 + r) * row_bytes + sb * BYTES)[w];
      }
      for (int e = 0; e < levels; ++e)
        for (int i = lane; i < ROWS * PLANE_WORDS; i += WARP) {
          const int r = i / PLANE_WORDS, w = i % PLANE_WORDS;
          if (row0 + r < out_features)
            stage.planes[e][r][w] = reinterpret_cast<const unsigned int*>(
                refinements + e * refinement_plane + (row0 + r) * refinement_row + sb * (QK_K / 4))[w];
        }
      __syncwarp();
      const unsigned char* rows[2] = {reinterpret_cast<const unsigned char*>(stage.base[g]),
                                      reinterpret_cast<const unsigned char*>(stage.base[g + 8])};
      float d[2], dmin[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) multipliers<B>(rows[h], d[h], dmin[h]);
#pragma unroll 4
      for (int s = 0; s < STEPS; ++s) {
        const int column = (sb * QK_K + 16 * s) / 2;  // in pairs of bf16: this step's column 0
        // the thread's weights: [0..3] row g, [4..7] row g+8, at columns 2q, 2q+1, 2q+8, 2q+9
        float w[8], plane_step[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float step, offset, code[4];
          decode_step<B>(rows[h], s, q, d[h], dmin[h], step, offset, code);
#pragma unroll
          for (int k = 0; k < 4; ++k) w[4 * h + k] = __fsub_rn(__fmul_rn(step, code[k]), offset);
          plane_step[h] = step * 0.25f;  // a power of two: exact
        }
        multiply_at(BASE_DEPTH, w, present, mine, tiles, xrow, column, acc);
#pragma unroll
        for (int e = 0; e < MAX_REFINEMENTS; ++e)
          if (e < levels) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              float code[4];
              plane_codes(stage.planes[e][g + 8 * h], s, q, code);
#pragma unroll
              for (int k = 0; k < 4; ++k)
                w[4 * h + k] = __fadd_rn(w[4 * h + k], __fmul_rn(plane_step[h], code[k] - 2.0f + 0.5f));
              plane_step[h] *= 0.25f;
            }
            multiply_at(BASE_DEPTH + e + 1, w, present, mine, tiles, xrow, column, acc);
          }
      }
      __syncwarp();  // the next super-block's staging overwrites what this one read
    }
  }

  // the warps' partial sums, [split][WARP][TOKEN_TILES * 4], added in the order of the warps
  __syncthreads();  // every warp is done with its stage
  float* sums = reinterpret_cast<float*>(dynamic);
#pragma unroll
  for (int n = 0; n < TOKEN_TILES; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k) sums[(warp * WARP + lane) * TOKEN_TILES * 4 + n * 4 + k] = acc[n][k];
  __syncthreads();
  if (warp != 0) return;
#pragma unroll
  for (int n = 0; n < TOKEN_TILES; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float total = sums[lane * TOKEN_TILES * 4 + n * 4 + k];
      for (int other = 1; other < split; ++other) total += sums[(other * WARP + lane) * TOKEN_TILES * 4 + n * 4 + k];
      // C: row g holds tokens 2q, 2q+1 in [0], [1]; row g+8 in [2], [3]
      const int row = row0 + g + 8 * (k >> 1);
      const int token = token0 + n * 8 + 2 * q + (k & 1);
      if (row < out_features && token < n_tokens) y[(long long)token * out_features + row] = total;
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

extern "C" __global__ void kquant_mma_q2_k(ARGUMENTS) {
  kquant_mma_rows<Q2K, 84, 1, 3>(blocks, refinements, x, depth, y, n_refinements, n_tokens, in_features, out_features,
                                  depth_stride);
}

extern "C" __global__ void kquant_mma_q4_k(ARGUMENTS) {
  kquant_mma_rows<Q4K, 144, 2, 2>(blocks, refinements, x, depth, y, n_refinements, n_tokens, in_features,
                                    out_features, depth_stride);
}

// Bases read from a published GGUF file (kquant.Q3_K, kquant.Q6_K): a copy over bartowski's Q2_K holds 70 modules on
// Q3_K and 17 on Q6_K.
extern "C" __global__ void kquant_mma_q3_k(ARGUMENTS) {
  kquant_mma_rows<Q3K, 110, 1, 3>(blocks, refinements, x, depth, y, n_refinements, n_tokens, in_features,
                                  out_features, depth_stride);
}

extern "C" __global__ void kquant_mma_q6_k(ARGUMENTS) {
  kquant_mma_rows<Q6K, 210, 3, 1>(blocks, refinements, x, depth, y, n_refinements, n_tokens, in_features,
                                  out_features, depth_stride);
}
