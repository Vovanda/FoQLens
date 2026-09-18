// A linear layer read straight from its refined copy: a row of weights is fetched only to the depth its
// block is read at, so the bytes that cross the memory bus follow the layout (issue #3).
//
// The stored copy is quant.RefinedWeight: depth e holds DEPTH_BITS-bit codes of the residual left by
// the depths before it, four codes to a byte, the first code in the low bits. A weight read to depth
// d is scale * sum over e < d of (code_e - ZERO + 0.5) * LEVELS^-e, so a row needs only its first d
// depths - which is the whole point: a shallow block never touches the deep refinements.
//
// One thread owns one (row, batch item) pair and walks the input once. There is no shared memory and
// no barrier: a barrier inside a loop that only some threads enter is undefined behaviour, and the
// input row is read by every thread of the block, so L2 serves it.

#define DEPTH_BITS 2
#define CODES_PER_BYTE 4
#define LEVELS 4
#define ZERO 2
#define SCALE_GROUP 64      // one scale per this many weights of a row
#define TILE_ROWS 64        // a FoQLens block: the unit a level is set on
#define MAX_DEPTH 4

extern "C" __global__ void refined_matmul(
    const unsigned char* __restrict__ packed,   // [depths, out, in / CODES_PER_BYTE]
    const float* __restrict__ scale,            // [out, in / SCALE_GROUP]
    const float* __restrict__ x,                // [batch, in]
    const unsigned char* __restrict__ depth,    // [out / TILE_ROWS] the depth to read per block
    float* __restrict__ y,                      // [batch, out]
    const int batch, const int in_features, const int out_features) {
  const int row = blockIdx.x * TILE_ROWS + threadIdx.y;
  const int b = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= out_features || b >= batch) return;

  const int read = depth[blockIdx.x];
  if (read == 0) {                            // a block at ZERO reads nothing at all
    y[(long long)b * out_features + row] = 0.0f;
    return;
  }

  const long long row_bytes = in_features / CODES_PER_BYTE;
  const long long depth_bytes = (long long)out_features * row_bytes;
  const unsigned char* row_codes[MAX_DEPTH];
  for (int e = 0; e < read; ++e) row_codes[e] = packed + e * depth_bytes + (long long)row * row_bytes;

  const float* xb = x + (long long)b * in_features;
  const float* row_scale = scale + (long long)row * (in_features / SCALE_GROUP);
  float acc = 0.0f;

  for (int byte = 0; byte < row_bytes; ++byte) {
    const int col = byte * CODES_PER_BYTE;
    unsigned char code_bytes[MAX_DEPTH];
    for (int e = 0; e < read; ++e) code_bytes[e] = row_codes[e][byte];

    const float s = row_scale[col / SCALE_GROUP];
    for (int c = 0; c < CODES_PER_BYTE; ++c) {
      float w = 0.0f, step = 1.0f;
      for (int e = 0; e < read; ++e) {
        const int code = (code_bytes[e] >> (c * DEPTH_BITS)) & (LEVELS - 1);
        w += step * ((float)code - (float)ZERO + 0.5f);
        step /= (float)LEVELS;
      }
      acc += w * s * xb[col + c];
    }
  }
  y[(long long)b * out_features + row] = acc;
}
