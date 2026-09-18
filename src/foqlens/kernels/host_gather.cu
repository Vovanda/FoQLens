// Rows of a table that lives in the host's pinned memory, gathered onto the device: the per-layer embedding table of
// Gemma 4 (4.37 GiB for E2B) stays off the card, and a decoding step fetches only the rows of its tokens.
//
// Pinned host memory is mapped for the device under unified addressing, so the kernel reads the host pointer as it
// reads its own memory, over PCIe, with no copy issued by the host - which keeps it inside a CUDA graph. A thread
// block copies one row, 16 bytes a thread at a time.

extern "C" __global__ void host_gather(const uint4* __restrict__ table,     // [rows, row_words] in pinned host memory
                                       const long long* __restrict__ ids,   // [n] the rows to fetch, on the device
                                       uint4* __restrict__ out,             // [n, row_words] on the device
                                       const int row_words) {               // 16-byte words of a row
  const uint4* source = table + ids[blockIdx.x] * (long long)row_words;
  uint4* target = out + (long long)blockIdx.x * row_words;
  for (int i = threadIdx.x; i < row_words; i += blockDim.x) target[i] = source[i];
}
