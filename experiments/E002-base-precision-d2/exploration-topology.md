# The topology of the published GGUF on our mechanics

2026-09-16, gemma-4-E2B-it. Exploration of E002 before its preregistration: every controlled tensor gets the bits
its GGUF type holds (Q2_K and IQ2 - 2, Q3_K and IQ3 - 3, Q4_K - 4, F16 - bf16) and is read by one of three
mechanics: (a) our slices, 3 bits rounded up to D4; (b) our slices, 3 bits rounded down to D2; (c) a scale and an
offset searched per group of 16, as in [the module classes exploration](exploration-module-classes.md), at exactly
the file's bits. The weights are replaced by a dequantized copy and read at bf16. Measured on the same 600 kept
questions of TriviaQA and NQ-open (short-0) and the same twelve words as in [the associations](exploration-associations.md).

On the modules we quantize, bartowski's Q2_K holds 61% of the weights at 2 bits, 35% at 3, 4% at 4 and more;
unsloth's UD-Q2_K_XL 55% at 2, 40% at 3, 4% at 4 and 1.5% at bf16.

| Topology | Mechanics | Stops on its own | EM | Weight error | Code bits a weight |
| --- | --- | --- | --- | --- | --- |
| bartowski Q2_K | our slices, 3 bits → D4 | 0.73 | 0.000 | 0.356 | 2.84 |
| bartowski Q2_K | our slices, 3 bits → D2 | 0.79 | 0.000 | 0.405 | 2.14 |
| bartowski Q2_K | asymmetric g16, the file's bits | 0.995 | 0.197 | 0.233 | 2.49 |
| unsloth UD-Q2_K_XL | our slices, 3 bits → D4 | 1.00 | 0.017 | 0.335 | 3.08 |
| unsloth UD-Q2_K_XL | our slices, 3 bits → D2 | 0.78 | 0.000 | 0.403 | 2.29 |
| unsloth UD-Q2_K_XL | asymmetric g16, the file's bits | 1.00 | 0.355 | 0.225 | 2.69 |

References from the module classes exploration: asymmetric g16 at 2 bits everywhere - EM 0.048; bartowski's file
itself - 0.382 (error 0.251), unsloth's file itself - 0.422 (0.235); bf16 - 0.672.

**No topology saves our slices.** Even at 3.08 code bits a weight EM is 0.017: unsloth's topology teaches our D2 to
stop and name the word ("cat", "dog, dog", "liquid, flow, sea"), not to answer.

**The topology saves the asymmetric grid.** At 2 bits everywhere it reaches EM 0.048, with unsloth's topology
0.355 - 84% of its file and 53% of bf16; the associations are sensible on all twelve words, on «кошка» in Russian
(«мурчание, пушистый, лапки, умиление, хищник»), which neither file did. With bartowski's topology - 0.197, half of
its file.

**The mechanics and the topology decide together.** Either alone is garbage or near zero. unsloth's topology beats
bartowski's at 0.2 bits more; they differ by 1.5% of the weights at bf16 and the share of 3-4 bits.

**Our weight error is no worse than the files'.** It is lower or equal (0.233 against 0.251, 0.225
against 0.235), yet EM is lower. What the files take the rest with - a step search weighted by importance
(imatrix), IQ types or something else - is not checked.

"Code bits" do not count the scales: our group of 16 stores an fp16 scale and offset, about +2 bits a weight, while
k-quants quantize their scales. Compared with the files by memory only once the scales are quantized too.
