# Which module classes hold knowledge at two bits

2026-09-16, gemma-4-E2B-it. Exploration of E017 before its preregistration: 600 kept questions of TriviaQA and NQ-open (short-0),
the controlled weights replaced by a dequantized 2-bit copy and read at bf16. EM of bf16 is 0.672.

**The mechanics lets D2 stop, not answer.** Our D2 (symmetric, step amax/2, group 64) stops on its own in 26%
of the answers and answers nothing. A scale and an offset searched per group, as k-quants do, stops in 92-99%;
at group 16 the weight error is near Q2_K's (0.264 against 0.251), but EM is 0.048 against 0.382.

**The layout holds the knowledge.** The floor at 2 bits (asymmetric, group 16), one class at bf16:

| At bf16 | Share of weights | EM |
| --- | --- | --- |
| nothing | 0 | 0.048 |
| q_proj / k_proj | 7% / <1% | 0.048 / 0.045 |
| v_proj | <1% | 0.092 |
| o_proj | 7% | 0.105 |
| gate_proj / up_proj / down_proj | 28% each | 0.093 / 0.077 / 0.063 |
| per_layer_input_gate / per_layer_projection | 1% each | 0.052 / 0.093 |
| v, o, down (bartowski's raised classes) | 35% | 0.245 |
| k, v, o, down, per-layer (unsloth's) | 37% | 0.413 |
| first 5 layers / last 5 layers | 10% / 17% | 0.068 / 0.060 |

- No class alone brings the knowledge back; together they bring back more than the sum of each alone: k and the per-layer
  modules, 2% of the weights, add 0.17 on top of v, o and down.
- The classes that write into the residual stream (o_proj, per_layer_projection) and v_proj matter most per
  weight; q_proj and k_proj alone do nothing.
- unsloth's classes at bf16 reach its GGUF (0.413 against 0.422), where those classes are at 3-4 bits.

**For the filter.** Sensitivity by class is knowledge that does not depend on the query. It can set the floor
by class, so that zones raise precision only where this query needs it. And it is a control: zones have to
beat "raise the sensitive classes" at the same memory. If they do not, what they find is fragile modules.

The group-16 floor stores an fp16 scale and offset per 16 weights, about 4 bits a weight; Q2_K quantizes its
scales and costs 2.625. Compared by memory only once the scales are quantized too.
