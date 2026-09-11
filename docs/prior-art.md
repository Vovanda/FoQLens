# What is already built - dynamic quantization

What dynamic quantization already has and how FoQLens differs from it. The list comes from a survey of the field on 2026-09-10 (section 5 of the author's private note on the precision controller), extended 2026-09-11.

## Common to all

The unit is fixed in advance (the whole model, layer, channel, token, expert) and precision is stepwise. They solve one problem: memory and speed under a given budget, with the signal taken from the data (entropy, sensitivity). **Nobody has precision allocated over the weights by the meaning of the query.**

## Switch the precision of the whole model

- **Any-Precision LLM** (Park et al., ICML 2024). One set of weights from which n-bit versions are read; the bit depth is chosen at runtime.
- **Matryoshka Quantization** (DeepMind, 2025). Nested int8/int4/int2 in the same weights: the high bits are the smaller model.
- **NestedFP** (arXiv 2506.02024). FP16 and FP8 in one set of weights.

## Precision per token or layer

- **MoBiQuant** (arXiv 2602.20191). Weights are cut into bit slices, an MLP router decides how many slices to read per token; a global threshold is tuned at runtime. 2/4/6 bits without repacking. The authors' finding - outlier migration: the set of sensitive tokens changes by itself when precision changes. Its residual quantization is the reference for step 5.
- **QuickSilver** (arXiv 2506.22396). Bit depth per token and layer by entropy: uncertain tokens keep full precision, confident ones are squeezed to 2 bits.
- **FlexQuant** (arXiv 2506.12024). Layer sensitivity offline via KL divergence, switching at runtime by perplexity entropy.

## Individual parts

- **D²MoE** (arXiv 2504.15299). Token-adaptive bit depth of experts in MoE.
- **GRINQH** (arXiv 2606.23419). Precision per activation channel by the magnitude of |x| against calibrated thresholds.
- **Adaptive KV-cache quantization** (arXiv 2604.04722). A trained MLP controller assigns 2/4/8/FP16 per token in the cache by frequency, attention variance and entropy uncertainty.

## What nobody has - the FoQLens territory

1. **An address, not only a scale.** Everyone decides "how many bits", nobody decides "where in the weights".
2. **The zone is not given by the architecture** but emerges from the distribution of precision.
3. **The address comes from the model's own intermediate activations**, without a separately trained router.
4. **Overlapping zones and isthmuses** as a measurable structure.
