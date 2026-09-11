# What is already built

What neighbouring fields already have and how FoQLens differs from it. Dynamic quantization comes from a survey of the field on 2026-09-10 (section 5 of the author's private note on the precision controller), extended 2026-09-11. Contextual sparsity, emergent modularity, static importance and per-input precision were added 2026-09-12; every arXiv id was checked against its abstract page. One line per work: the works are read in depth when the stage that needs them comes.

## Dynamic quantization

### Common to all

The unit is fixed in advance (the whole model, layer, channel, token, expert) and precision is stepwise. They solve one problem: memory and speed under a given budget, with the signal taken from the data (entropy, sensitivity).

### Switch the precision of the whole model

- **Any-Precision LLM** (Park et al., ICML 2024). One set of weights from which n-bit versions are read; the bit depth is chosen at runtime.
- **Matryoshka Quantization** (DeepMind, 2025). Nested int8/int4/int2 in the same weights: the high bits are the smaller model.
- **NestedFP** (arXiv 2506.02024). FP16 and FP8 in one set of weights.

### Precision per token or layer

- **MoBiQuant** (arXiv 2602.20191). Weights are cut into bit slices, an MLP router decides how many slices to read per token; a global threshold is tuned at runtime. 2/4/6 bits without repacking. The authors' finding - outlier migration: the set of sensitive tokens changes by itself when precision changes. Its residual quantization is the reference for step 5.
- **QuickSilver** (arXiv 2506.22396). Bit depth per token and layer by entropy: uncertain tokens keep full precision, confident ones are squeezed to 2 bits.
- **FlexQuant** (arXiv 2506.12024). Layer sensitivity offline via KL divergence, switching at runtime by perplexity entropy.

### Individual parts

- **D²MoE** (arXiv 2504.15299). Token-adaptive bit depth of experts in MoE.
- **GRINQH** (arXiv 2606.23419). Precision per activation channel by the magnitude of |x| against calibrated thresholds.
- **Adaptive KV-cache quantization** (arXiv 2604.04722). A trained MLP controller assigns 2/4/8/FP16 per token in the cache by frequency, attention variance and entropy uncertainty.

## Per-input precision - the closest competitors

- **DP-LLM** (Kwon et al., NeurIPS 2025, arXiv 2508.06041). Each layer's bit width is chosen at runtime from the input by an error estimator with fine-tuned thresholds. The nearest dense competitor: the unit is the layer and the selector is trained.
- **HOBBIT** (Tang et al., 2024, arXiv 2411.01433). In an MoE, less critical experts that miss the cache are replaced by low-precision copies per token.
- **PMPD** (Chen et al., 2024, arXiv 2410.13461). Precision lowered progressively along the decoded sequence; the unit is the phase, not a place in the weights.
- **DynaExq** (Chu et al., 2025, arXiv 2511.15015). Hot experts, estimated from router traces, get high precision - from aggregate traffic, not per input.

No work was found that gives different bits to different blocks inside a dense layer per query, by meaning, at a fixed mean-bits budget. The search covered the web and arXiv, not exhaustively.

## Contextual sparsity - which weights to compute per input

- **Deja Vu** (Liu et al., ICML 2023, arXiv 2310.17157). Per input, a trained lookahead predictor keeps some attention heads and MLP neurons and skips the rest.
- **PowerInfer** (Song et al., SOSP 2024, arXiv 2312.12456). Neuron activity follows a power law: hot neurons stay on the GPU, cold input-dependent ones go to the CPU, chosen by trained predictors. Hot neurons are a static importance split - the same signal as the backbone of FoQLens ([results](../experiments/E005-backbone/results.md)).
- **TEAL** (Liu et al., ICLR 2025, arXiv 2408.14690). Training-free sparsity of hidden states by magnitude, 40-50%.
- **CoreInfer** (Wang et al., 2024, arXiv 2410.18311). Sentence-level core neurons whose activation patterns follow the semantics, chosen from the model's own activations - the closest in spirit, but keep or drop instead of bits.
- **GRIFFIN** (Dong et al., 2024, arXiv 2404.01365). Training-free: feed-forward neurons chosen once per sequence from the prompt's own activations.
- Also: CATS (arXiv 2404.08763), R-Sparse (ICLR 2025, arXiv 2504.19449), LLM in a flash (ACL 2024, arXiv 2312.11514), ShadowLLM (EMNLP 2024, arXiv 2406.16635).

The difference from FoQLens: a weight is computed or skipped; there is no step between, and the units are fixed neurons or heads.

## Emergent modularity - do zones exist in dense models

- **MoEfication** (Zhang et al., ACL Findings 2022, arXiv 2110.01786). Feed-forward neurons split into experts, one of the ways by clustering the co-activation graph, and 10-30% of them routed. The closest to the weight map of FoQLens; the groups are fixed partitions inside a layer, a router is added, and the rest is skipped.
- **Emergent Modularity in Pre-trained Transformers** (Zhang et al., ACL 2023, arXiv 2305.18390). Neurons specialize and group into functional experts during pretraining.
- **EMoE** (Qiu et al., NAACL 2024, arXiv 2310.10908). A dense model turned into an MoE along these emergent groups, with no added parameters.
- **LLaMA-MoE** (Zhu et al., 2024, arXiv 2406.16554). The feed-forward layer partitioned into experts, then continued pretraining with a trained router.
- **Knowledge Neurons** (Dai et al., ACL 2022, arXiv 2104.08696), **language-specific neurons** (Tang et al., ACL 2024, arXiv 2402.16438), **query-relevant neurons** (Chen et al., AAAI 2025, arXiv 2406.10868). Facts, languages and domains located inside the weights - evidence that an address by meaning exists; nothing is compressed.

## Static importance - which weights matter for every query

- **AWQ** (Lin et al., MLSys 2024, arXiv 2306.00978), **OWQ** (Lee et al., AAAI 2024, arXiv 2306.02272), **SpQR** (Dettmers et al., 2023, arXiv 2306.03078). Salient channels and outlier weights, found with calibration activations, are protected. This is the generic importance that carries most of the budget in FoQLens ([results](../experiments/E005-backbone/results.md)).
- **SqueezeLLM** (Kim et al., ICML 2024, arXiv 2306.07629). Fisher (gradient) sensitivity drives non-uniform quantization - the static counterpart of the gradient mask of FoQLens.
- **HAWQ** (Dong et al., ICCV 2019, arXiv 1905.03696). Hessian-based bit widths per layer.
- **Wanda** (Sun et al., ICLR 2024, arXiv 2306.11695), **SparseGPT** (Frantar et al., 2023, arXiv 2301.00774). Static pruning by activations or second-order information.
- **Calibration data** (Williams and Aletras, ACL 2024, arXiv 2311.09755). Downstream quality of compressed models varies substantially with the calibration set - importance depends on the data.
- **D-Pruner** (Zhang et al., NAACL Findings 2024, arXiv 2405.06275). Pruning with a domain-specific calibration set, fixed offline.
- **Instruction-Following Pruning** (Hou et al., ICML 2025, arXiv 2501.02086). A mask predictor trained with the model picks parameters per instruction - per query, but trained, and skip instead of bits.

## Where FoQLens stands

1. **Per query, precision inside a layer.** Input-dependent decisions about a place in the model exist - channels (GRINQH), experts (HOBBIT), layers (DP-LLM), and which neurons to compute (contextual sparsity). What is not found: different bits for different blocks inside a dense layer, per query, by meaning, at a fixed mean-bits budget.
2. **Zones used for precision.** That dense models hold emergent co-activation modules is established (MoEfication, Emergent Modularity, EMoE). New is allocating precision by these zones, with a falloff around their centers instead of hard partitions.
3. **No trained router for precision.** Training-free, self-signalled selection exists for sparsity (GRIFFIN, TEAL, CoreInfer). For precision the known selectors are trained (DP-LLM) or routed (MoBiQuant); FoQLens takes the address from the model's own activations and gradients.
4. **Overlapping zones and isthmuses** as a measurable structure.
