# Results - uniform quantization with a working base precision D2

The D2 floor works: it keeps 51.4% of bf16's knowledge, and the filter can now be tested from D2. I measured as the
[preregistration](PREREG.md) says: the whole frozen corpus, the same judge as in E016.

## How I fixed D2

First I checked other people's models. I asked our D2, bartowski's Q2_K and unsloth's UD-Q2_K_XL for five
associations to each of twelve plain words, from "cat" to "собака". Our D2 gave scraps of markup and symbols on all
twelve, and both files answered sensibly, in Russian too ([associations](exploration-associations.md)). It became
obvious that we had broken the model at D2 ourselves, with our rounding. Then I took apart what these files keep
above two bits and how they quantize: the knowledge came back once the k, v, o, down and per-layer modules went above
two bits and the quantization method changed ([module classes](exploration-module-classes.md),
[topology](exploration-topology.md)).

I moved away from the MoBiQuant algorithm. My slices are still residual, but the first plane is now a k-quant base
after llama.cpp, where every block has a fitted scale and minimum, and I take the step of every slice from the step
of its block. The sensitive classes start at Q4_K. At D2 and D4 they read the same, and they get their first slice
over the base at D6. That puts the D2 floor at about 3.3 bits per weight on average, and it works.

## What each rung keeps

| Model | Excellent on known | Retention | Good | Bad | Incoherent | Excellent on unknown |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 (E016) | 91.9% | 100% | 3.4% | 3.7% | 0.9% | 8.1% |
| D8 | 90.8% | 98.8% | 3.3% | 4.5% | 1.3% | 11.0% |
| D6 | 89.0% | 96.8% | 3.5% | 5.9% | 1.5% | 12.5% |
| D4 | 83.6% | 91.0% | 3.5% | 11.0% | 1.9% | 17.2% |
| D2 | 47.2% | 51.4% | 3.3% | 29.0% | 20.5% | 22.2% |
| UD-Q2_K_XL unsloth | 73.5% | 80.0% | 2.8% | 20.3% | 3.4% | - |
| Q4_K_M unsloth | 87.3% | 95.0% | 3.4% | 7.7% | 1.6% | - |

Excellent is Correct and Nearly, good is Partial, bad is Wrong and Related, incoherent is Noise and Garbage. 18,576
known questions, 2,064 unknown.

Before the measurement I decided the floor had to keep at least half of bf16's knowledge
([why](../../notes/2026-09-17-floor-at-least-half.md)), and D2 reached that bar. D4 on the new method rose from 85.7%
to 91.0%; D6 and D8 barely moved.

Knowledge still goes from the weights first. At D4 the answer in the passage loses 2.5%, the answer only in the
weights 13.0%, HotpotQA 11.7%. At D2 it is 35.3%, 51.8% and 59.4%.

The published files keep more at the same rungs: 80.0% against my 51.4% at D2 and 95.0% against 91.0% at D4. They
are calibrated on data for one fixed precision. I built my copy so that its precision can change, and for testing the
filter that is enough: the zones and uniform quantization read the same copy.

On unknown questions the share of excellent answers grows with coarsening: 8.1% for bf16, 22.2% for D2. I saw this
trend in E016 as well; H4's premise rests on it. I have not read D2's answers to the unknown questions yet, so I do not
know yet how many are knowledge and how many match the reference by chance.

When D2 ran into the answer cap, I gave it 1,024 tokens instead of 512: 40 ARC-Challenge solutions reached the right
answer ([cut answers](exploration-reask.md)).
