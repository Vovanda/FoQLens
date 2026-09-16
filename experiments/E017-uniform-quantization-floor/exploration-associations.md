# Associations to plain words at two bits

2026-09-16, gemma-4-E2B-it. Exploration of E017 before its preregistration: twelve plain words, for each the request «Write five words
you associate with "<word>". Only the words, separated by commas.», greedy decoding, up to 40 tokens, the first
line of the reply. The controlled weights are replaced by a dequantized copy and read at bf16.

| Word | Our D2 | Q2_K bartowski | UD-Q2_K_XL unsloth |
| --- | --- | --- | --- |
| cat | Highest: | purr, sleek, curious, playful, soft | Feline, purr, sleek, playful, soft |
| dog | ... | Loyal, playful, wagging, bark, canine | Loyal, playful, energetic, furry, companion |
| house | `** $\text{...**$ ** $\quad**` | brick, roof, walls, door, home | Roof, walls, door, structure, home |
| water | sings. | liquid, flow, ocean, dew, wet | Liquid, clear, flow, wet, life |
| sun | `**Delta** ** ** ****` | bright, yellow, warm, golden, light | Bright, warm, radiant, golden, light |
| tree | `Structure 望im... �nong...` | wood, trunk, roots, branches, forest | Wood, roots, branches, leaves, trunk |
| bread | `Summer is tallest... $\s$` | wheat, crust, sourdough, yeast, grain | Yeast, grain, crust, baked, wholesome |
| book | `ed. 望a. / / / /` | story, pages, chapters, novel, read | read, story, pages, knowledge, imagination |
| doctor | ... | stethoscope, diagnosis, patient, doctor, medical | Healer, physician, diagnosis, expertise, medical. |
| winter | `Shares 望gruouslys ... ** **` | snow, ice, frost, cold, frosty | Snowy, cold, ice, frost, white |
| кошка (cat) | `$sings$sings $\textcposal$` | cat, purr, sleek, graceful, feline | Feline, graceful, sleek, mysterious, playful. |
| собака (dog) | `BLIM / / ... *s$` | bark, loyal, playful, energetic, canine | Loyal, playful, energetic, obedient, canine. |

**Our D2 broke the model.** On all twelve words: fragments of markup, symbols and hieroglyphs, not one association.
The ready two-bit files of the same checkpoint answer sensibly on all twelve, in Russian too (with English
associations).

**Nearly the same budget, another mechanics.** Our D2 quantizes only the controlled modules (attention, MLP,
per-layer), the embeddings stay at bf16. On the same modules the GGUF files hold mostly 2-3 bits:

| | 2 bits | 3 bits | 4 bits and more |
| --- | --- | --- | --- |
| Q2_K bartowski | 61% (Q2_K) | 35% (Q3_K) | 4% |
| UD-Q2_K_XL unsloth | 55% (Q2_K, IQ2_S, IQ2_XS) | 39% (Q3_K, IQ3_S, IQ3_XXS) | 7% |

Over all the weights of a file the share of two bits is smaller (25% and 22%): the embeddings are at Q6_K and
Q4_K. Our D2 is every controlled weight at two bits: a symmetric step of amax/2 per group of 64, with no zero and
no search of the step; Q2_K is a scale and a minimum per block of 16, searched, with the scales themselves
quantized per super-block of 256. The model need not break at this budget.

**What follows.** The way the bench acts on the network has to be revised. If our mechanics breaks D2, it may
partly damage D4 too: then D4's loss in E016 (14.3% of bf16's knowledge) measures our way of quantizing.
This is an indirect sign. The next step is our D4 against Q4_K of the same checkpoint on
the same questions. The probe of module classes ([the module classes exploration](exploration-module-classes.md))
adds a caveat: an asymmetric scale searched per group of 16, every weight at two bits, teaches D2 to stop but does
not bring the knowledge back (EM 0.048) - at two bits both the mechanics and which classes stay higher decide.
