# What I expect of FoQLens

2026-09-17. Volodya.

The goal is to surpass E2B bf16. An example on a corpus of 40,000 questions: bf16 knows half of them and does not know the other half. The rows bf16-D2 follow from the measured shares of excellent answers - Correct and Nearly by the judge (D8-D4 from E016, D2 from E017), the FoQLens rows are my expectations.

| Model | Known | Unknown | Answers of 40,000 | Share |
| --- | --- | --- | --- | --- |
| bf16 | 91.9% | 8.1% | 20,000 | 50.0% |
| D8 | 90.8% | 11.0% | 20,360 | 50.9% |
| D6 | 88.8% | 12.7% | 20,300 | 50.8% |
| D4 | 78.8% | 16.4% | 19,040 | 47.6% |
| D2 | 47.2% | 22.2% | 13,880 | 34.7% |
| FoQLens, base precision D2 - success | 90.8% | 29.2% | 24,000 | 60.0% |
| FoQLens, base precision D2 - beyond success | 90.8% | 49.2% | 28,000 | 70.0% |

Fixed models top out at about 50%. Success is 60%: at base precision D2 the model keeps D8 quality on the known questions and surpasses the base model on the whole corpus.

The catch: it would be very annoying if the zones work, but together with precision in the zones bf16's modesty comes back. The main hope is that the zones will not always be read at the highest precision.
