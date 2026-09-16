# Why base precision has to keep at least half

2026-09-17. Volodya.

Before the first measurement of D2 I set myself a bar: base precision has to keep at least half of bf16's knowledge. I did not expect it to reach it - I rather thought it would fall short. But I wanted 50, so as to have less to put up with.

Why 50. Say a model earns +1 for a right answer and -1 for a wrong one. A model right half the time earns zero on average: no use, no harm. Below half that expectation goes negative. The judge has more than two grades, so +1/-1 is an engineering bound here: below it base precision does more harm than good. I need a base precision that does not drag the system below zero and stays a buffer before ZERO. On such a base precision it makes sense to raise precision with zones.

What came out on all 18,576 known questions of the frozen corpus:

| Judge's grade | D2 | bf16 |
| --- | --- | --- |
| Correct | 44.1% | 86.9% |
| Nearly | 3.1% | 5.0% |
| Partial | 3.3% | 3.4% |
| Related | 3.7% | 1.8% |
| Wrong | 25.2% | 1.9% |
| Noise | 9.0% | 0.3% |
| Garbage | 11.5% | 0.6% |

The judge accepts 47.2% of D2's answers against 91.9% of bf16's: a retention of 51.4%. For D2 I split the grades this way:
- excellent - right and nearly right (Correct, Nearly): 47.2%;
- good - partial (Partial): 3.3%, the answer is right but incomplete;
- bad - wrong and only on the topic (Wrong, Related): 28.9%;
- refusals and incoherent answers (Noise, Garbage): 20.5%.

Excellent and good together are 50.5%. The average result suits me well.
