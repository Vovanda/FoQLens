# Addendum 10 - the parameter names, chosen from a tier list

Fixed 2026-09-12. Not edited after its commit.

The names of ADDENDUM-09 were taken from the first word at hand. They are replaced after a tier list of candidates, judged on four criteria: the meaning is obvious from the word, the direction from 0 to 1 reads the natural way, no collision with a name already in the project or a common ML term, short.

| ADDENDUM-09 | Now | Why |
| --- | --- | --- |
| precision `p` | **precision_share** | the value is a share, 0 ... 1, like `gpu_share`; the bare `precision` collides with the precision of a block (`precision.py`, bits per weight) and with the precision/recall metric |
| spread `s` | **focus_area** | the area read sharp: 0 the zones' centers only, 0.5 the zones as found, 1 the whole weight map - no mask; a larger value is a wider area, it cannot be read the other way round |

The scales, the formulas (mean bits `4 + 4 precision_share`; `R = r focus_area / (1 - focus_area)`), the grid, the order, the criterion and the predictions M1-M3 of ADDENDUM-09 are unchanged; ADDENDUM-09 and its results read with this table.
