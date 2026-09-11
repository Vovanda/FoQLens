"""Topic pairs of the mask experiments: a polar pair and a hard pair (see docs/data-sources.md)."""

PAIRS = {"biology": "math", "math": "biology", "history": "geography", "geography": "history"}
SPECS = {"biology": "biology", "math": "math", "history": "heldout/history", "geography": "heldout/geography"}
PAIR_NAMES = (("biology", "math"), ("history", "geography"))
