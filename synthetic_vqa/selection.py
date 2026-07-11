from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List

from .models import Sample


SMOKE_BO3_QUOTA = {0: 2, 1: 6, 2: 8, 3: 4}


def select_by_bo3(samples: List[Sample], quota: Dict[int, int], seed: int) -> List[Sample]:
    buckets = {score: [] for score in quota}
    for sample in samples:
        if sample.bo3 is None:
            raise ValueError("All samples must be scored before selection")
        if sample.bo3 in buckets:
            buckets[sample.bo3].append(sample)

    rng = random.Random(seed)
    selected: List[Sample] = []
    for score in sorted(quota):
        bucket = sorted(buckets[score], key=lambda item: item.source_id)
        rng.shuffle(bucket)
        need = quota[score]
        if len(bucket) < need:
            raise ValueError("BO3=%d needs %d rows but only %d are available" % (score, need, len(bucket)))
        selected.extend(bucket[:need])
    selected.sort(key=lambda item: item.id)
    return selected


def bo3_counts(samples: List[Sample]) -> Dict[int, int]:
    counts = Counter(sample.bo3 for sample in samples)
    return {score: int(counts.get(score, 0)) for score in range(4)}

