from collections import defaultdict
from typing import Any, Dict, Iterator, List


def yield_batches(
    data: List[Dict[str, Any]], batch_size: int
) -> Iterator[Dict[str, List[Any]]]:
    """Yield batches of data."""
    for i in range(0, len(data), batch_size):
        yield list2batch(data[i : i + batch_size])


def list2batch(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert from a list of normal target dicts to a batched target
    dict."""

    batch = defaultdict(list)
    for item in data:
        for k, v in item.items():
            batch[k].append(v)

    return dict(batch)
