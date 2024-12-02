from typing import Any, Dict, List

import numpy as np


def batch2list(data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Convert from a batched target dict to list of normal target
    dicts."""
    batch_size = len(data["image_batch"])
    items = []
    for i in range(batch_size):
        item = {}
        for k, v in data.items():
            if k.endswith("_batch"):
                item_k = to_unbatched_name(k)
                item[item_k] = v[i]
            else:
                raise ValueError(
                    f"All key must have '_batch' suffix, got `{k}`"
                )
        items.append(item)
    return items


def list2batch(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert from a list of normal target dicts to a batched target
    dict."""

    if len(data) == 0:
        raise ValueError("The input should have at least one item.")

    item = data[0]
    batch: Dict[str, Any] = {f"{k}_batch": [] for k in item.keys()}
    for item in data:
        for k, v in item.items():
            batch_k = to_batched_name(k)
            batch[batch_k].append(v)

    return batch


def unbatch_all(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all names in the given dict from batched names to normal
    names and concatenates all lists."""
    new_data = {}
    for key, value in data.items():
        key = to_unbatched_name(key)
        if isinstance(value, np.ndarray):
            new_data[key] = value
        elif isinstance(value, list):
            if value and isinstance(value[0], (list, np.ndarray)):
                new_data[key] = np.concatenate(value)
            else:
                new_data[key] = np.array(value)
        else:
            new_data[key] = value

    return new_data


def batch_all(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all names in the given dict from normal names to batched
    names."""
    return {to_batched_name(k): v for k, v in data.items()}


def to_unbatched_name(batched_name: str) -> str:
    """Get a normal target name from a batched target name If the given
    name does not have "_batched" suffix, ValueError will be raised."""
    if not batched_name.endswith("_batch"):
        raise ValueError(
            f"Batched target name must have '_batch' suffix, got `{batched_name}`"
        )
    return batched_name.replace("_batch", "")


def to_batched_name(name: str) -> str:
    """Get a unbatched target name from a normal target name If the
    given name already has had "_batched" suffix, ValueError will be
    raised."""

    if name.endswith("_batch"):
        raise ValueError(
            f"Non batched target name must not have '_batch' suffix, got `{name}`"
        )
    return f"{name}_batch"
