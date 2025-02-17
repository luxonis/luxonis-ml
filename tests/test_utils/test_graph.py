from typing import Dict, List, Tuple

import pytest

from luxonis_ml.utils import is_acyclic, traverse_graph


@pytest.mark.parametrize(
    ("graph", "acyclic"),
    [
        ({}, True),
        ({"a": []}, True),
        ({"a": ["b"], "b": ["a"]}, False),
        ({"a": ["b"], "b": []}, True),
        ({"a": ["b"], "b": ["c"], "c": ["a"]}, False),
        ({"a": ["b"], "b": ["c"], "c": []}, True),
        ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}, True),
        ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": ["a"]}, False),
    ],
)
def test_acyclic(graph: Dict[str, List[str]], acyclic: bool):
    assert is_acyclic(graph) == acyclic


@pytest.mark.parametrize(
    ("graph", "nodes", "expected"),
    [
        ({}, {}, []),
        (
            {"a": []},
            {"a": 1},
            [("a", 1, [], [])],
        ),
        (
            {"a": ["b"], "b": []},
            {"a": 1, "b": 2},
            [("b", 2, [], ["a"]), ("a", 1, ["b"], [])],
        ),
        (
            {"a": ["b"], "b": ["c"], "c": []},
            {"a": 1, "b": 2, "c": 3},
            [
                ("c", 3, [], ["a", "b"]),
                ("b", 2, ["c"], ["a"]),
                ("a", 1, ["b"], []),
            ],
        ),
        (
            {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []},
            {"a": 1, "b": 2, "c": 3, "d": 4},
            [
                ("d", 4, [], ["a", "b", "c"]),
                ("b", 2, ["d"], ["a", "c"]),
                ("c", 3, ["d"], ["a"]),
                ("a", 1, ["b", "c"], []),
            ],
        ),
    ],
)
def test_traverse(
    graph: Dict[str, List[str]],
    nodes: Dict[str, int],
    expected: List[Tuple[str, int, List[str], List[str]]],
):
    result = list(traverse_graph(graph, nodes))
    assert result == expected


@pytest.mark.parametrize(
    ("graph", "nodes"),
    [
        ({"a": ["b"], "b": ["a"]}, {"a": 1, "b": 2}),
        (
            {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": ["a"]},
            {"a": 1, "b": 2, "c": 3, "d": 4},
        ),
    ],
)
def test_traverse_fail(graph: Dict[str, List[str]], nodes: Dict[str, int]):
    with pytest.raises(RuntimeError):
        list(traverse_graph(graph, nodes))
