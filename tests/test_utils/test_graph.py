import pytest
from hypothesis import given
from hypothesis import strategies as st

from luxonis_ml.utils import is_acyclic, traverse_graph


@st.composite
def dags(
    draw: st.DrawFn, min_nodes: int = 0, max_nodes: int = 8
) -> dict[str, list[str]]:
    """Build a random acyclic predecessor graph.

    Node ``i`` may only depend on nodes with a lower index, which makes
    a cycle impossible by construction. The keys are returned in a random
    order so that traversals cannot rely on insertion order.
    """
    names = [
        f"node{i}" for i in range(draw(st.integers(min_nodes, max_nodes)))
    ]

    graph = {
        name: draw(st.lists(st.sampled_from(names[:i]), unique=True))
        if i
        else []
        for i, name in enumerate(names)
    }
    return dict(draw(st.permutations(list(graph.items()))))


@st.composite
def cyclic_graphs(draw: st.DrawFn) -> dict[str, list[str]]:
    """Build a random graph containing at least one cycle."""
    graph = draw(dags(min_nodes=1))
    cycle = draw(
        st.lists(st.sampled_from(sorted(graph)), min_size=1, unique=True)
    )
    for node, successor in zip(cycle, [*cycle[1:], cycle[0]], strict=True):
        graph[node] = [*graph[node], successor]
    return graph


def index_nodes(graph: dict[str, list[str]]) -> dict[str, int]:
    return {name: i for i, name in enumerate(graph)}


@given(graph=dags())
def test_generated_dags_are_acyclic(graph: dict[str, list[str]]):
    assert is_acyclic(graph)


@given(graph=cyclic_graphs())
def test_cycles_are_detected(graph: dict[str, list[str]]):
    assert not is_acyclic(graph)


@given(graph=dags())
def test_traversal_visits_each_node_once_after_its_predecessors(
    graph: dict[str, list[str]],
):
    nodes = index_nodes(graph)
    visited: list[str] = []

    for name, value, predecessors, remaining in traverse_graph(graph, nodes):
        assert name not in visited
        assert value == nodes[name]
        assert predecessors == graph[name]
        assert set(predecessors) <= set(visited)

        visited.append(name)
        assert remaining == sorted(set(nodes) - set(visited))

    assert set(visited) == set(nodes)


@given(graph=dags())
def test_traversal_does_not_depend_on_dictionary_order(
    graph: dict[str, list[str]],
):
    nodes = index_nodes(graph)
    reversed_graph = dict(reversed(list(graph.items())))
    reversed_nodes = dict(reversed(list(nodes.items())))

    assert list(traverse_graph(graph, nodes)) == list(
        traverse_graph(reversed_graph, reversed_nodes)
    )


@given(graph=dags())
def test_traversal_leaves_the_graph_untouched(graph: dict[str, list[str]]):
    original = {
        name: list(predecessors) for name, predecessors in graph.items()
    }

    list(traverse_graph(graph, index_nodes(graph)))

    assert graph == original


@given(graph=cyclic_graphs())
def test_traversal_rejects_cyclic_graphs(graph: dict[str, list[str]]):
    with pytest.raises(RuntimeError, match="Malformed graph"):
        list(traverse_graph(graph, index_nodes(graph)))


@pytest.mark.parametrize(
    ("graph", "acyclic"),
    [
        ({}, True),
        ({"a": []}, True),
        ({"a": ["a"]}, False),
        ({"a": ["b"], "b": ["a"]}, False),
        ({"a": ["b"], "b": []}, True),
        ({"a": ["b"], "b": ["c"], "c": ["a"]}, False),
        ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}, True),
        ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": ["a"]}, False),
    ],
)
def test_acyclic(graph: dict[str, list[str]], acyclic: bool):
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
    graph: dict[str, list[str]],
    nodes: dict[str, int],
    expected: list[tuple[str, int, list[str], list[str]]],
):
    result = list(traverse_graph(graph, nodes))
    assert result == expected
