from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import TypeVar

T = TypeVar("T")


def traverse_graph(
    graph: dict[str, list[str]], nodes: Mapping[str, T]
) -> Iterator[tuple[str, T, list[str], list[str]]]:
    """Traverse the graph in topological order, starting from the nodes
    with no predecessors.

    Example:
        >>> graph = {"a": ["b"], "b": []}
        >>> nodes = {"a": 1, "b": 2}
        >>> for name, value, preds, rem in traverse_graph(graph, nodes):
        ...     print(name, value, preds, rem)
        b 2 [] ['a']
        a 1 ['b'] []

    Args:
        graph: Graph represented as a dictionary of predecessors. Keys
            are node names, values are node predecessors.
        nodes: Dictionary mapping node names to values.

    Yields:
        Tuples containing node name, node value, node predecessors, and
        remaining unprocessed nodes.

    Raises:
        RuntimeError: If the graph is malformed.

    """
    # sort the set for consistent behavior
    unprocessed_nodes = sorted(set(nodes.keys()))
    processed: set[str] = set()

    graph = deepcopy(graph)
    while unprocessed_nodes:
        unprocessed_nodes_copy = unprocessed_nodes.copy()
        for node_name in unprocessed_nodes_copy:
            node_dependencies = graph[node_name]
            if not node_dependencies or all(
                dependency in processed for dependency in node_dependencies
            ):
                unprocessed_nodes.remove(node_name)
                yield (
                    node_name,
                    nodes[node_name],
                    node_dependencies,
                    unprocessed_nodes.copy(),
                )
                processed.add(node_name)

        if unprocessed_nodes_copy == unprocessed_nodes:
            raise RuntimeError(
                "Malformed graph. "
                "Please check that all nodes are connected in a directed acyclic graph."
            )


def is_acyclic(graph: dict[str, list[str]]) -> bool:
    """Test if graph is acyclic.

    Args:
        graph: Graph represented as a dictionary of predecessors.

    Returns:
        ``True`` if graph is acyclic, ``False`` otherwise.

    Examples:
        >>> is_acyclic({"a": ["b"], "b": []})
        True
        >>> is_acyclic({"a": ["b"], "b": ["a"]})
        False
        >>> is_acyclic({})
        True

    """
    graph = graph.copy()

    def dfs(node: str, visited: set[str], recursion_stack: set[str]) -> bool:
        visited.add(node)
        recursion_stack.add(node)

        for predecessor in graph.get(node, []):
            if predecessor in recursion_stack:
                return True
            if predecessor not in visited and dfs(
                predecessor, visited, recursion_stack
            ):
                return True

        recursion_stack.remove(node)
        return False

    visited: set[str] = set()
    recursion_stack: set[str] = set()

    for node in graph:
        if node not in visited and dfs(node, visited, recursion_stack):
            return False

    return True
