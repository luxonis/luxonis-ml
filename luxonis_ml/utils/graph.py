from copy import deepcopy
from typing import Dict, Iterator, List, Mapping, Set, Tuple, TypeVar

T = TypeVar("T")


def traverse_graph(
    graph: Dict[str, List[str]], nodes: Mapping[str, T]
) -> Iterator[Tuple[str, T, List[str], List[str]]]:
    """Traverses the graph in topological order, starting from the nodes
    with no predecessors.

    Example:

        >>> graph = {"a": ["b"], "b": []}
        >>> nodes = {"a": 1, "b": 2}
        >>> for name, value, preds, rem in traverse_graph(graph, nodes):
        ...     print(name, value, preds, rem)
        b 2 [] ['a']
        a 1 ['b'] []

    @type graph: dict[str, list[str]]
    @param graph: Graph in a format of a dictionary of predecessors.
        Keys are node names, values are node predecessors (list of node
        names).
    @type nodes: dict[str, T]
    @param nodes: Dictionary mapping node names to values.
    @rtype: Iterator[tuple[str, T, list[str], list[str]]]
    @return: Iterator of tuples containing node name, node value, node
        predecessors, and remaining unprocessed nodes.
    @raises RuntimeError: If the graph is malformed.
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


def is_acyclic(graph: Dict[str, List[str]]) -> bool:
    """Tests if graph is acyclic.

    @type graph: dict[str, list[str]]
    @param graph: Graph in a format of a dictionary of predecessors.
    @rtype: bool
    @return: True if graph is acyclic, False otherwise.
    """
    graph = graph.copy()

    def dfs(node: str, visited: Set[str], recursion_stack: Set[str]) -> bool:
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
